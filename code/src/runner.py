from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import sys,os
import yaml
import random
from models.learning.model import Model
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import log_loss, mean_squared_error, mean_squared_log_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from typing import Callable, List, Optional, Tuple, Union
from kfold import MovingWindowKFold
from util import Logger, Util
from src.params import Cv, KFoldParams, StratifiedKFoldParams, GroupKFoldParams, TimeSeriesSplitParams, HoldOutParams, Setting, PreprocessingParams, ModelParams

# 定数
shap_sampling = 10000
corr_sampling = 10000
class Runner:

    def __init__(self
                , model_cls: Callable[[str, dict], Model]
                , features: List[str]
                , setting: Setting
                , params: ModelParams
                , cv: Cv
                , feature_dir_name: str
                , out_dir_name: str):
        """コンストラクタ
        :run_name: runの名前
        :model_cls: モデルのクラス
        :features: 特徴量のリスト
        :setting: 設定リスト
        :params: ハイパーパラメータ
        :cv: CVの設定
        :feature_dir_name: 特徴量を読み込むディレクトリ
        :out_dir_name: 学習に使用するファイルを保存するディレクトリ
        """
        self.task_type = setting.task_type
        self.target = setting.target
        self.id_column = setting.id_column
        self.calc_shap = setting.calc_shap
        self.debug = setting.debug
        self.model_cls: Model = model_cls
        self.features = features
        self.run_name = params.run_name
        self.preprocessing_params = params.preproccesing_params
        self.params = params.model_params
        self.save_train_pred = setting.save_train_pred
        self.cv_method = cv.cv_class.method
        self.n_splits = cv.n_splits
        self.random_state = cv.random_state
        self.shuffle = cv.shuffle
        self.cv_params = cv.cv_class
        self.feature_dir_name = feature_dir_name
        self.model_dir_name = out_dir_name
        self.remove_train_index = None # trainデータからデータを絞り込む際に使用する。除外するindexを保持。
        self.train_x, self.train_y = self.load_train()
        self.test_x = self.load_x_test()
        self.out_dir_name = out_dir_name
        self.logger = Logger(self.out_dir_name)
        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)
        self.metrics = mean_absolute_error
        self.logger.info(f'DEBUG MODE {self.debug}')
        self.logger.info(f'{self.run_name} - train_x shape: {self.train_x.shape}')
        self.logger.info(f'{self.run_name} - train_y shape: {self.train_y.shape}')
        self.logger.info(f'{self.run_name} - test_x shape: {self.test_x.shape}')


    def visualize_corr(self):
        """相関係数を算出する
        """
        fig, ax = plt.subplots(figsize=(30,20))
        plt.rcParams["font.size"]=12 # 図のfontサイズ
        plt.tick_params(labelsize=14) # 図のラベルのfontサイズ
        plt.tight_layout()

        # use a ranked correlation to catch nonlinearities
        df = self.train_x.copy()
        df[self.target] = self.train_y.copy()
        corr = df.sample(corr_sampling).corr(method='spearman')
        sns.heatmap(corr.round(3), annot=True,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)

        # 保存
        plt.savefig(self.out_dir_name + self.run_name + '_corr.png', dpi=300, bbox_inches="tight")
        plt.close()

        del df, corr

    def get_feature_name(self):
        """ 学習に使用する特徴量を返却
        """
        return self.train_x.columns.values.tolist()


    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Union[np.ndarray, float], Union[np.ndarray, float], Union[float, float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = self.train_x.copy()
        train_y = self.train_y.copy()

        if validation:
            i_fold = int(i_fold)

            # 学習データ・バリデーションデータのindexを取得
            if self.cv_method == 'KFold':
                tr_idx, va_idx = self.load_index_k_fold(i_fold)
            elif self.cv_method == 'StratifiedKFold':
                tr_idx, va_idx = self.load_index_sk_fold(i_fold)
            elif self.cv_method == 'GroupKFold':
                tr_idx, va_idx = self.load_index_gk_fold(i_fold)
            elif self.cv_method == 'TimeSeriesSplit':
                tr_idx, va_idx = self.load_index_ts_fold(i_fold)
                self.logger.info(f'{min(tr_idx), max(tr_idx), len(tr_idx)} - training index')
                self.logger.info(f'{min(va_idx), max(va_idx), len(va_idx)} - valid index')
            elif self.cv_method == 'HoldOut':
                tr_idx, va_idx = self.hold_out()
            else:
                print('CVメソッドが正しくないため終了します')
                sys.exit(0)

            ##################### numpyにしているため
            tr_x, tr_y = train_x[tr_idx], train_y[tr_idx]
            va_x, va_y = train_x[va_idx], train_y[va_idx]
            self.logger.info(f'{self.run_name} - i_fold: {i_fold} - tr_x, tr_y shape: {tr_x.shape, tr_y.shape}')
            self.logger.info(f'{self.run_name} - i_fold: {i_fold} - va_x, va_y shape: {va_x.shape, va_y.shape}')

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            if self.calc_shap:
                va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
            else:
                # 回帰問題
                if self.task_type == 'regression':
                    va_pred = model.predict(va_x)
                # 二項分類(0.5以上を1とする)
                elif self.task_type == 'binary':
                    va_pred = (va_pred > 0.5).astype(int)
                # 多項分類
                elif self.task_type == 'multiclass':
                    va_pred = np.argmax(va_pred, axis=1)
                        
            ############ (要修正)モデルに持たせたい
            if self.model_cls.__name__=='ModelLSTM':
                score = self.metrics(va_y.squeeze().reshape(-1, 1).squeeze(), va_pred)
            else:
                score = self.metrics(va_y, va_pred)

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, np.nan, np.nan, np.nan


    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        self.logger.info(f'{self.run_name} - start training cv')
        self.logger.info(f'{self.run_name} - cv method: {self.cv_method} - target: {self.target}')

        va_idxes = np.empty(0, int) # 各foldのvalidationデータのindexを保存
        preds = np.empty(0) # 各foldの推論結果を保存
        scores = np.empty(self.n_splits) # 各foldのscoreを保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes = np.append(va_idxes, va_idx)
            preds = np.append(preds, va_pred)
            scores[i_fold] = score

        # 各foldの結果をまとめる
        id_idx = self.id_idx_train[va_idxes]
        id_idx = pd.Series(id_idx)
        preds_series = pd.Series(preds)
        preds_df = pd.concat([id_idx, preds_series], axis=1)
        preds_df.columns = [self.id_column, 'pred']
        preds_df = preds_df.sort_values(self.id_column).reset_index(drop=True)

        self.logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(preds_df, self.out_dir_name + f'.{self.run_name}-train.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores)


    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.out_dir_name)
            pred = model.predict(self.test_x)
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)
        if self.task_type == 'regression':
            # 回帰
            pred_sub = pred_avg
        elif self.task_type == 'binary':
            # 二項分類(0.5以上を1とする)
            pred_sub = (pred_avg > 0.5).astype(int)
        elif self.task_type == 'multiclass':
            # 多クラス分類
            pred_sub = np.argmax(pred_avg, axis=1)
        else:
            pred_sub = np.empty(0)
            print('task_typeが正しくありません。')
            sys.exit(0)

        preds_series = pd.Series(pred_sub)
        id_idx_series = pd.Series(self.id_idx_test)
        preds_df = pd.concat([id_idx_series, preds_series], axis=1)
        preds_df.columns = [self.id_column, 'pred']
        preds_df = preds_df.sort_values(self.id_column).reset_index(drop=True)

        # 推論結果の保存（submit対象データ）
        Util.dump_df_pickle(preds_df, self.out_dir_name + f'{self.run_name}-pred.pkl')

        self.logger.info(f'{self.run_name} - end prediction cv')


    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        self.logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(self.out_dir_name)

        self.logger.info(f'{self.run_name} - end training all')


    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う
        あらかじめrun_train_allを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction all')

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model(self.out_dir_name)
        pred = model.predict(self.test_x)

        # 推論結果の保存（submit対象データ）
        Util.dump_df_pickle(pd.DataFrame(pred), self.out_dir_name + f'{self.run_name}-pred.pkl')

        self.logger.info(f'{self.run_name} - end prediction all')


    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-fold{i_fold}'
        return self.model_cls(run_fold_name, self.params)


    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_train.pkl') for f in self.features]
        df = pd.concat(dfs, axis=1)

        # ########feature_nameの取得方法を考える
        self.use_feature_name = df.columns.values.tolist()

        # 特定の値を除外して学習させる場合 -------------
        # self.remove_train_index = df[(df['age']==64) | (df['age']==66) | (df['age']==67)].index
        # df = df.drop(index = self.remove_train_index)
        # -----------------------------------------
        return df


    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む
        対数変換や使用するデータを削除する場合には、このメソッドの修正が必要
        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = pd.read_pickle(self.feature_dir_name + self.target + '_train.pkl')

        # 特定の値を除外して学習させる場合 -------------
        # train_y = train_y.drop(index = self.remove_train_index)
        # -----------------------------------------
        return pd.Series(train_y[self.target])


    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        train_x, train_y, self.preprocessing_settings_train = self.model_cls.preprocessing_train(self.load_x_train(), self.load_y_train(), self.preprocessing_params, self.feature_dir_name)
        
        # 列ID取得
        self.id_idx_train = self.preprocessing_settings_train.id_idx

        if self.debug is True:
            """サンプル数を50程度にする
            """
            idx = random.sample(range(0, len(train_x)), 50)
            return train_x[idx], train_y[idx]
        else:
            return train_x, train_y

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_test.pkl') for f in self.features]
        test_x = pd.concat(dfs, axis=1)

        test_x, self.preprocessing_settings_test = self.model_cls.preprocessing_test(test_x, self.preprocessing_params, self.feature_dir_name)

        # 列ID取得
        self.id_idx_test = self.preprocessing_settings_test.id_idx

        if self.debug is True:
            """サンプル数を20程度にする
            """
            idx = random.sample(range(0, len(test_x)), 20)
            return test_x[idx]
        else:
            return test_x


    def load_stratify_or_group_target(self) -> pd.Series:
        """
        groupKFoldで同じグループが異なる分割パターンに出現しないようにデータセットを分割したい対象カラムを取得する
        または、StratifiedKFoldで分布の比率を維持したいカラムを取得する
        :return: 分布の比率を維持したいデータの特徴量
        """
        # エラー処理入れたい
        if isinstance(self.cv_params, GroupKFoldParams):
            df = pd.read_pickle(self.feature_dir_name + self.cv_params.target + '_train.pkl')
            return df[self.cv_params.target]

    def load_index_k_fold(self, i_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        dummy_x = np.zeros(len(self.train_x))
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x))[i_fold]

    def load_index_sk_fold(self, i_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        stratify_data = self.load_stratify_or_group_target() # 分布の比率を維持したいデータの対象
        dummy_x = np.zeros(len(stratify_data))
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x, stratify_data))[i_fold]

    def load_index_gk_fold(self, i_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        group_series = self.load_stratify_or_group_target()
        dummy_x = np.zeros(len(group_series))
        kf = GroupKFold(n_splits=self.n_splits)
        return list(kf.split(dummy_x, self.train_y, groups=group_series))[i_fold]

    def load_index_ts_fold(self, i_fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        ts_index = self.preprocessing_settings_train.ts_index
        if isinstance(self.cv_params, TimeSeriesSplitParams):
            kf = MovingWindowKFold(ts_col=self.cv_params.ts_col, clipping=self.cv_params.clipping, n_splits=self.n_splits)

        ts = pd.DataFrame(ts_index)
        ts.columns = [self.cv_params.ts_col]
        return list(kf.split(ts))[i_fold]

    def hold_out(self) -> Tuple[np.ndarray, np.ndarray]:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        tr_idx = self.id_idx_train.loc[self.id_idx_train[self.id_column] < self.cv_params.min_id, :].index
        va_idx = self.id_idx_train.loc[self.id_idx_train[self.id_column] >= self.cv_params.min_id, :].index
        return tr_idx, va_idx