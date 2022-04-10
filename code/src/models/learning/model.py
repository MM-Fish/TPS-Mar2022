import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass
from params import PreprocessingSettings, PreprocessingParams

class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @classmethod
    def preprocessing_train(self, train_x: pd.DataFrame, train_y: pd.Series, params: PreprocessingParams, feature_dir_name: str) -> Tuple[np.ndarray, np.ndarray, PreprocessingSettings]:
        ts_col = params.ts_col
        id_col = params.id_col

        # 時系列カラム読み込み
        ts_index = pd.read_pickle(feature_dir_name + f'{ts_col}_train.pkl').to_numpy()

        # 特に前処理をする必要がなければしない
        keep_index = np.repeat(True, len(train_x))
        train_x, train_y = train_x.loc[keep_index, :], train_y[keep_index]
        train_x, train_y = np.array(train_x), np.array(train_y)
        sort_index = keep_index

        # バリデーションのデータ分割に使用
        id_idx = pd.read_pickle(feature_dir_name + f'{id_col}_train.pkl').loc[sort_index, id_col].to_numpy()
        preprocessing_settings = PreprocessingSettings(keep_index, ts_index, id_idx)
        return train_x, train_y, preprocessing_settings

    @classmethod
    def preprocessing_test(self, test_x: pd.DataFrame, params: PreprocessingParams, feature_dir_name: str) -> Tuple[np.ndarray, PreprocessingSettings]:
        ts_col = params.ts_col
        id_col = params.id_col

        # データ読み込み
        ts_index = pd.read_pickle(feature_dir_name + f'{ts_col}_test.pkl').to_numpy()

        # 特に前処理をする必要がなければしない
        keep_index = np.repeat(True, len(test_x))
        test_x = test_x.loc[keep_index, :]
        test_x = np.array(test_x)
        sort_index = keep_index

        # バリデーションのデータ分割に使用
        id_idx = pd.read_pickle(feature_dir_name + f'{id_col}_test.pkl').loc[sort_index, id_col].to_numpy()
        preprocessing_settings = PreprocessingSettings(keep_index, ts_index, id_idx)
        return test_x, preprocessing_settings

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
                va_x: Optional[pd.DataFrame] = None,
                va_y: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        pass
