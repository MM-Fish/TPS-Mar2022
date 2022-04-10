from dataclasses import asdict
import os
from pickletools import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
from model import Model, PreprocessingSettings
from util import Util
from typing import Tuple, Dict
from keras.models import Sequential
from keras.layers import Bidirectional, Dropout, Dense, Input, LSTM
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from params import PreprocessingSettings, PreprocessingParams

# 各foldのモデルを保存する配列
model_array = []
result_array = []

class ModelLSTM(Model):
    # tr_x->pd.DataFrame, tr_y->pd.Series 型定義
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        params = asdict(self.params)

        if params['task_type'] == 'multiclass':
            tr_y = pd.DataFrame(np_utils.to_categorical(np.array(tr_y)))
            if va_y is not None:
                va_y = pd.DataFrame(np_utils.to_categorical(np.array(va_y)))

        # データのセット
        validation = va_x is not None

        # ハイパーパラメータの設定
        if params['optimizer'] == 'SGD':
            learning_rate = params['learning_rate']
            decay_rate = params['learning_rate'] / params['epochs']
            momentum = params['momentum']
            optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

        # モデルの定義
        self.model = self.build_model(tr_x.shape[-2:], optimizer)

        # 学習
        if validation:
            result = self.model.fit(
                                tr_x,
                                tr_y,
                                epochs = params['epochs'],
                                batch_size = params['batch_size'],
                                validation_data = [va_x, va_y]
                                )
            result_array.append(result)
            model_array.append(self.model)

    # shapを計算しないver
    def predict(self, te_x):
        return self.model.predict(te_x).squeeze().reshape(-1, 1).squeeze()


    # shapを計算するver
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        return valid_prediticion, fold_importance

    def build_model(self, train_shape, optimizer):
        model = Sequential()
        model.add(Input(shape=train_shape))
        model.add(Bidirectional(LSTM(1024, return_sequences=True)))
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dense(128, activation='selu'))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss="mae")
        return model

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)

    @classmethod
    def preprocessing_train(self, train_x: pd.DataFrame, train_y: pd.Series, params: PreprocessingParams, feature_dir_name: str) -> Tuple[np.ndarray, np.ndarray, PreprocessingSettings]:
        ts_col = params.ts_col
        id_col = params.id_col
        sort_cols = params.sort_cols
        sort_col_feats = params.sort_col_feats

        ############### lstmに内部的に切り出して良いかも?
        # データ読み込み
        dfs = [pd.read_pickle(feature_dir_name + f'{f}_train.pkl') for f in sort_col_feats]
        df_ts_base_feats = pd.concat(dfs, axis=1)

        # train_xの不要な行を削除
        drop_index = (train_x.isnull().any(axis=1)) | (train_y.isna())

        # 不要な行を1行でも含む日付を削除と並び替え
        drop_dates = df_ts_base_feats.loc[drop_index, ts_col].unique()
        keep_index = (~df_ts_base_feats[ts_col].isin(drop_dates) & df_ts_base_feats['pm'] == 1).to_numpy()
        df_ts_base_feats = df_ts_base_feats.loc[keep_index, :]
        df_ts_base_feats = df_ts_base_feats.sort_values(sort_cols)
        sort_index = df_ts_base_feats.index

        # LSTM用にデータのshapeを変形する
        size_name = len(df_ts_base_feats['accum_minutes_half_day'].unique())
        train_x, train_y = train_x.loc[sort_index, :], train_y[sort_index]
        train_x = np.array(train_x).reshape(-1, size_name, train_x.shape[-1])
        train_y = train_y.to_numpy().reshape(-1, size_name)
        ts_index = np.array(df_ts_base_feats[ts_col]).reshape(-1, size_name)[:, 0]
        ##############
        
        # バリデーションのデータ分割に使用
        id_idx = pd.read_pickle(feature_dir_name + f'{id_col}_train.pkl').loc[sort_index, id_col].to_numpy()
        preprocessing_settings = PreprocessingSettings(keep_index, ts_index, id_idx)
        return train_x, train_y, preprocessing_settings

    @classmethod
    def preprocessing_test(self, test_x: pd.DataFrame, params: PreprocessingParams, feature_dir_name: str) -> Tuple[np.ndarray, PreprocessingSettings]:
        ts_col = params.ts_col
        id_col = params.id_col
        sort_cols = params.sort_cols
        sort_col_feats = params.sort_col_feats

        ############### lstmに内部的に切り出して良いかも?
        # データ読み込み
        dfs = [pd.read_pickle(feature_dir_name + f'{f}_test.pkl') for f in sort_col_feats]
        df_ts_base_feats = pd.concat(dfs, axis=1)

        # 並び替え
        df_ts_base_feats = df_ts_base_feats.sort_values(sort_cols)
        sort_index = df_ts_base_feats.index

        # LSTM用にデータのshapeを変形する
        size_name = len(df_ts_base_feats['accum_minutes_half_day'].unique())
        test_x = test_x.loc[sort_index, :]
        test_x = np.array(test_x).reshape(-1, size_name, test_x.shape[-1])
        ts_index = np.array(df_ts_base_feats[ts_col]).reshape(-1, size_name)[:, 0]
        ##############
        
        # バリデーションのデータ分割に使用
        id_idx = pd.read_pickle(feature_dir_name + f'{id_col}_test.pkl').loc[sort_index, id_col].to_numpy()
        preprocessing_settings = PreprocessingSettings(None, ts_index, id_idx)
        return test_x, preprocessing_settings

    @classmethod
    def calc_loss_curve(self, dir_name, run_name):
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(result_array[0].history['loss'])
        ax.plot(result_array[0].history['val_loss'])
        ax.legend(['Train', 'Val'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(dir_name + run_name + '_loss_curve.png', dpi=300, bbox_inches="tight")
        plt.close()