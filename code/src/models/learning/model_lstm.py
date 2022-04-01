import os
from pickletools import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
from model import Model
from util import Util
from keras.models import Sequential
from keras.layers import Bidirectional, Dropout, Dense, Input, LSTM
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils

# 各foldのモデルを保存する配列
model_array = []
result_array = []

class ModelLSTM(Model):

    # tr_x->pd.DataFrame, tr_y->pd.Series 型定義
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        params = dict(self.params)

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
    def calc_loss_curve(self, dir_name, run_name):
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(result_array[0].history['loss'])
        ax.plot(result_array[0].history['val_loss'])
        ax.legend(['Train', 'Val'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(dir_name + run_name + '_loss_curve.png', dpi=300, bbox_inches="tight")
        plt.close()