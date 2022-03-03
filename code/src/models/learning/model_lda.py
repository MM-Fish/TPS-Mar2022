import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from model import Model
from util import Util
import seaborn as sns
from keras.utils import np_utils

# 各foldのモデルを保存する配列
model_array = []

class ModelLDA(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None

        # ハイパーパラメータの設定
        params = dict(self.params)

        # 学習
        lda = LDA(solver=params['solver'], n_components=params['n_components'])
        self.model = lda.fit(tr_x, tr_y)
        model_array.append(self.model)

    # shapを計算しないver
    def predict(self, te_x):
        # 他クラス分類時のlgbの返り値に合わせる
        return np_utils.to_categorical(self.model.predict(te_x))


    # shapを計算するver
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        return valid_prediticion, fold_importance


    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)

    @classmethod
    def plot_scatter(self, dir_name, run_name, train_x, train_y, params):
        lda = LDA(solver=params['solver'], n_components=params['n_components'])
        transformed_train_x = lda.fit_transform(train_x, train_y)

        # グラフ化
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(transformed_train_x[:, 0], transformed_train_x[:, 1], alpha=0.8, hue=train_y, ax=ax)
        plt.savefig(dir_name + run_name + f'scatter.png', dpi=300, bbox_inches="tight")
        plt.close()
