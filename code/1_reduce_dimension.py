import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
import json
import collections as cl
from sklearn.model_selection import train_test_split
from typing import Callable, List, Optional, Tuple, Union

sys.path.append('./src')
sys.path.append('./src/models/dimensionality_reduction')
from src.util import Logger, Util
from src.base import Feature, get_arguments, generate_features
from src.models.dimensionality_reduction.model_dimensionality_reduction import modelPCA, modelTSNE, modelUMAP
import datetime
import warnings

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
EDA_DIR_NAME = yml['SETTING']['EDA_DIR_NAME']  # EDAに関する情報を格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']


class DimensionalityReduction():
    def __init__(self, model_cls, model_params, features, setting):
        self.run_name = setting.get('run_name')
        self.params = model_params
        self.model = model_cls
        self.target = setting.get('target')
        self.features = features
        self.feature_dir_name = setting.get('feature_dir_name')
        self.out_dir_name = setting.get('out_dir_name')
        self.debug = setting.get('debug')
        self.train_x, self.train_y = self.load_train()
        self.test_x = self.load_x_test()
        self.logger = Logger(self.out_dir_name)
        self.logger.info(f'DEBUG MODE {self.debug}')
        self.logger.info(f'{self.run_name} - train_x shape: {self.train_x.shape}')
        self.logger.info(f'{self.run_name} - train_y shape: {self.train_y.shape}')
        self.logger.info(f'{self.run_name} - test_x shape: {self.test_x.shape}')

    def run(self):
        self.logger.info(f'{self.run_name} - start train')
        model = self.model(self.params)
        transformed_data = model.fit_transform(self.train_x, self.test_x)
        self.logger.info(f'{self.run_name} - end train')

        # 散布図作成
        self.logger.info(f'{self.run_name} - start plot')
        train_size = len(self.train_x)
        x, y = transformed_data[:train_size, 0], transformed_data[:train_size, 1]
        model.plotfig(x, y, self.train_y, self.out_dir_name, self.run_name)
        self.logger.info(f'{self.run_name} - end plot')

        # モデルのconfigをjsonで保存
        key_list = ['features', 'model_params', 'setting']
        value_list = [features, model_params, setting]
        save_model_config(key_list, value_list, self.out_dir_name, self.run_name)

        # データ保存
        self.logger.info(f'{self.run_name} - test_x shape: {self.test_x.shape}')
        processed_train, processed_test = model.processing_data(transformed_data)
        Util.dump_df_pickle(processed_train, self.out_dir_name + f'{self.run_name}-train.pkl')
        Util.dump_df_pickle(processed_test, self.out_dir_name + f'{self.run_name}-train.pkl')

    # 多クラス分類のデータ分割
    def load_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        train_x, train_y = self.load_x_train(), self.load_y_train()
        if self.debug is True:
            """サンプル数を200程度にする
            """
            test_size = 200 / len(train_y)
            _, train_x, _, train_y = train_test_split(train_x, train_y, test_size=test_size, stratify=train_y)
            return train_x, train_y
        else:
            return  train_x, train_y        

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_train.pkl') for f in self.features]
        df = pd.concat(dfs, axis=1)

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

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_test.pkl') for f in self.features]
        df = pd.concat(dfs, axis=1)
        
        if self.debug is True:
            """サンプル数を100程度にする
            """
            df = df.sample(n=100)
        return df

def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def save_model_config(key_list, value_list, dir_name, run_name):
    """jsonファイル生成
    """
    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data

    fw = open(dir_name + run_name  + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)

def set_default(obj):
    """json出力の際にset型のオブジェクトをリストに変更する
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':
    DEBUG = True # スクリプトが動くかどうか検証する
    now = datetime.datetime.now()
    suffix = now.strftime("_%m%d_%H%M")
    if DEBUG is True:
        suffix += '-debug'

    features = ['rawdata']

    # # ######################################################
    # # # 次元削減 PCA  #######################################
    # run_name = 'pca'
    # run_name = run_name + suffix
    # out_dir_name = EDA_DIR_NAME + run_name + '/'

    # setting = {
    #     'run_name': run_name,  # run名
    #     'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'out_dir_name': out_dir_name, #結果出力用ディレクトリ
    #     'target': 'target',  # 目的変数
    #     'debug': DEBUG
    # }

    # model_params = {
    #     'thres': 0.8
    # }

    # features = ['rawdata']

    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    # dr = DimensionalityReduction(modelPCA, model_params, features, setting)
    # dr.run()
    # # ######################################################



    ######################################################
    # 次元削減 TSNE  ######################################
    run_name = 'tsne'
    run_name = run_name + suffix
    out_dir_name = EDA_DIR_NAME + run_name + '/'

    setting = {
        'run_name': run_name,  # run名
        'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'out_dir_name': out_dir_name, #結果出力用ディレクトリ
        'target': 'target',  # 目的変数
        'debug': DEBUG
    }

    model_params = {
        'n_components': 2, 
        'perplexity': 10
    }

    my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    dr = DimensionalityReduction(modelTSNE, model_params, features, setting)
    dr.run()
    ######################################################



    # ######################################################
    # # 次元削減 UMAP  ######################################
    # run_name = 'umap'
    # run_name = run_name + suffix
    # out_dir_name = EDA_DIR_NAME + run_name + '/'

    # setting = {
    #     'run_name': run_name,  # run名
    #     'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'out_dir_name': out_dir_name, #結果出力用ディレクトリ
    #     'target': 'target',  # 目的変数
    #     'debug': DEBUG
    # }

    # model_params = {
    #     'n_components': 2, 
    #     'n_neighbors': 10
    # }

    # features = ['rawdata']

    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    # dr = DimensionalityReduction(modelUMAP, model_params, features, setting)
    # dr.run()
    # ######################################################