import pandas as pd
import numpy as np
import collections as cl
from sklearn.model_selection import train_test_split
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from src.util import Logger, Util
from contextlib import contextmanager
from pathlib import Path
import time

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

class FigurePlot():
    base_dir_name = ''
    def __init__(self, model_params, features, setting):
        self.run_name = setting.get('run_name')
        self.params = model_params
        self.features = features
        self.feature_dir_name = setting.get('feature_dir_name')
        self.out_dir_name = setting.get('out_dir_name')
        self.train = self.load_train()
        self.test = self.load_test()

    def run_and_save(self):
        path = Path(self.out_dir_name) / f'{self.run_name}.png'
        if path.exists():
            print(path.name, 'was skipped')
        else:
            with timer(path):
                self.create_figure()
                plt.savefig(str(path), dpi=300, bbox_inches="tight")
                plt.close()

    def load_train(self) -> pd.DataFrame:
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

    def load_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        dfs = [pd.read_pickle(self.feature_dir_name + f'{f}_test.pkl') for f in self.features]
        df = pd.concat(dfs, axis=1)
        return df