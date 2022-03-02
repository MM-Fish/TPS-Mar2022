from copyreg import pickle
import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings
sys.path.append('./src')
from src.base import Feature, get_arguments, generate_features

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
Feature.dir = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


# Target
class target(Feature):
    def create_features(self):
        col_name = 'congestion'
        self.train[col_name] = train[col_name]

        # 文字列変換が必要な場合
        # self.train[col_name] = train[col_name].map(lambda x: yml['SETTING']['TARGET_ENCODING'][x])
        create_memo(col_name,'種名。今回の目的変数。')

class rawdata(Feature):
    def create_features(self):
        self.train = train.iloc[:, 1:-1].copy()
        self.test = test.iloc[:, 1:].copy()
        create_memo('all_raw_data', '全初期データ')

# # 学習モデルを特徴量データとして追加
# class keras_0226_0937(Feature):
#     def create_features(self):
#         dir_name = self.__class__.__name__
#         self.train = pd.read_pickle(MODEL_DIR_NAME + f'{dir_name}/.{dir_name}-train.pkl').rename(columns={0: dir_name})
#         self.test = pd.read_pickle(MODEL_DIR_NAME + f'{dir_name}/{dir_name}-pred.pkl').rename(columns={0: dir_name})
#         create_memo('all_raw_data', 'lgb_0226_0545のデータ')

# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '/_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path,"w") as f:
            writer = csv.writer(f)
            writer.writerow([col_name, desc])

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])

if __name__ == '__main__':

    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)