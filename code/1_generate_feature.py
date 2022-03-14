from copyreg import pickle
from dataclasses import replace
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
from datetime import timedelta
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split

from xfeat import SelectCategorical, LabelEncoder, Pipeline, ConcatCombination, SelectNumerical, \
    ArithmeticCombinations, TargetEncoder, aggregation, GBDTFeatureSelector, GBDTFeatureExplorer


sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
Feature.dir = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


# Target
class congestion(Feature):
    def create_features(self):
        col_name = 'congestion'
        self.train[col_name] = train[col_name]

        # 文字列変換が必要な場合
        # self.train[col_name] = train[col_name].map(lambda x: yml['SETTING']['TARGET_ENCODING'][x])
        create_memo(col_name,'種名。今回の目的変数。')

# 生データ
class rawdata(Feature):
    def create_features(self):
        self.train = train.iloc[:, 1:-1].copy()
        self.test = test.iloc[:, 1:].copy()
        create_memo('all_raw_data', '全初期データ')

# 座標
class coordinate(Feature):
    def create_features(self):
        self.train['x'] = train['x']
        self.train['y'] = train['y']
        create_memo('coordinate', '座標')

## カテゴリカル変数の結合
# 方角と座標
class x_y_direction(Feature):
    def create_features(self):
        # 結合しないcolsを明示
        exclude_cols = ['time']
        train['x'] = train['x'].map(str)
        train['y'] = train['y'].map(str)
        test['x'] = test['x'].map(str)
        test['y'] = test['y'].map(str)
        encoder = Pipeline([
            SelectCategorical(exclude_cols=exclude_cols),
            ConcatCombination(
                drop_origin=True,
                output_suffix="_re", 
                r=3),
        ])
        self.train = encoder.fit_transform(train)
        self.test = encoder.transform(test)
        create_memo('x_y_direction', 'x_y_direction')

## 数値変数の結合
# 座標
class add_x_y(Feature):
    def create_features(self):
        input_cols = ['x', 'y']
        exclude_cols = ['congestion']
        train['x'] = train['x'].map(int)
        train['y'] = train['y'].map(int)
        test['x'] = test['x'].map(int)
        test['y'] = test['y'].map(int)
        encoder = Pipeline([
            SelectNumerical(exclude_cols=exclude_cols),
            ArithmeticCombinations(
                input_cols=input_cols,
                drop_origin=True, 
                operator="+", 
                r=2),
        ])
        self.train = encoder.fit_transform(train)
        self.test = encoder.transform(test)
        create_memo('add_x_y', 'xとyの加算')

# 時間
class datetime_element(Feature):
    def create_features(self):
        col = 'time'
        train[col] = pd.to_datetime(train[col])
        # self.train['year'] = train[col].dt.year
        self.train['month'] = train[col].dt.month
        self.train['weekday'] = train[col].dt.weekday
        self.train['day'] = train[col].dt.day
        self.train['hour'] = train[col].dt.hour
        self.train['minute'] = train[col].dt.minute
        # self.train['second'] = train[col].dt.second

        test[col] = pd.to_datetime(test[col])
        # self.test['year'] = test[col].dt.year
        self.test['month'] = test[col].dt.month
        self.test['weekday'] = test[col].dt.weekday
        self.test['day'] = test[col].dt.day
        self.test['hour'] = test[col].dt.hour
        self.test['minute'] = test[col].dt.minute
        # self.test['second'] = test[col].dt.second
        create_memo('time', '年、月、週、日、時間、分、秒')

# 1日の積算の分
class accum_minutes(Feature):
    def create_features(self):
        col = 'time'
        train[col] = pd.to_datetime(train[col])
        self.train['accum_minutes'] = (train[col] - train[col].dt.floor('D')).dt.total_seconds() / 60

        test[col] = pd.to_datetime(test[col])
        self.test['accum_minutes'] = (test[col] - test[col].dt.floor('D')).dt.total_seconds() / 60
        create_memo('accum_minutes', '積算分')


# 0.5日の積算の分
class accum_minutes_half_day(Feature):
    def create_features(self):
        col = 'time'
        train[col] = pd.to_datetime(train[col])
        self.train['accum_minutes_half_day'] = (train[col] - train[col].dt.floor('D')).dt.total_seconds() / 60

        test[col] = pd.to_datetime(test[col])
        self.test['accum_minutes_half_day'] = (test[col] - test[col].dt.floor('D')).dt.total_seconds() / 60

        # テストデータが午後のみのため、午前と午後に区別する
        self.train['pm'] = 0
        self.train.loc[self.train['accum_minutes_half_day']>=720, 'pm'] = 1
        self.train.loc[self.train['accum_minutes_half_day']>=720, 'accum_minutes_half_day'] = self.train.loc[self.train['accum_minutes_half_day']>=720, 'accum_minutes_half_day'] - 720
        self.train['accum_minutes_half_day'] = self.train['accum_minutes_half_day'].map(int)
        
        self.test['pm'] = 0
        self.test.loc[self.test['accum_minutes_half_day']>=720, 'pm'] = 1
        self.test.loc[self.test['accum_minutes_half_day']>=720, 'accum_minutes_half_day'] = self.test.loc[self.test['accum_minutes_half_day']>=720, 'accum_minutes_half_day'] - 720
        self.test['accum_minutes_half_day'] = self.test['accum_minutes_half_day'].map(int)
        create_memo('accum_minutes_half_day', '積算分(半日)')

## 差分
## 時系列差(1階差分)分
class diff_days(Feature):
    def create_features(self):
        train['time'] = pd.to_datetime(train['time'])
        test['time'] = pd.to_datetime(test['time'])
        for i in [1]:
            train['diff'] = train['time'] + timedelta(days=i)
            train['time_categorical'] = train['time'].map(lambda x: str(x)+ '_') + train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']
            train['diff_categorical'] = train['diff'].map(lambda x: str(x)+ '_') + train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']
            df = pd.merge(train[['time_categorical', 'congestion']], train[['diff_categorical', 'congestion']], left_on='time_categorical', right_on='diff_categorical', how='left')
            self.train[f'diff_{i}days'] = df['congestion_y']

            test['time_categorical'] = test['time'].map(lambda x: str(x)+ '_') + test['x'].map(lambda x: str(x) + '_') + test['y'].map(lambda x: str(x) + '_') + test['direction']
            df = pd.merge(test[['time_categorical']], train[['diff_categorical', 'congestion']], left_on='time_categorical', right_on='diff_categorical', how='left')
            self.test[f'diff_{i}days'] = df['congestion'] 
        create_memo('diff_days', '1日差分')

## 差分
## 時系列差分(1~3階差分)
class diff_3days(Feature):
    def create_features(self):
        train['time'] = pd.to_datetime(train['time'])
        test['time'] = pd.to_datetime(test['time'])
        for i in [1, 2, 3]:
            train['diff'] = train['time'] + timedelta(days=i)
            train['time_categorical'] = train['time'].map(lambda x: str(x)+ '_') + train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']
            train['diff_categorical'] = train['diff'].map(lambda x: str(x)+ '_') + train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']
            df = pd.merge(train[['time_categorical', 'congestion']], train[['diff_categorical', 'congestion']], left_on='time_categorical', right_on='diff_categorical', how='left')
            self.train[f'diff_{i}days'] = df['congestion_y']

            test['time_categorical'] = test['time'].map(lambda x: str(x)+ '_') + test['x'].map(lambda x: str(x) + '_') + test['y'].map(lambda x: str(x) + '_') + test['direction']
            df = pd.merge(test[['time_categorical']], train[['diff_categorical', 'congestion']], left_on='time_categorical', right_on='diff_categorical', how='left')
            self.test[f'diff_{i}days'] = df['congestion']
        create_memo('diff_days', '3日差分')

## one-hot encoding
# 方角
class direction_dummies(Feature):
    def create_features(self):
        self.train = pd.get_dummies(train['direction'])
        self.test = pd.get_dummies(test['direction'])
        create_memo('direction', '方角')

## one-hot encoding
# 方角と座標
class x_y_direction_dummies(Feature):
    def create_features(self):
        train['x_y_direction'] = train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']        
        test['x_y_direction'] = test['x'].map(lambda x: str(x) + '_') + test['y'].map(lambda x: str(x) + '_') + test['direction']
        self.train = pd.get_dummies(train['x_y_direction'])
        self.test = pd.get_dummies(test['x_y_direction'])
        create_memo('x_y_direction', 'x,y,directionを結合')

## 統計量作成
# 集約して差分をとる
class agg_shift_by_date(Feature):
    def create_features(self):
        time_col = 'time'
        train[time_col] = pd.to_datetime(train[time_col])        
        train['date'] = train[time_col].dt.date
        train['pm'] = 0
        train.loc[train[time_col].dt.hour>=12, 'pm'] = 1

        test[time_col] = pd.to_datetime(test[time_col])
        test['date'] = test[time_col].dt.date
        test['pm'] = 0
        test.loc[test[time_col].dt.hour>=12, 'pm'] = 1

        dropped_cols_train, dropped_cols_test = train.columns, test.columns
        agg_cols = ['min', 'max', 'mean', 'std']
        cols_set = [['date'], ['date', 'pm'], ['date', 'direction', 'x', 'y']]

        self.train, self.test = train.copy(), test.copy()

        for cols in cols_set:
            grp_df = train.groupby(cols)['congestion'].agg(agg_cols)
            
            # 過去のデータの統計量を特徴量とする
            for i in [1, 2, 3]:
                grp_df_shift = grp_df.copy()
                col_name = '_'.join(cols)
                grp_df_shift.columns = [f'{col_name}_shift{i}_{c}' for c in grp_df_shift.columns]
                grp_df_shift = grp_df_shift.reset_index()
                grp_df_shift['date'] = grp_df_shift['date'] + timedelta(days=i)

                self.train = self.train.merge(grp_df_shift, on=cols, how='left')
                self.test = self.test.merge(grp_df_shift, on=cols, how='left')
        
        # 不要なカラム削除
        self.train.drop(dropped_cols_train, axis=1, inplace=True)
        self.test.drop(dropped_cols_test, axis=1, inplace=True)
        create_memo('agg_shift_by_date', f'{cols_set}で集約&差分')


## target encoding
# 時間と方角について
class target_encoded_feats(Feature):
    def create_features(self):
        fold = KFold(n_splits=5, shuffle=False)
        initial_train_cols = train.columns
        initial_test_cols = test.columns
        col = 'time'
        categorical_cols = ['direction', 'hour', 'x', 'y']
        target_col = 'congestion'
        train[col] = pd.to_datetime(train[col])
        train['hour'] = train[col].dt.hour
        train['hour'] = train['hour'].map(str)
        train['x'] = train['x'].map(str)
        train['y'] = train['y'].map(str)

        test[col] = pd.to_datetime(test[col])
        test['hour'] = test[col].dt.hour
        test['hour'] = test['hour'].map(str)
        test['x'] = test['x'].map(str)
        test['y'] = test['y'].map(str)
        encoder = TargetEncoder(
            input_cols=categorical_cols,
            target_col=target_col,
            fold=fold,
            output_suffix="_re"
            )

        self.train = encoder.fit_transform(train).drop(initial_train_cols, axis=1)
        self.test = encoder.transform(test).drop(initial_test_cols, axis=1)
        create_memo('target_encoding', f'{categorical_cols}でtarget_encoding')


# ## 特徴量集約
# # 差分無し
# class agg_shift_by_date(Feature):
#   def create_features(self):
#       dropped_cols_train, dropped_cols_test = train.columns, test.columns
#       agg_cols = ['min', 'max', 'mean', 'std']
#       cols_set = [['date'], ['date', 'pm'], ['date', 'direction', 'x', 'y']]

#       for cols in cols_set:
#           grp_df = train.groupby(cols)['congestion'].agg(agg_cols)
#           col_name = '_'.join(cols)
#           grp_df.columns = [f'{col_name}_shift{i}_{c}' for c in grp_df.columns]
#           train = train.merge(grp_df, on=cols, how='left')
#           test = test.merge(grp_df, on=cols, how='left')          

#       # 不要なカラム削除
#       self.train = train.drop(dropped_cols_train, axis=1)
#       self.test = test.drop(dropped_cols_test, axis=1)
    #   create_memo('agg_shift_by_date', f'{cols_set}で集約')


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
    # train = pd.read_csv(RAW_DIR_NAME + 'train_imputation.csv')
    test = pd.read_csv(RAW_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)