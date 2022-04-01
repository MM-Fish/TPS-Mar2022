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

# 前処理後
RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME_IMP']
Feature.dir = yml['SETTING']['FEATURE_DIR_NAME_IMP']

# ディレクトリ確認
print(RAW_DIR_NAME)
print(Feature.dir)

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
class row_id(Feature):
    def create_features(self):
        self.train['row_id'] = train['row_id']
        self.test['row_id'] = test['row_id']
        create_memo('id', 'ID')

# 方角
class direction(Feature):
    def create_features(self):
        self.train['direction'] = train['direction']
        self.test['direction'] = test['direction']
        create_memo('direction', '方角')

# 座標
class coordinate(Feature):
    def create_features(self):
        self.train['x'] = train['x']
        self.train['y'] = train['y']

        self.test['x'] = test['x']
        self.test['y'] = test['y']
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

## 方角
class combine_decompose_direction(Feature):
    def create_features(self):
        dir_mapper = {'EB': [1,0], 
                    'NB': [0,1], 
                    'SB': [0,-1], 
                    'WB': [-1,0], 
                    'NE': [1,1], 
                    'SW': [-1,-1], 
                    'NW': [-1,1], 
                    'SE': [1,-1]}
        self.train['direction_x_axis'] = train['direction'].map(lambda x: dir_mapper[x][0])
        self.train['direction_y_axis'] = train['direction'].map(lambda x: dir_mapper[x][1])
        self.train['x+y+directionx'] = train['x'].astype('str') + train['y'].astype('str') + self.train['direction_x_axis'].astype('str')
        self.train['x+y+directiony'] = train['x'].astype('str') + train['y'].astype('str') + self.train['direction_y_axis'].astype('str')

        self.test['direction_x_axis'] = test['direction'].map(lambda x: dir_mapper[x][0])
        self.test['direction_y_axis'] = test['direction'].map(lambda x: dir_mapper[x][1])
        self.test['x+y+directionx'] = test['x'].astype('str') + test['y'].astype('str') + self.test['direction_x_axis'].astype('str')
        self.test['x+y+directiony'] = test['x'].astype('str') + test['y'].astype('str') + self.test['direction_y_axis'].astype('str')
        create_memo('decompose_direction', '方角をx軸とy軸に分解')

## 方角
class decompose_direction(Feature):
    def create_features(self):
        dir_mapper = {'EB': [1,0], 
                    'NB': [0,1], 
                    'SB': [0,-1], 
                    'WB': [-1,0], 
                    'NE': [1,1], 
                    'SW': [-1,-1], 
                    'NW': [-1,1], 
                    'SE': [1,-1]}
        self.train['direction_x_axis'] = train['direction'].map(lambda x: dir_mapper[x][0])
        self.train['direction_y_axis'] = train['direction'].map(lambda x: dir_mapper[x][1])

        self.test['direction_x_axis'] = test['direction'].map(lambda x: dir_mapper[x][0])
        self.test['direction_y_axis'] = test['direction'].map(lambda x: dir_mapper[x][1])
        create_memo('decompose_direction', '方角をx軸とy軸に分解')

# 時間
class datetime_element(Feature):
    def create_features(self):
        time_col = 'time'
        train[time_col] = pd.to_datetime(train[time_col])
        # self.train['year'] = train[time_col].dt.year
        self.train['month'] = train[time_col].dt.month
        self.train['weekday'] = train[time_col].dt.weekday
        self.train['day'] = train[time_col].dt.day
        self.train['hour'] = train[time_col].dt.hour
        self.train['minute'] = train[time_col].dt.minute
        # self.train['second'] = train[time_col].dt.second

        test[time_col] = pd.to_datetime(test[time_col])
        # self.test['year'] = test[time_col].dt.year
        self.test['month'] = test[time_col].dt.month
        self.test['weekday'] = test[time_col].dt.weekday
        self.test['day'] = test[time_col].dt.day
        self.test['hour'] = test[time_col].dt.hour
        self.test['minute'] = test[time_col].dt.minute
        # self.test['second'] = test[time_col].dt.second
        create_memo('datetime_element', '年、月、週、日、時間、分、秒')

# 週末
class is_weekend(Feature):
    def create_features(self):
        time_col = 'time'
        train[time_col] = pd.to_datetime(train[time_col])
        self.train['is_weekend'] = (train[time_col].dt.dayofweek > 4).astype('int')

        test[time_col] = pd.to_datetime(test[time_col])
        self.test['is_weekend'] = (test[time_col].dt.dayofweek > 4).astype('int')
        create_memo('is_weekend', '週末')

# 日付のみ取得
# kflodに使う特徴量（クラス名とカラム名を揃える）
class date_obj(Feature):
    def create_features(self):
        time_col = 'time'
        self.train['date_obj'] = pd.to_datetime(train[time_col]).dt.date
        self.test['date_obj'] = pd.to_datetime(test[time_col]).dt.date
        create_memo('data_obj', '日付')

# 1日の積算の分
class accum_minutes(Feature):
    def create_features(self):
        time_col = 'time'
        train[time_col] = pd.to_datetime(train[time_col])
        self.train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60

        test[time_col] = pd.to_datetime(test[time_col])
        self.test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        create_memo('accum_minutes', '積算分')


# 0.5日の積算の分
class accum_minutes_half_day(Feature):
    def create_features(self):
        time_col = 'time'
        train[time_col] = pd.to_datetime(train[time_col])
        self.train['accum_minutes_half_day'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60

        test[time_col] = pd.to_datetime(test[time_col])
        self.test['accum_minutes_half_day'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60

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

## one-hot encoding
# 方角
class direction_dummies(Feature):
    def create_features(self):
        col = 'direction'

        ohe = OneHotEncoder(sparse=False)
        train_dummies = ohe.fit_transform(train[[col]])
        self.train = pd.DataFrame(train_dummies, columns=ohe.categories_[0])
        
        test_dummies = ohe.transform(test[[col]])
        self.test = pd.DataFrame(test_dummies, columns=ohe.categories_[0])
        create_memo(col, '方角')

## one-hot encoding
# 方角と座標
class x_y_direction_dummies(Feature):
    def create_features(self):
        col = 'x_y_direction'
        train[col] = train['x'].map(lambda x: str(x) + '_') + train['y'].map(lambda x: str(x) + '_') + train['direction']        
        test[col] = test['x'].map(lambda x: str(x) + '_') + test['y'].map(lambda x: str(x) + '_') + test['direction']

        ohe = OneHotEncoder(sparse=False)
        train_dummies = ohe.fit_transform(train[[col]])
        self.train = pd.DataFrame(train_dummies, columns=ohe.categories_[0])
        
        test_dummies = ohe.transform(test[[col]])
        self.test = pd.DataFrame(test_dummies, columns=ohe.categories_[0])

        create_memo(col, f'{col}を結合')

## 時系列
## shift(1階差分)
class shift_1day(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60

        train_and_test = pd.concat([train, test])
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [1]:
            shift_series = grp_df.shift(i)
            outputs.append(pd.DataFrame(shift_series).add_prefix(f'shift{i}_'))

        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :]
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :]
        create_memo('shift_day', '1日shift')

## 時系列
## shift(1~3階差分)
class shift_3days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [1, 2, 3]:
            shift_series = grp_df.shift(i)
            outputs.append(pd.DataFrame(shift_series).add_prefix(f'shift{i}_'))

        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :]
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :]
        create_memo('shift_3days', '3日shift')

## 時系列
## 差分(1~3階差分)
class diff_3days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [1, 2, 3]:
            shift_series = grp_df.diff(i)
            outputs.append(pd.DataFrame(shift_series).add_prefix(f'diff{i}_'))

        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :]
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :]
        create_memo('shift_days', '3日差分')

## 時系列
## 移動平均
class rolling_days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        train_and_test.index = train_and_test['row_id']
        # 1階差分shiftさせる（過去データを含まないようにするため）
        train_and_test[target_col] = train_and_test.groupby(cols)[target_col].shift(1)
        
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [2, 3, 5]:
            rolling_df = grp_df.rolling(i).agg(agg_cols)
            rolling_df = pd.DataFrame(rolling_df).add_prefix(f'rolling{i}_')
            rolling_df.index = rolling_df.index.map(lambda x: x[4])
            outputs.append(rolling_df.sort_index())
        
        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :].reset_index(drop=True)
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :].reset_index(drop=True)
        create_memo('rolling', '移動平均')

## 時系列
## 移動平均(10日)
class rolling_10days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        train_and_test.index = train_and_test['row_id']
        # 1階差分shiftさせる（過去データを含まないようにするため）
        train_and_test[target_col] = train_and_test.groupby(cols)[target_col].shift(1)
        
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [10]:
            rolling_df = grp_df.rolling(i).agg(agg_cols)
            rolling_df = pd.DataFrame(rolling_df).add_prefix(f'rolling{i}_')
            rolling_df.index = rolling_df.index.map(lambda x: x[4])
            outputs.append(rolling_df.sort_index())
        
        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :].reset_index(drop=True)
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :].reset_index(drop=True)
        create_memo('rolling_10days', '移動平均')

## 移動平均(30日)
class rolling_30days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        train_and_test.index = train_and_test['row_id']
        # 1階差分shiftさせる（過去データを含まないようにするため）
        train_and_test[target_col] = train_and_test.groupby(cols)[target_col].shift(1)
        
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [30]:
            rolling_df = grp_df.rolling(i).agg(agg_cols)
            rolling_df = pd.DataFrame(rolling_df).add_prefix(f'rolling{i}_')
            rolling_df.index = rolling_df.index.map(lambda x: x[4])
            outputs.append(rolling_df.sort_index())
        
        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :].reset_index(drop=True)
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :].reset_index(drop=True)
        create_memo('rolling_30days', '移動平均')


## 移動平均(50日)
class rolling_50days(Feature):
    def create_features(self):
        cols = ['accum_minutes', 'direction', 'x', 'y']
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['accum_minutes'] = (train[time_col] - train[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_train = len(train)

        test[time_col] = pd.to_datetime(test[time_col])
        test['accum_minutes'] = (test[time_col] - test[time_col].dt.floor('D')).dt.total_seconds() / 60
        n_test = len(test)

        train_and_test = pd.concat([train, test])
        train_and_test.index = train_and_test['row_id']
        # 1階差分shiftさせる（過去データを含まないようにするため）
        train_and_test[target_col] = train_and_test.groupby(cols)[target_col].shift(1)
        
        grp_df = train_and_test.groupby(cols)[target_col]

        outputs = []
        for i in [50]:
            rolling_df = grp_df.rolling(i).agg(agg_cols)
            rolling_df = pd.DataFrame(rolling_df).add_prefix(f'rolling{i}_')
            rolling_df.index = rolling_df.index.map(lambda x: x[4])
            outputs.append(rolling_df.sort_index())
        
        self.train = pd.concat(outputs, axis=1).iloc[:n_train, :].reset_index(drop=True)
        self.test = pd.concat(outputs, axis=1).iloc[n_train:, :].reset_index(drop=True)
        create_memo('rolling_30days', '移動平均')

## 統計量作成
# 集約して差分をとる
class agg_shift_by_date(Feature):
    def create_features(self):
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        cols_set = [['date'], ['date', 'pm'], ['date', 'direction', 'x', 'y'], ['date', 'pm', 'direction'], ['date', 'pm', 'y'], ['date', 'pm', 'x']]
        time_col = 'time'
        target_col = 'congestion'
        train[time_col] = pd.to_datetime(train[time_col])        
        train['date'] = train[time_col].dt.date
        train['pm'] = 0
        train.loc[train[time_col].dt.hour>=12, 'pm'] = 1

        test[time_col] = pd.to_datetime(test[time_col])
        test['date'] = test[time_col].dt.date
        test['pm'] = 0
        test.loc[test[time_col].dt.hour>=12, 'pm'] = 1

        dropped_cols_train, dropped_cols_test = train.columns, test.columns

        self.train, self.test = train.copy(), test.copy()

        for cols in cols_set:
            grp_df = train.groupby(cols)[target_col].agg(agg_cols)
            
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

# 午前中のみを集約して差分をとる
class agg_by_am(Feature):
  def create_features(self):
        agg_cols = ['min', 'max', 'mean', 'median', 'std']
        cols_set = [['date', 'am'], ['date', 'am', 'direction', 'x', 'y'], ['date', 'am', 'direction'], ['date', 'am', 'y'], ['date', 'am', 'x']]
        target_col = 'congestion'
        time_col = 'time'

        train[time_col] = pd.to_datetime(train[time_col])
        train['date'] = train[time_col].dt.date
        train['am'] = 1
        train.loc[train[time_col].dt.hour>=12, 'am'] = 0

        test[time_col] = pd.to_datetime(test[time_col])
        test['date'] = test[time_col].dt.date
        test['am'] = 1
        test.loc[test[time_col].dt.hour>=12, 'am'] = 0

        dropped_cols_train, dropped_cols_test = train.columns, test.columns
        # train=の形にすると、ここ以前のtrainがundefinedになる
        self.train, self.test = train.copy(), test.copy()

        for cols in cols_set:
            grp_df = train.groupby(cols)[target_col].agg(agg_cols)
            col_name = '_'.join(cols)
            grp_df.columns = [f'{col_name}_{c}' for c in grp_df.columns]
            # self.train = train.merge(grp_df, on=cols, how='left')
            # self.test = test.merge(grp_df, on=cols, how='left')
            # 午前のデータを午前と午後に結合
            cols.remove('am')
            self.train = self.train.merge(grp_df.reset_index().query('am==1').drop(['am'], axis=1), on=cols, how='left')
            self.test = self.test.merge(grp_df.reset_index().query('am==1').drop(['am'], axis=1), on=cols, how='left')

        # 不要なカラム削除
        self.train = self.train.drop(dropped_cols_train, axis=1)
        self.test = self.test.drop(dropped_cols_test, axis=1)
        create_memo('agg_by_am', f'{cols_set}で集約')

## target encoding
# 時間と方角について
class target_encoded_feats(Feature):
    def create_features(self):
        fold = KFold(n_splits=5, shuffle=False)
        initial_train_cols = train.columns
        initial_test_cols = test.columns
        time_col = 'time'
        categorical_cols = ['direction', 'hour', 'x', 'y']
        target_col = 'congestion'
        train[time_col] = pd.to_datetime(train[time_col])
        train['hour'] = train[time_col].dt.hour
        train['hour'] = train['hour'].map(str)
        train['x'] = train['x'].map(str)
        train['y'] = train['y'].map(str)

        test[time_col] = pd.to_datetime(test[time_col])
        test['hour'] = test[time_col].dt.hour
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
# # 時系列の場合はリークする
# class agg_feats(Feature):
#   def create_features(self):
#         agg_cols = ['min', 'max', 'mean', 'std']
#         cols_set = [['accum_minutes', 'direction', 'x', 'y']]
#         time_col = 'time'
#         dropped_cols_train, dropped_cols_test = train.columns, test.columns

#         for cols in cols_set:
#             grp_df = train.groupby(cols)[time_col].agg(agg_cols)
#             col_name = '_'.join(cols)
#             grp_df.columns = [f'{col_name}_shift{i}_{c}' for c in grp_df.columns]
#             train = train.merge(grp_df, on=cols, how='left')
#             test = test.merge(grp_df, on=cols, how='left')          

#         # 不要なカラム削除
#         self.train = train.drop(dropped_cols_train, axis=1)
#         self.test = test.drop(dropped_cols_test, axis=1)
#         create_memo('agg_feats', f'{cols_set}で集約')

# # 学習モデルを特徴量データとして追加
# class keras_0226_0937(Feature):
#     def create_features(self):
#         dir_name = self.__class__.__name__
#         self.train = pd.read_pickle(MODEL_DIR_NAME + f'{dir_name}/.{dir_name}-train.pkl').rename(columns={0: dir_name})
#         self.test = pd.read_pickle(MODEL_DIR_NAME + f'{dir_name}/{dir_name}-pred.pkl').rename(columns={0: dir_name})
#         create_memo('all_raw_data', 'lgb_0226_0545のデータ')

# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    if not os.path.isfile(feature_memo_path):
        with open(feature_memo_path,"w") as f:
            writer = csv.writer(f)
            writer.writerow([col_name, desc])

    with open(feature_memo_path, 'r+') as f:
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
    if RAW_DIR_NAME == yml['SETTING']['RAW_DIR_NAME_IMP']:
        train = pd.read_csv(RAW_DIR_NAME + 'train.csv').drop(['original_row_id'], axis=1)
    else:
        train = pd.read_csv(RAW_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)