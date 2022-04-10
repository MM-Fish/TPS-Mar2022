import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import TimeSeriesSplit

class MovingWindowKFold(TimeSeriesSplit):
    """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

    def __init__(self, ts_col, clipping=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.ts_col = ts_col
        # trainのデータ数を統一するフラグ
        self.clipping = clipping

    def split(self, ts_df: pd.DataFrame, *args, **kwargs):
        # 時系列でソートする
        sorted_unique_ts_df = ts_df.sort_values(self.ts_col).drop_duplicates()
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize
        # スーパークラスのメソッドで添字を計算する
        for train_index, test_index in super().split(sorted_unique_ts_df, *args, **kwargs):
            
            # 添字を元々の DataFrame の iloc として使える値に変換する
            train_ts = sorted_unique_ts_df.iloc[train_index][self.ts_col].values
            test_ts = sorted_unique_ts_df.iloc[test_index][self.ts_col].values

            train_iloc_index = ts_df.loc[ts_df[self.ts_col].isin(train_ts), :].index
            test_iloc_index = ts_df.loc[ts_df[self.ts_col].isin(test_ts), :].index

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])