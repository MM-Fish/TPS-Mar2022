import pandas as pd
import numpy as np
from datetime import timedelta

# 時系列データにおける欠損行の補完
def impute_time_series(df, time_col, freq):
  df['merged_feat'] = df['x'].map(lambda x: str(x) + '_') + df['y'].map(lambda x: str(x) + '_') + df['direction']

  df[time_col] = pd.to_datetime(df[time_col])

  unique_time = pd.DataFrame(df[time_col].unique())
  max_time = unique_time.max()[0]
  min_time = unique_time.min()[0]

  # 完全な時系列 * 特徴量のnp.arrayを作成
  absolute_series_arrary = np.meshgrid(np.array(pd.DataFrame(pd.date_range(start=min_time, end=max_time, freq=freq))), df['merged_feat'].unique())

  absolute_series_index = pd.Series(absolute_series_arrary[0].flatten())
  absolute_series_values = pd.Series(absolute_series_arrary[1].flatten()).str.split('_', expand=True).rename(columns = {
    0: 'x',
    1: 'y',
    2: 'direction'
  })

  absolute_series = pd.concat([absolute_series_index, absolute_series_values], axis=1).rename(columns={0: 'time'})
  absolute_series['x'] = absolute_series['x'].map(int)
  absolute_series['y'] = absolute_series['y'].map(int)

  df_imputation = pd.merge(df, absolute_series, how='outer')
  print(df_imputation.shape)
  return df_imputation.sort_values(['time', 'x', 'y', 'direction', 'congestion'])


time_col = 'time'
freq = '20min'
df = train.copy()
impute_time_series(df, time_col, freq)

