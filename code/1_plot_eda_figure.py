import sys,os
import yaml
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('./src')
sys.path.append('./src/figures')
from src.figures.line_plots import PlotSeries5axis
from src.figures.count_plots import PlotCount
from src.figures.dist_plots import PlotDist2axis

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
EDA_DIR_NAME = yml['SETTING']['EDA_DIR_NAME']  # EDAに関する情報を格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']

def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def set_default(obj):
    """json出力の際にset型のオブジェクトをリストに変更する
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':
    # ###############################
    # # 線形グラフ
    # plot_type = 'line_plots'
    # out_dir_name = EDA_DIR_NAME + plot_type + '/'

    # setting = {
    #     'run_name': '',  # run名
    #     'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'out_dir_name': out_dir_name #結果出力用ディレクトリ
    # }

    # model_params_list = [
    # {
    #     'col': 'y',
    #     'row': 'x',
    #     'x': 'accum_minutes',
    #     'y': 'congestion',
    #     'z': 'direction',
    #     'is_xlim': True,
    #     'is_ylim': True
    # },
    # {
    #     'col': 'month',
    #     'row': 'day',
    #     'x': 'accum_minutes',
    #     'y': 'congestion',
    #     'z': 'direction',
    #     'is_xlim': True,
    #     'is_ylim': True
    # }]

    # features = ['rawdata', 'congestion', 'datetime_info', 'accum_minutes']

    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    # for model_params in model_params_list:
    #     plot_series_5axis = PlotSeries5axis(model_params, features, setting)
    #     plot_series_5axis.run_and_save()
    #     del plot_series_5axis
    

    # ###############################
    # # 各列のデータ数を計測
    # plot_type = 'count_plots'
    # out_dir_name = EDA_DIR_NAME + plot_type + '/'

    # setting = {
    #     'run_name': '',  # run名
    #     'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'out_dir_name': out_dir_name #結果出力用ディレクトリ
    # }

    # model_params = {
    #     'cols': ['direction', 'congestion'],
    #     'target': 'congestion'
    # }

    # features = ['rawdata', 'congestion', 'datetime_info', 'accum_minutes']

    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    # plot_counts = PlotCount(model_params, features, setting)
    # plot_counts.run_and_save()
    # del plot_counts
    

    ###############################
    # 確率分布
    plot_type = 'dist_plots'
    out_dir_name = EDA_DIR_NAME + plot_type + '/'

    setting = {
        'run_name': '',  # run名
        'feature_dir_name': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'out_dir_name': out_dir_name #結果出力用ディレクトリ
    }

    model_params_list = [
    {
        'col': 'x',
        'row': 'direction',
        'target': 'congestion'
    },
    {
        'col': 'y',
        'row': 'direction',
        'target': 'congestion'
    },
    {
        'col': 'x',
        'row': 'y',
        'target': 'congestion'
    }]

    features = ['rawdata', 'congestion']

    my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる
    for model_params in model_params_list:
        plot_dist_2axis = PlotDist2axis(model_params, features, setting)
        plot_dist_2axis.run_and_save()
        del plot_dist_2axis
    