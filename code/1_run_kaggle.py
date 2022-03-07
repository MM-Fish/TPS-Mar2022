from distutils.log import debug
from pyexpat import model
import sys
import os
import shutil
import datetime
import yaml
import json
import collections as cl
import warnings

sys.path.append('./src')
sys.path.append('./src/models/learning')
from src.runner import Runner
from src.util import Submission
from src.models.learning.model_lgb import ModelLGB
from src.models.learning.model_xgb import ModelXGB
from src.models.learning.model_keras import ModelKERAS
from src.models.learning.model_lda import ModelLDA

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
key_list = ['load_features', 'use_features', 'model_params', 'cv', 'dataset']

def exist_check(path, run_name):
    """学習ファイルの存在チェックと実行確認
    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)

    # 通常の実行確認
    print('特徴量ディレクトリ:{} で実行しますか？[Y/n]'.format(FEATURE_DIR_NAME))
    x = input('>> ')
    if x != 'Y':
        print('終了します')
        sys.exit(0)


def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def my_makedirs_remove(path):
    """引数のpathディレクトリを新規作成する（存在している場合は削除→新規作成）
    path:ディレクトリ名
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
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

    # pklからロードする特徴量の指定
    features = ['rawdata']

    features = ['pca_output',
                'tsne_output',
                'umap_output',
                'keras_0226_0937',
                'lda_0226_0509',
                'lgb_0222_0016',
                'lgb_0226_0545']
    
    # features = ['pca_output']

    # CVの設定.methodは[KFold, StratifiedKFold ,GroupKFold]から選択可能
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFoldの場合は、cv_targetに対象カラム名を設定する
    cv = {
        'method': 'KFold',
        'n_splits': 5,
        'random_state': 42,
        'shuffle': True,
        'cv_target': 'target'
    }
    if DEBUG is True:
        cv['n_splits'] = 2

    # # ######################################################
    # # 学習・推論 LightGBM ###################################

    # # run nameの設定
    # run_name = 'lgb'
    # run_name = run_name + suffix
    # out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # 諸々の設定
    # setting = {
    #     'run_name': run_name,  # run名 <- settingに不要？
    #     'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'target': 'target',  # 目的変数
    #     'calc_shap': False,  # shap値を計算するか否か
    #     'save_train_pred': True,  # trainデータでの推論値を保存するか否か
    #     'task_type': 'multiclass',
    #     'debug': DEBUG
    # }

    # # モデルのパラメータ
    # model_params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'multiclass',
    #     'metric': 'multi_logloss',
    #     'num_class' : 10,
    #     'num_round': 5000,
    #     'early_stopping_rounds': 1000,
    #     'verbose': 1000,
    #     'random_state': 999
    # }
    # if DEBUG is True:
    #     model_params['num_round'] = 1000
    
    # runner = Runner(ModelLGB, features, setting, model_params, cv, FEATURE_DIR_NAME, out_dir_name)

    # use_feature_name = runner.get_feature_name() # 今回の学習で使用する特徴量名を取得

    # # モデルのconfigをjsonで保存
    # value_list = [features, use_feature_name, model_params, cv, setting]
    # save_model_config(key_list, value_list, out_dir_name, run_name)

    # # runner.visualize_corr() # 相関係数を可視化して保存
    # if cv.get('method') == 'None':
    #     runner.run_train_all()  # 全データで学習
    #     runner.run_predict_all()  # 推論
    # else:
    #     runner.run_train_cv()  # 学習
    #     ModelLGB.calc_feature_importance(out_dir_name, run_name, use_feature_name)  # feature_importanceを計算
    #     runner.run_predict_cv()  # 推論

    # Submission.create_submission(run_name, out_dir_name, setting.get('target'), setting.get('task_type'))  # submit作成


    # # ######################################################
    # # 学習・推論 xgboost ###################################

    # # run nameの設定
    # run_name = 'xgb'
    # run_name = run_name + suffix
    # out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # 諸々の設定
    # setting = {
    #     'run_name': run_name,  # run名
    #     'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'target': 'target',  # 目的変数
    #     'calc_shap': False,  # shap値を計算するか否か
    #     'save_train_pred': True,  # trainデータでの推論値を保存するか否か
    #     'task_type': 'multiclass',
    #     'debug': DEBUG
    # }

    # model_params = {
    #     'objective': 'multi:softprob',
    #     'metric': 'mlogloss',
    #     'num_class': 10,
    #     'num_round': 5000,
    #     'early_stopping_rounds': 1000,
    #     'verbose': 1000,
    #     'random_state': 999
    # }
    # if DEBUG is True:
    #     model_params['num_round'] = 1000

    # runner = Runner(ModelXGB, features, setting, model_params, cv, FEATURE_DIR_NAME, out_dir_name)

    # use_feature_name = runner.get_feature_name() # 今回の学習で使用する特徴量名を取得

    # # モデルのconfigをjsonで保存
    # value_list = [features, use_feature_name, model_params, cv, setting]
    # save_model_config(key_list, value_list, out_dir_name, run_name)

    # # runner.visualize_corr() # 相関係数を可視化して保存
    # if cv.get('method') == 'None':
    #     runner.run_train_all()  # 全データで学習
    #     runner.run_predict_all()  # 推論
    # else:
    #     runner.run_train_cv()  # 学習
    #     ModelXGB.calc_feature_importance(out_dir_name, run_name, use_feature_name)  # feature_importanceを計算
    #     runner.run_predict_cv()  # 推論

    # Submission.create_submission(run_name, out_dir_name, setting.get('target'), setting.get('task_type'))  # submit作成


    # ######################################################
    # 学習・推論 keras ###################################

    # run nameの設定
    run_name = 'keras'
    run_name = run_name + suffix
    out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'target',  # 目的変数
        'calc_shap': False,  # shap値を計算するか否か
        'save_train_pred': True,  # trainデータでの推論値を保存するか否か
        'task_type': 'multiclass',
        'debug': DEBUG
    }

    model_params = {
        'task_type': 'multiclass',
        'epochs': 50,
        'batch_size': 999,
        'learning_rate': 0.1,
        'momentum': 0.8,
        'optimizer': 'SGD'
    }
    if DEBUG is True:
        model_params['epochs'] = 3

    runner = Runner(ModelKERAS, features, setting, model_params, cv, FEATURE_DIR_NAME, out_dir_name)

    use_feature_name = runner.get_feature_name() # 今回の学習で使用する特徴量名を取得

    # モデルのconfigをjsonで保存
    value_list = [features, use_feature_name, model_params, cv, setting]
    save_model_config(key_list, value_list, out_dir_name, run_name)

    # runner.visualize_corr() # 相関係数を可視化して保存
    if cv.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        runner.model_cls.calc_loss_curve(out_dir_name, run_name)  # feature_importanceを計算
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, out_dir_name, setting.get('target'), setting.get('task_type'))  # submit作成

    # ######################################################


    # ######################################################
    # 学習・推論 Linear Discriminant Analysis ###################################

    # # run nameの設定
    # run_name = 'lda'
    # run_name = run_name + suffix
    # out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # 諸々の設定
    # setting = {
    #     'run_name': run_name,  # run名
    #     'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
    #     'target': 'target',  # 目的変数
    #     'calc_shap': False,  # shap値を計算するか否か
    #     'save_train_pred': True,  # trainデータでの推論値を保存するか否か
    #     'task_type': 'multiclass',
    #     'debug': DEBUG
    # }

    # # モデルのパラメータ
    # model_params = {
    #     'solver': 'svd',
    #     'n_components': 5
    # }
    # if DEBUG is True:
    #     model_params['num_round'] = 1000
    
    # runner = Runner(ModelLDA, features, setting, model_params, cv, FEATURE_DIR_NAME, out_dir_name)

    # use_feature_name = runner.get_feature_name() # 今回の学習で使用する特徴量名を取得

    # # モデルのconfigをjsonで保存
    # value_list = [features, use_feature_name, model_params, cv, setting]
    # save_model_config(key_list, value_list, out_dir_name, run_name)

    # if cv.get('method') == 'None':
    #     runner.run_train_cv()  # 学習
    #     runner.run_predict_cv()  # 推論
    # else:
    #     runner.run_train_cv()  # 学習
    #     runner.run_predict_cv()  # 推論

    # # 散布図の作成(散布図は全ての学習データで作成する)
    # ModelLDA.plot_scatter(out_dir_name, run_name, runner.train_x, runner.train_y, model_params)

    # Submission.create_submission(run_name, out_dir_name, setting.get('target'), setting.get('task_type'))  # submit作成