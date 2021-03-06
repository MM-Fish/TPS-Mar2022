from dataclasses import dataclass, asdict
from distutils.log import debug
from pyexpat import model
from random import shuffle
import sys
import os
import shutil
import datetime
import yaml
import json
import collections as cl
import warnings
from google.cloud import storage
import glob

sys.path.append('./src')
sys.path.append('./src/models/learning')
from src.params import Cv, KFoldParams, StratifiedKFoldParams, GroupKFoldParams, TimeSeriesSplitParams, HoldOutParams, Setting, PreprocessingParams, ModelParams, LgbParams
from src.runner import Runner
from src.util import Submission
from src.models.learning.model_lgb import ModelLGB
from src.models.learning.model_xgb import ModelXGB
from src.models.learning.model_keras import ModelKERAS
from src.models.learning.model_lstm import ModelLSTM
from src.models.learning.model_lda import ModelLDA

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']

# 前処理後
RAW_DIR_NAME = yml['SETTING']['RAW_DIR_NAME_IMP']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME_IMP']

# ディレクトリ確認
print(RAW_DIR_NAME)
print(FEATURE_DIR_NAME)

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
key_list = ['load_features', 'use_features', 'model_params', 'cv', 'dataset']

BUCKET_NAME = 'kaggleops-bucket-msm'
BLOB_NAME = 'models'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs-key.json'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

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

def upload_gcs_from_directory(bucket: storage.bucket.Bucket, directory_path: str, blob_name: str, root_position=2):
  rel_paths = glob.glob(directory_path + '/**', recursive=True)
  for local_file in rel_paths:
    remote_path = f'{blob_name}/{"/".join(local_file.split(os.sep)[root_position:])}'
    if os.path.isfile(local_file):
      blob = bucket.blob(remote_path)
      blob.upload_from_filename(local_file)


########################################################
####### 以下が実行コード

if __name__ == '__main__':
    ts_col = 'date_obj'
    target = 'congestion'
    DEBUG = True
    
    now = datetime.datetime.now()
    suffix = now.strftime("_%m%d_%H%M")
    if DEBUG is True:
        suffix += '-debug'

    # pklからロードする特徴量の指定
    features = [
        "shift_3days",
        "datetime_element",
        'direction_dummies',
        # "coordinate",
        # "decompose_direction",
        'accum_minutes_half_day',
        'agg_by_am',
        "agg_shift_by_date",
        "diff_3days",
        'is_weekend'
        ]

    # バリデーションの設定
    cv = Cv(
        cv_class = KFoldParams('KFold'),
        # cv_class = StratifiedKFoldParams('StratifiedKFold'),
        # cv_class = GroupKFoldParams('GroupKFold', target),
        # cv_class = TimeSeriesSplitParams('TimeSeriesSplit', False, ts_col),
        # cv_class = HoldOutParams('HoldOut', 844155),
        n_splits = 2,
        random_state = 42,
        shuffle = True,
    )
    if DEBUG is True:
        cv.n_splits = 2

    # 諸々の設定
    setting = Setting(
        feature_directory = FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        target = target,  # 目的変数
        id_column = 'row_id', # 行番号
        calc_shap = False,  # shap値を計算するか否か
        save_train_pred = True,  # trainデータでの推論値を保存するか否か
        task_type = 'regression',
        debug = DEBUG
    )

    # ######################################################
    # 学習・推論 LightGBM ###################################

    # run name定義とディレクトリ作成
    run_name = 'lgb'
    run_name = run_name + suffix
    out_dir_name = MODEL_DIR_NAME + run_name + '/'
    my_makedirs(out_dir_name)

    # モデルのパラメタ
    model_params = LgbParams(
            boosting_type =  'gbdt',
            objective = 'regression',
            metric = 'mae',
            num_round = 5000,
            early_stopping_rounds = 1000,
            verbose = 1000,
            random_state = 999,
            optuna = False,
    )
    if DEBUG is True:
        model_params.num_round = 500
        model_params.early_stopping_rounds = 100

    # 前処理の設定
    preprocessing_params = PreprocessingParams(
        cv_method = cv.cv_class.method, 
        ts_col = ts_col, 
        id_col = setting.id_column, 
        sort_cols = None, 
        sort_col_feats = None
    )

    params = ModelParams(
        run_name = run_name,
        model_params = model_params,
        preproccesing_params = preprocessing_params
    )

    runner = Runner(ModelLGB, features, setting, params, cv, FEATURE_DIR_NAME, out_dir_name)

    # モデルのconfigをjsonで保存
    use_feature_name = runner.use_feature_name # 今回の学習で使用する特徴量名を取得
    value_list = [features, use_feature_name, asdict(model_params), asdict(cv), asdict(setting)]
    save_model_config(key_list, value_list, out_dir_name, run_name)

    # runner.visualize_corr() # 相関係数を可視化して保存
    if cv.cv_class == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        ModelLGB.calc_loss_curve(out_dir_name, run_name)  # loss_curveを出力
        ModelLGB.calc_feature_importance(out_dir_name, run_name, use_feature_name)  # feature_importanceを計算
        # ModelLGB.save_optuna_best_params(out_dir_name, run_name, use_feature_name)  # optunaのbest paramsを保存
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, out_dir_name, setting.target, setting.task_type)  # submit作成

    # upload to GCS
    if DEBUG == False:
        directry_path = f'../models/{run_name}/'
        upload_gcs_from_directory(bucket, directry_path, BLOB_NAME)


    # # ######################################################
    # # 学習・推論 xgboost ###################################

    # # run nameの設定
    # run_name = 'xgb'
    # run_name = run_name + suffix
    # out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # 諸々の設定
    # model_params = {
    #     'run_name': run_name,
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
    # 学習・推論 Linear Discriminant Analysis ###################################

    # # run nameの設定
    # run_name = 'lda'
    # run_name = run_name + suffix
    # out_dir_name = MODEL_DIR_NAME + run_name + '/'

    # # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    # my_makedirs(out_dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # モデルのパラメータ
    # model_params = {
    #     'run_name': run_name,
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