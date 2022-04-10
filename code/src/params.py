from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Union

@dataclass
class Setting:
    feature_directory: str # 特徴量の読み込み先ディレクトリ
    target: str # 目的変数の列名
    id_column: str # 行番号の列名
    calc_shap: bool  # shap値を計算するか否か
    save_train_pred: bool  # trainデータでの推論値を保存するか否か
    task_type: str
    debug: bool

# CV関連のクラス
@dataclass
class KFoldParams:
    method: str
    
@dataclass
class StratifiedKFoldParams:
    method: str

@dataclass
class GroupKFoldParams:
    method: str
    target: str

@dataclass
class TimeSeriesSplitParams:
    method: str
    clipping: bool
    ts_col: str

@dataclass
class HoldOutParams:
    method: str
    min_id: int

@dataclass
class Cv:
    cv_class: Optional[Union[KFoldParams, StratifiedKFoldParams, GroupKFoldParams, TimeSeriesSplitParams, HoldOutParams]]
    n_splits: int
    random_state: int
    shuffle: bool


# runner内での前処理関連のクラス
@dataclass
class PreprocessingParams:
    cv_method: str
    ts_col: str
    id_col: str
    sort_cols: Optional[List[str]]
    sort_col_feats: Optional[List[str]]

@dataclass
class PreprocessingSettings:
    keep_index: Optional[np.ndarray]
    ts_index: np.ndarray
    id_idx: np.ndarray


# モデル関連のクラス
@dataclass
class LgbParams:
    boosting_type: str
    objective: str
    metric: str
    num_round: int
    early_stopping_rounds: int
    verbose: int
    random_state: int
    optuna: bool

@dataclass
class LstmParams:
    task_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    optimizer: str

@dataclass
class ModelParams:
    run_name: str
    model_params: Union[LgbParams, LstmParams]
    preproccesing_params: PreprocessingParams