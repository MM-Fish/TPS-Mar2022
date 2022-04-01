# playgound-series-mar-2022

## コンペ概要
- アメリカの地下鉄の24時間の交通量予測
- 時系列コンペ
https://www.kaggle.com/c/tabular-playground-series-mar-2022/overview


### データ
row_id - a unique identifier for this instance
time - the 20-minute period in which each measurement was taken
x - the east-west midpoint coordinate of the roadway
y - the north-south midpoint coordinate of the roadway
direction - the direction of travel of the roadway. EB indicates "eastbound" travel, for example, while SW indicates a "southwest" direction of travel.
congestion - congestion levels for the roadway during each hour; the target. The congestion measurements have been normalized to the range 0 to 100.

#### 固有値（train, test共通）
- x: range(0, 3, 1)
- y: range(0, 4, 1)
- direction: NB, SB, EB, WB, NE, SE, NW, SW
- congestion: 0-100
- time
    - train: (1991-04-01 00:00:00, 1991-09-30 11:40:00, 20min)
    - test: (1991-09-30 12:00:00, 1991-09-30 23:40:00, 20min)

#### 特徴
direction, x, y, timeの組み合わせで1レコードを成す
- direction, x, yの組み合わせ
    - unique_count: 65
 
- time(20分単位の時系列)
    - train
        - unique_count: 13059
        - (本来ならば)72/日 * 182日 + 36/日（最後の半日）= 13140
        - 欠損値あり
    - test
        - unique_count: 36
        - 36/日
        - 欠損値なし

- shape
    - train: (848835, 6)
        - 13059 × 65 = 848835
    - test: (2340, 5)
        - 35 × 65 = 2340

validationに使用
- 1991-09-29 00:00:00 ~
    row_id: 841815

- 1991-09-29 12:00:00 ~
    row_id: 844155

- 1991-09-30 00:00:00 ~
    row_id: 846495


## 22/03/03
コンペ参加


# 22/03/06
EDA
データ理解
first submit

# 22/03/8-10
1日1時間ずつ
lstmの実装
kaggleで実行出来るようにした

## 22/03/11
時間に欠損があるかも？

## 22/03/12
lstm 1回目
時間に欠損行があるせいでlstmのデータreshapeが上手くいかないので、
欠損行を補完したデータrowdataの作成->特徴量作成->欠損がある日を全て削除してlstmの実装

（リファクタリング）
rawデータを変えたい時。今回competationをつけた
get_dummiesなどをどこで行うか。runnerで処理を行なってしまうと、今のログ設計では使った列を正しく把握出来ない。
train_x, train_yをnumpyで返すように全体を構成し直したい。
lstmなどでは、numpyが基本になっているため

(idea)
congestionの特徴量を使用したい
前日の最大値、最小値などを使用したい
同じaccum_minutesの最大値、最小値、平均値
同じcoordinateの最大値、最小値、平均値

(error)
kaggle notebookだとtpu、gpuがうまく利用出来ない。
kerasをバージョンアップしたのが原因？
```
2022-03-12 03:06:21.400591: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/lib
2022-03-12 03:06:21.400655: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-03-12 03:06:21.400694: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (3a3b9dcf2352): /proc/driver/nvidia/version does not exist
2022-03-12 03:06:21.401853: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

## 22/03/13
(リファクタリング)
前処理のコードをどこに記載するか
前処理のデータと通常のデータと両方で特徴量を作成するようにして、フォルダを使い分けるのが良さそう

lgbをダミー変数化、欠損行を補完したデータを使って作成した特徴量
を使用して提出

(error)
- shapeが819000になっている
    欠損行の作成したのち、差分がうまくいっていない？

- 特徴量作成の際にtrainのundefinedエラーが発生する
    特徴量作成クラス内でtrain=, test=をしてしまうと、train, testがglobal変数を参照出来なくなる？

### 新規特徴量
- カテゴリカル変数×数値変数の特徴量エンジニアリングの実装
    時間とカテゴリカル変数をgroupbyして、congestionの最大値、最小値、平均、標準偏差


今週やることをkanbanflowに作成

## 22/03/14
xfeatの導入
参考ページ
https://acro-engineer.hatenablog.com/entry/2020/12/15/120000

リポジトリ
https://github.com/pfnet-research/xfeat/blob/b199c3cdacef408b3d2b1d920b22c243cfe0667c/xfeat/num_encoder/_arithmetic_combinations.py

## 22/03/17
imputationデータ
欠損行を補完したデータの結果があまり伸びないので、データの処理がうまくいっている確認した

## 22/03/20
差分、移動平均とshfitのクラスを作成した

(point)
線形グラフから時間的なずれよりも、座標のずれの方がcongestionに与える影響は大きい
移動平均は平均よりも中央値の方が良い
50日移動平均のスコアの違い
- 中央値：4.935
- 平均値：5.047

## 22/03/22
lgb-0322-1220の実行
[5000]	training's l1: 4.99007	valid_1's l1: 5.99693
トレーニングとバリデーションのスコアに乖離がある
→リークしているから？

(idea)
時系列データのバリデーションを修正する
https://blog.amedama.jp/entry/time-series-cv → 済

lstmのlossが全然下がらない。
予測値が50前後（中央値付近）にしかならない。

(リファクタリング)
kfoldに使う特徴量はクラス名とカラム名を揃えたい。
load_time_seriesなどで使用


## 22/03/23
時系列データのバリデーションを実装した
training's l1: 4.85846	valid_1's l1: 6.7615
トレーニングとバリデーションの乖離がより大きくなった
→データ数が少ないかも？ train_index → 151190レコード()


## 22/03/26
(point)
時系列に関するを特徴量に入れる場合は、kfoldよりholdoutの方が高くなる場合がある。
lgb_0326_0533とlgb_0326_0536の比較
同じ特徴量でバリデーションを変更
holdoutの方がスコアが高い
→おそらくリークしていないため

(idea)
回帰ではなく分類問題にしてみる？


(point)
accum_minutes_half_dayとaccum_minutesの比較
accum_minutes_half_dayの方が僅かに良い
- accum_minutes_half_day
    run_name: lgb_0326_1253
    score: 5.266
- accum_minutes
    run_name: lgb_0327_0037
    score: 5.284

同じバリデーションを使用
"cv": {
    "method": "HoldOut",
    "min_id": 841815,
    "n_splits": 1,
    "random_state": 42,
    "shuffle": true,
    "cv_target": "congestion"
},

"load_features": [
    "shift_3days",
    "datetime_element",
    "coordinate",
    "decompose_direction",
    "accum_minutes"( or "accum_minutes_half_day" ),
    "agg_shift_by_date",
    "rolling_30days",
    "diff_3days",
    "is_weekend"
],


## 22/03/27
午前の統計量を特徴量として追加
scoreが0.083減少
- 午前の統計量無し
    run_name: lgb_0327_0037
    score: 5.284
- 午前の統計量有り
    run_name: lgb_0327_0118
    score: 5.201

autoMLモデルを使ってみる

(リファクタリング)
rollingの際に欠損値を含んでも可能にしたい


特徴量を追加した方がスコアが低い
- lgb_0328_1409
    - 5.697

- lgb_0328_2339
    - 5.755


## 22/03/30
notebookの実装
欠損値を平均で埋めてみる
https://www.kaggle.com/code/martynovandrey/tps-mar-22-fe-model-selection



## 22/03/31
public in top 15%
private in top 50%
shake down
rank 1st comment
https://www.kaggle.com/competitions/tabular-playground-series-mar-2022/discussion/316271


### 所感
1ヶ月通してコンペに初参加
pandas, numpyの前処理などに時間をかけ過ぎ
関数化、綺麗なコード管理をしようとし過ぎ
→学びや発見がないと集中力が切れやすい。。
次回は、最短でsubmission出来るように、可視化や関数化などに拘りすぎずに進めるのが良さそう
例えば、日曜の午後に1週間の振り返りをするなど。。。