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
        - 13059 × 65 = 84835
    - test: (2340, 5)
        - 35 × 65 = 2340


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
前処理後のデータの情報を取得しないときちんとどのデータを使ったか追えない

(idea)
congestionの特徴量を使用したい
前日の最大値、最小値などを使用したい
同じaccum_minutesの最大値、最小値、平均値
同じcoordinateの最大値、最小値、平均値