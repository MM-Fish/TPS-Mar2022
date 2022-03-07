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

#### 特徴
- shape
    - train
    - test

- unique
    - x: range(0, 3, 1)
    - y: range(0, 4, 1)
    - direction: NB, SB, EB, WB, NE, SE, NW, SW
    - congestion: 0-100

#### 要約
x, y, directionが64通り存在する



## 22/03/03
コンペ参加


# 22/03/06
EDA
データ理解
