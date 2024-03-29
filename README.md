# kaggle-inclass-diamonds

https://www.kaggle.com/c/predict-diamonds-price/overview

プライベートコンペでのダイヤモンド価格予測

# memo

- x,y,z の欠損値について、中央値で埋めると CV と private が良くなり、public が悪くなった（trust CV!)
- カテゴリ変数は、ワンホットよりも順序を意識したラベルエンコーディング
- boxcox で直すと精度向上
- polynomial features をいくつか導入。lightgbm のスコアに寄与した上位数個を組み込むと良くなる

# Structures

```
.
├── configs
├── data
│   ├── input
│   ├── (original)
│   └── output
├── features
├── logs
├── models
├── notebooks
├── scripts
├── utils
├── .gitignore
├── README.md
└── run.py

```

## configs

実験ごとに利用している特徴量とパラメータの管理をする。
json ファイルで記載。

## data

コンペのデータ置き場。
input は操作しない。
output は出力結果を保存するだけ。
形式は `sub_(year-month-day-hour-min)_(score).csv`

## features

自分で生成した特徴量諸々
create.py に作成したクラスを元に生成される

## logs

ログの結果。
形式は `log_(year-month-day-hour-min)_(score).log`
提出用の csv ファイルと照合できるようにする。

- 利用した特徴量
- train の shape
- 学習機のパラメータ
- cv のスコア

## models

学習機のフォルダ。別のコンペでも使い回せることを意識して入出力を構築

- 入力 dataframe, prameter
- 出力 予測結果

## notebook

試行錯誤するための notebook
ここで試行錯誤した結果を適切なフォルダの python ファイルに取り込む

## scripts

汎用的な python ファイルを配置

## utils

汎用的な python 関数を配置し、呼び出せるように

# フォルダ構成の参考

- flowlight さん [優勝したリポジトリ](https://github.com/flowlight0/talkingdata-adtracking-fraud-detection)
- u++さん [【Kaggle のフォルダ構成や管理方法】タイタニック用の GitHub リポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- amaotone さん [Kaggle で使える Feather 形式を利用した特徴量管理法](https://amalog.hateblo.jp/entry/kaggle-feature-management)
- amaotone さん[LightGBM の callback を利用して学習履歴をロガー経由で出力する](https://amalog.hateblo.jp/entry/lightgbm-logging-callback)
