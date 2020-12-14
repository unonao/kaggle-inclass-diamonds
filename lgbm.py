import seaborn as sns
import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import argparse
import json
import numpy as np

from utils import load_datasets, load_target
from logs.logger import log_best
from models import LightGBM
import matplotlib as mpl
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 1000)

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

# log の設定
now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.INFO
)

feats = config['features']
target_name = config['target_name']
params = config['params']

NFOLDS = 5


def train_and_predict_lightgbm(X_train_all, y_train_all, X_test):

    # 学習前にy_trainに、log(y+1)で変換
    y_train_all = np.log(y_train_all + 1)  # np.log1p() でもOK

    y_preds = []
    models = []
    kf = KFold(n_splits=NFOLDS)
    for train_index, valid_index in kf.split(X_train_all):
        X_train, X_valid = (X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :])
        y_train, y_valid = (y_train_all.iloc[train_index], y_train_all.iloc[valid_index])

        # lgbmの実行
        lgbm = LightGBM()
        y_pred, y_valid_pred, model = lgbm.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params)

        # 結果の保存
        y_preds.append(y_pred)
        models.append(model)

        # スコア
        log_best(model, config['loss'])

    # CVスコア
    scores = [
        m.best_score['valid_0'][config['loss']] for m in models
    ]
    score = sum(scores) / len(scores)
    print('===CV scores===')
    print(scores)
    print(score)
    logging.info('===CV scores===')
    logging.info(scores)
    logging.info(score)

    # 重要度の出力
    feature_imp_np = np.zeros(X_train_all.shape[1])
    for model in models:
        feature_imp_np += model.feature_importance()/len(models)
    feature_imp = pd.DataFrame(sorted(zip(feature_imp_np, X_train_all.columns)), columns=['Value', 'Feature'])

    print(feature_imp)
    logging.info(feature_imp)

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('./logs/plots/sub_{0:%Y%m%d%H%M%S}_{1}.png'.format(now, score))

    # submitファイルの作成
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])

    y_sub = sum(y_preds) / len(y_preds)

    # 最後に、予測結果に対しexp(y)-1で逆変換
    y_sub = np.exp(y_sub) - 1  # np.expm1() でもOK

    sub[target_name] = y_sub

    sub.to_csv(
        './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
        index=False
    )


def main():
    logging.info('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

    logging.info('config: {}'.format(options.config))
    logging.info(feats)
    logging.info(params)

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(feats)
    y_train_all = load_target(target_name)
    logging.info("X_train_all shape: {}".format(X_train_all.shape))
    train_and_predict_lightgbm(X_train_all, y_train_all, X_test)


if __name__ == '__main__':
    main()
