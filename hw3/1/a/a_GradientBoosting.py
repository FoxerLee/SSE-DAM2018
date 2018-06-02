import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import warnings
warnings.filterwarnings('ignore')

from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import utils  # 减少代码重复


def main():
    ll_data_2g = utils.gongcan_to_ll()
    train_data = utils.ll_to_grid(ll_data_2g)

    # print(train_data)
    # 删除原有的ID，不作为训练特征
    for i in range(1, 8):
        train_data.drop(['RNCID_'+str(i)], axis=1, inplace=True)
        train_data.drop(['CellID_'+str(i)], axis=1, inplace=True)
    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)

    # features和labels
    X = train_data.drop(['MRTime', 'Longitude', 'Latitude',
                         'Num_connected', 'grid_num'], axis=1, inplace=False).as_matrix()
    y = train_data[['grid_num', 'Longitude', 'Latitude']].as_matrix()

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    # random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # errors_all = []
    # GradientBoosting 调参
    # start = datetime.datetime.now()
    # errors = []
    # overall_pres = []
    # top10_pres = []
    # top10_recalls = []
    # top10_fs = []
    # print(y[:,0])
    print("GradientBoosting")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

    param_test1 = {'n_estimators': range(10, 61, 10),
                   'learning_rate': np.arange(0.01, 0.1, 10)}
    param_test2 = {'max_depth': range(3, 14, 2)}
    param_test3 = {'max_features': range(7, 20, 2),
                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300),
                            param_grid=param_test1, scoring='f1_micro', cv=5)
    gsearch1.fit(np.delete(X, 0, axis=1), y[:, 0])
    print("Best param: {}".format(gsearch1.best_params_))
    print("Best score: {}".format(gsearch1.best_score_))

    print("****************************")


if __name__ == '__main__':
    main()