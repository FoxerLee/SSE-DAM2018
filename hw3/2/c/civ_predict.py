import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from pandas import DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import civ
import util


def main():
    print("Generator train data!")
    months = ['02', '03', '04']
    overall = '234'
    label_month = '05'
    train_datas = civ.train_generator(months, overall, label_month)
    train_datas.set_index(['vipno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X_train = np.delete(train, train.shape[1] - 1, axis=1)
    y_train = train[:, train.shape[1] - 1]

    print("Generator test data!")
    months = ['05', '06', '07']
    overall = '567'
    label_month = '08'
    test_datas = civ.train_generator(months, overall, label_month, predict=True)
    y_test = test_datas.index.tolist()
    X_test = test_datas.drop(columns=['vipno'], inplace=False).as_matrix()

    # K近邻分类器
    f = open('predict/civ/1552674_2civ_KNeighborsRegressor.txt', 'w')
    clf = KNeighborsRegressor()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')
    f.close()

    # 决策树分类器
    f = open('predict/civ/1552674_2civ_DecisionTreeRegressor.txt', 'w')
    clf = DecisionTreeRegressor()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')
    f.close()

    # 随机森林
    f = open('predict/civ/1552674_2civ_RandomForestRegressor.txt', 'w')
    clf = RandomForestRegressor(max_depth=20, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')
    f.close()

    # AdaBoost
    f = open('predict/civ/1552674_2civ_AdaBoostRegressor.txt', 'w')
    clf = AdaBoostRegressor(n_estimators=90, learning_rate=0.02)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')
    f.close()

    # Bagging
    f = open('predict/civ/1552674_2civ_BaggingRegressor.txt', 'w')
    clf = BaggingRegressor(n_estimators=20)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')

    f.close()

    # GradientBoosting
    f = open('predict/civ/1552674_2civ_GradientBoostingRegressor.txt', 'w')
    clf = GradientBoostingRegressor(learning_rate=0.02, n_estimators=50,
                                         max_depth=13, max_features=19, subsample=0.6)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        f.write(str(i) + ','+str(j)+'\n')
    f.close()


if __name__ == '__main__':
    main()

