import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

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
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    errors_all = []
    # 高斯朴素贝叶斯分类器
    errors = []
    print("Gaussian")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        gnb = GaussianNB()
        y_pred = gnb.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:,0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # K近邻分类器
    errors = []
    print("KNeighbors")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        neigh = KNeighborsClassifier(n_neighbors=3)
        y_pred = neigh.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(errors)
    print("****************************")
    # 决策树分类器
    errors = []
    print("DecisionTree")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = DecisionTreeClassifier()
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # 随机森林
    errors = []
    print("RandomForest")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = RandomForestClassifier(max_depth=20, random_state=0)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # AdaBoost
    errors = []
    print("AdaBoost")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = AdaBoostClassifier(base_estimator=None)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # Bagging
    errors = []
    print("GradientBoosting")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = BaggingClassifier(n_estimators=20)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # GradientBoosting
    errors = []
    print("GradientBoosting")
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = GradientBoostingClassifier(n_estimators=2)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)

    utils.cdf_figure(errors_all)


if __name__ == '__main__':
    main()
    # read_data('./data/data_2g.csv')
    # iris = datasets.load_iris()
    # print(iris.data)
    # print(type(iris.data))
