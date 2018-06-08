import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import warnings
warnings.filterwarnings('ignore')

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
    top10_pres_all = []
    top10_recalls_all = []
    top10_fs_all = []
    overall_pres_all = []

    # 高斯朴素贝叶斯分类器
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        gnb = GaussianNB()
        y_pred = gnb.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("Gaussian")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # K近邻分类器
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        neigh = KNeighborsClassifier(n_neighbors=3)
        y_pred = neigh.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("KNeighbors")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # 决策树分类器
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = DecisionTreeClassifier()
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("DecisionTree")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # 随机森林
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = RandomForestClassifier(max_depth=20, random_state=0)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("RandomForest")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # AdaBoost
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20), learning_rate=0.01, n_estimators=30,
                                 algorithm='SAMME.R')
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("AdaBoost")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # Bagging
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = BaggingClassifier(n_estimators=20)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("Bagging")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    # GradientBoosting
    start = datetime.datetime.now()
    errors = []
    overall_pres = []
    top10_pres = []
    top10_recalls = []
    top10_fs = []
    for i in range(10):
        print(i)
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.01)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:,0]).predict(np.delete(X_test, 0, axis=1))
        overall_pre, top10_pre, top10_recall, top10_f = utils.precision_recall(y_test[:, 0], y_pred)
        overall_pres.append(overall_pre)
        top10_pres.append(top10_pre)
        top10_recalls.append(top10_recall)
        top10_fs.append(top10_f)
        errors.append(utils.pos_error(y_test, y_pred))

    print("GradientBoosting")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Top10 precision: %.3f" % np.array(top10_pres).mean(axis=0).mean())
    print("Top10 recall: %.3f" % np.array(top10_recalls).mean(axis=0).mean())
    print("Top10 f-measurement: %.3f" % np.array(top10_fs).mean(axis=0).mean())
    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Time spend: {}".format(datetime.datetime.now() - start))
    errors_all.append(errors)
    top10_recalls_all.append(np.array(top10_recalls).mean(axis=0).mean())
    top10_pres_all.append(np.array(top10_pres).mean(axis=0).mean())
    overall_pres_all.append(np.mean(np.array(overall_pres)))
    top10_fs_all.append(np.array(top10_fs).mean(axis=0).mean())
    print("****************************")

    utils.cdf_figure(errors_all)
    utils.res_figure(overall_pres_all, top10_pres_all, top10_recalls_all, top10_fs_all)


if __name__ == '__main__':
    main()
    # utils.time_figure([])

