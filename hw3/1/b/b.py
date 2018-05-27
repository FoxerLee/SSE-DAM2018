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

from scipy.signal import savgol_filter

import utils  # 减少代码重复

# 左下角坐标
lb_Longitude = 121.20120490000001
lb_Latitude = 31.28175691
# 右上角坐标
rt_Longitude = 121.2183295
rt_Latitude = 31.29339344
# 格子个数是 82*65
y_box_num = 65
X_box_num = 82
# 每一个格子所占的经纬度
per_lon = (rt_Longitude - lb_Longitude)/X_box_num
per_lat = (rt_Latitude - lb_Latitude)/y_box_num


def cosine_law(a, b, c):
    return (a*a + b*b - c*c)/ (2 * a * b)


def fix_error(y_train, y_test, y_pred):
    ll_pred = []

    for i in range(len(y_pred)):
        # lon = lb_Longitude + (y % X_box_num) * ()
        # lat = lb_Latitude +
        X_box = int(y_pred[i] % X_box_num)
        y_box = int(y_pred[i] / X_box_num) + 1
        if X_box == 0:
            X_box = X_box_num
            y_box -= 1
        lon = lb_Longitude + per_lon * X_box
        lat = lb_Latitude + per_lat * y_box
        # i 值是作为预测值的编号，方便之后比较，对于作为train的部分，编号都赋值 -1
        ll_pred.append([lon, lat, y_test[i][3], y_test[i][4], i])
    y_train = np.delete(y_train, 0, axis=1)
    ids = np.empty(len(y_train))
    ids[:] = -1
    y_train = np.insert(y_train, 4, values=ids, axis=1)

    y_train = DataFrame(y_train, columns=['Longitude', 'Latitude', 'IMSI', 'MRTime', 'id'])
    ll_pred = DataFrame(ll_pred, columns=['Longitude', 'Latitude', 'IMSI', 'MRTime', 'id'])
    # print(y_train)
    # print(ll_pred)
    datas = pd.concat([y_train, ll_pred], axis=0)
    # datas[['MRTime']] = datas[['MRTime']].astype(String)
    datas.set_index(['MRTime'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    datas.set_index(['IMSI'], inplace=True, drop=False)
    for IMSI in set(datas.index.tolist()):
        print(datas.loc[IMSI])
        MS_datas = datas.loc[IMSI].as_matrix()
        x = MS_datas[:, 0]
        y = MS_datas[:, 1]
        plt.scatter(x, y)
        plt.show()

        # for r in range(len(MS_datas)):
        #     if MS_datas[r][4] != -1:
        #         before = MS_datas[r-1]
        #         later = MS_datas[r+1]
        #         pre = MS_datas[r]
        #
        #         a = utils.haversine(pre[0], pre[1], later[0], later[1])
        #         b = utils.haversine(before[0], before[1], pre[0], pre[1])
        #         c = utils.haversine(before[0], before[1], later[0], later[1])
        #         print(a)
        #         print(b)
        #         print(c)
        #         cosC = cosine_law(a, b, c)
        #         angle = np.arccos(cosC)*360/2/np.pi
        #
        #         print(angle)
        #         print("===")


def main():
    ll_data_2g = utils.gongcan_to_ll()
    train_data = utils.ll_to_grid(ll_data_2g)

    # print(train_data)
    # 删除原有的ID，不作为训练特征
    for i in range(1, 8):
        train_data.drop(['RNCID_' + str(i)], axis=1, inplace=True)
        train_data.drop(['CellID_' + str(i)], axis=1, inplace=True)
    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)

    # features和labels
    X = train_data.drop(['MRTime', 'Longitude', 'Latitude',
                         'Num_connected', 'grid_num'], axis=1, inplace=False).as_matrix()
    y = train_data[['grid_num', 'Longitude', 'Latitude', 'IMSI', 'MRTime']].as_matrix()

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    errors_all = []
    # 高斯朴素贝叶斯分类器
    errors = []
    print("Gaussian")
    # for i in range(10):
        # 切分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[0])

    gnb = GaussianNB()
    y_pred = gnb.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    fix_error(y_train, y_test, y_pred)
    overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    errors.append(utils.pos_error(y_test, y_pred))
    errors_all.append(errors)
    print("****************************")
    # K近邻分类器
    # errors = []
    # print("KNeighbors")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     neigh = KNeighborsClassifier(n_neighbors=3)
    #     y_pred = neigh.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(errors)
    # print("****************************")
    # # 决策树分类器
    # errors = []
    # print("DecisionTree")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     clf = DecisionTreeClassifier()
    #     y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(utils.pos_error(y_test, y_pred))
    # errors_all.append(errors)
    # print("****************************")
    # # 随机森林
    # errors = []
    # print("RandomForest")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     clf = RandomForestClassifier(max_depth=20, random_state=0)
    #     y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(utils.pos_error(y_test, y_pred))
    # errors_all.append(errors)
    # print("****************************")
    # # AdaBoost
    # errors = []
    # print("AdaBoost")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     clf = AdaBoostClassifier(base_estimator=None)
    #     y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(utils.pos_error(y_test, y_pred))
    # errors_all.append(errors)
    # print("****************************")
    # # Bagging
    # errors = []
    # print("GradientBoosting")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     clf = BaggingClassifier(n_estimators=20)
    #     y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(utils.pos_error(y_test, y_pred))
    # errors_all.append(errors)
    # print("****************************")
    # # GradientBoosting
    # errors = []
    # print("GradientBoosting")
    # for i in range(10):
    #     # 切分训练集和验证集
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
    #
    #     clf = GradientBoostingClassifier(n_estimators=2)
    #     y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
    #     overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
    #     errors.append(utils.pos_error(y_test, y_pred))
    # errors_all.append(errors)
    #
    # utils.cdf_figure(errors_all)


if __name__ == '__main__':
    main()
