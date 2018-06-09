import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

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


def velocity(p1, p2):
    """
    计算速率，并且是拆分为经度方向（y轴）、纬度方向（x轴）的分速率
    :param p1:
    :param p2:
    :return:
    """
    time = p2[3] - p1[3]
    x_dis = p2[0] - p1[0]
    y_dis = p2[1] - p1[1]
    x_vel = float(x_dis) / float(time)
    y_vel = float(y_dis) / float(time)

    if p2[0] < p1[0]:
        x_vel *= -1
    if p2[1] < p1[1]:
        y_vel *= -1

    return x_vel, y_vel


def fix_error(y_train, y_test, y_pred):
    """
    按照时间戳排序后，根据预测点的前后两点的速度对预测点进行修正，这是b问和a问中相差的部分，也是最关键的部分
    :param y_train:
    :param y_test:
    :param y_pred:
    :return:
    """
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

    datas_cor = DataFrame(columns=['Longitude', 'Latitude', 'IMSI', 'MRTime', 'id'])
    for IMSI in set(datas.index.tolist()):
        # print(datas.loc[IMSI])
        MS_datas = datas.loc[IMSI].as_matrix()
        # x = MS_datas[:, 0]
        # y = MS_datas[:, 1]
        # plt.scatter(x, y)
        # plt.show()

        for r in range(len(MS_datas)-1):
            if MS_datas[r][4] != -1:
                before = MS_datas[r-1]
                later = MS_datas[r+1]
                pred = MS_datas[r]

                x_vel_true, y_vel_true = velocity(before, later)
                # print(x_vel_true)
                # print(y_vel_true)
                x_vel_pred, y_vel_pred = velocity(before, pred)

                if y_vel_true == 0.0 or x_vel_true == 0.0:
                    continue

                x_vel_cor = x_vel_pred
                y_vel_cor = y_vel_pred

                # 用来表示预测的x轴方向速率改变值和y轴方向速率改变值是否大于真实速率的20倍， 0 表示没有，1表示有
                x_flag = y_flag = 0
                if abs(x_vel_pred - x_vel_true) > abs(x_vel_true)*20:
                    x_flag = 1
                if abs(y_vel_pred - y_vel_true) > abs(y_vel_true)*20:
                    y_flag = 1
                # 如果都大于了10倍，那么就都乘0.1
                if x_flag == 1 and y_flag == 1:
                    x_vel_cor *= 0.1
                    y_vel_cor *= 0.1
                    continue
                # 如果不是都大于10倍，那么就用小于10倍的那个速率改变值与真实的速率的比值进行修正，详细见文档描述
                else:
                    # 用 _flag 来控制是否修正
                    x_vel_cor = (abs(y_vel_true - y_vel_pred) / abs(y_vel_true) * abs(x_vel_true) + abs(x_vel_true)) * x_flag
                    y_vel_cor = (abs(x_vel_true - x_vel_pred) / abs(x_vel_true) * abs(y_vel_true) + abs(y_vel_true)) * y_flag

                # 用来定义方向
                x_dir = y_dir = 1
                if before[0] > pred[0]:
                    x_dir = -1
                if before[1] > pred[1]:
                    y_dir = -1
                # if x_vel_cor != 0.0:
                #     print("x")
                #     print(x_vel_true)
                #     print(x_vel_pred)
                #     print(x_vel_cor)
                #     print(MS_datas[r][0])
                #     MS_datas[r][0] = (pred[3] - before[3]) * x_vel_cor * x_dir + before[0]
                #     print(MS_datas[r][0])
                #
                # if y_vel_cor != 0.0:
                #     print("y")
                #     print(y_vel_true)
                #     print(y_vel_pred)
                #     print(y_vel_cor)
                #     print(MS_datas[r][1])
                #     MS_datas[r][1] = (pred[3] - before[3]) * y_vel_cor * y_dir + before[1]
                #     print(MS_datas[r][1])
                MS_datas[r][0] = (pred[3] - before[3]) * x_vel_cor * x_dir + before[0]
                MS_datas[r][1] = (pred[3] - before[3]) * y_vel_cor * y_dir + before[1]
                # print("=========")

        MS_datas = DataFrame(MS_datas, columns=['Longitude', 'Latitude', 'IMSI', 'MRTime', 'id'])
        datas_cor = pd.concat([datas_cor, MS_datas], axis=0)
    datas_cor = datas_cor[datas_cor['id'] > -1]
    datas_cor.set_index(['id'], inplace=True, drop=False)
    datas_cor.sort_index(inplace=True)

    ll_pred_cor = datas_cor[['Longitude', 'Latitude']].as_matrix().tolist()
    ll_true = np.delete(y_test, [0, 3, 4], axis=1).tolist()

    # for i in range(len(ll_pred_cor)):
    #
    #     print(ll_pred_cor[i])
    #     print(ll_pred[['Longitude', 'Latitude']].as_matrix().tolist()[i])
    #     print(ll_true[i])
    #     print("==========")
    errors = []
    for (true, pred) in zip(ll_true, ll_pred_cor):
        error = utils.haversine(true[0], true[1], pred[0], pred[1])
        errors.append(error)
    errors.sort()
    return errors


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
    errors_all_cor = []
    # 高斯朴素贝叶斯分类器
    errors = []
    errors_cor = []
    print("Gaussian")

    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[0])

        gnb = GaussianNB()
        y_pred = gnb.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))

        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)

        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # K近邻分类器
    errors = []
    errors_cor = []
    print("KNeighbors")

    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        neigh = KNeighborsClassifier()
        y_pred = neigh.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        # print(classification_report(y_test[:, 0], y_pred))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # 决策树分类器
    errors = []
    errors_cor = []
    print("DecisionTree")

    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = DecisionTreeClassifier()
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # 随机森林
    errors = []
    errors_cor = []
    print("RandomForest")

    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = RandomForestClassifier(max_depth=20, random_state=0)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # AdaBoost
    errors = []
    errors_cor = []
    print("AdaBoost")

    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20), learning_rate=0.01, n_estimators=30,
                                 algorithm='SAMME.R')
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # Bagging
    errors = []
    errors_cor = []
    print("Bagging")
    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = BaggingClassifier(n_estimators=20)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    # GradientBoosting
    errors = []
    errors_cor = []
    print("GradientBoosting")
    for i in range(10):
        start = datetime.datetime.now()
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.01)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))
        errors_cor.append(fix_error(y_train, y_test, y_pred))
        # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
        errors.append(utils.pos_error(y_test, y_pred))
        print("Finish range {}".format(i))
        print("Time: {}".format(datetime.datetime.now() - start))

    print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
    print("Median error after correct: {}".format(np.percentile(np.array(errors_cor).mean(axis=0), 50)))
    errors_all.append(errors)
    errors_all_cor.append(errors_cor)
    print("****************************")

    utils.cdf_figure(errors_all, errors_all_cor)


if __name__ == '__main__':
    main()
