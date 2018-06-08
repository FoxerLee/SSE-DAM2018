import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

import utils # 减少代码重复

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


def main():
    train_data = utils.gongcan_to_ll()
    # 删除原有的ID，不作为训练特征
    for i in range(1, 8):
        train_data.drop(['RNCID_' + str(i)], axis=1, inplace=True)
        train_data.drop(['CellID_' + str(i)], axis=1, inplace=True)

    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)
    rel_lon = []
    rel_lat = []
    for index, row in train_data.iterrows():
        rel_lon.append(row['Longitude'] - row['Longitude_1'])
        rel_lat.append(row['Latitude'] - row['Latitude_1'])

    train_data['rel_Longitude'] = np.array(rel_lon)
    train_data['rel_Latitude'] = np.array(rel_lat)

    # features和labels
    train_data.set_index(['Longitude_1', 'Latitude_1'], inplace=True, drop=False)
    train_data.sort_index(inplace=True)
    ids = list(set(train_data.index.tolist()))

    errors_all = []
    amount = []
    for id in ids:
        MS_datas = train_data.loc[id]
        X = MS_datas.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                           'Num_connected'], axis=1, inplace=False).as_matrix()
        y = MS_datas[['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()

        # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
        random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        # 随机森林
        print("MS {}".format(id))
        errors = []
        for i in range(10):

            # 切分训练集和验证集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

            regr = RandomForestRegressor(max_depth=20, random_state=0)
            y_pred = regr.fit(X_train, np.delete(y_train, [2, 3, 4, 5], axis=1)).predict(X_test)

            error = utils.pos_error(y_test, y_pred)
            errors.append(error)

            # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
            # errors.append(utils.pos_error(y_test, y_pred))

        # 将每个数据集的点做出来
        # plt.title("Median error: %.3f" %np.percentile(np.array(errors).mean(axis=0), 50) +
        #           " Data amount: {}".format(X.shape[0]))
        # ax = plt.gca()
        # ax.get_xaxis().get_major_formatter().set_useOffset(False)
        # plt.scatter(y[:,2], y[:,3])
        # plt.xlim([lb_Longitude, rt_Longitude])
        # plt.ylim([lb_Latitude, rt_Latitude])
        # plt.show()

        # print("Different data amount: {}".format(len(set(X[:,0]))))
        print("Data amount: {}".format(X.shape[0]))
        print("Median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))
        errors_all.append([id, errors])
        amount.append([X.shape[0], np.percentile(np.array(errors).mean(axis=0), 50)])
        # amount.append([len(set(X[:, 0])), np.percentile(np.array(errors).mean(axis=0), 50)])

        print("****************************")
    utils.cdf_figure(errors_all)
    utils.mean_figure(errors_all)
    # utils.cdf_figure_overall(errors_all)

    # 将每个基站的中位误差和总的数据集个数输出
    amount = np.array(amount)
    amount = amount[amount[:, 0].argsort()]
    for a in amount:
        print(a)

    return errors_all


def compare():
    """
    将a问与c问结果比较
    :return:
    """
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
    X_ = train_data.drop(['MRTime', 'Longitude', 'Latitude',
                         'Num_connected', 'grid_num'], axis=1, inplace=False).as_matrix()
    y_ = train_data[['grid_num', 'Longitude', 'Latitude']].as_matrix()
    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    start = datetime.datetime.now()
    errors_all = []

    for i in range(10):

        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=random_states[i])

        clf = RandomForestClassifier(max_depth=20, random_state=0)
        y_pred = clf.fit(np.delete(X_train, 0, axis=1), y_train[:, 0]).predict(np.delete(X_test, 0, axis=1))

        ll_pred = []
        for y in y_pred:
            X_box = int(y % X_box_num)
            y_box = int(y / X_box_num) + 1
            if X_box == 0:
                X_box = X_box_num
                y_box -= 1
            lon = lb_Longitude + per_lon * X_box - 0.5 * per_lon
            lat = lb_Latitude + per_lat * y_box - 0.5 * per_lat

            ll_pred.append([lon, lat])
        ll_true = np.delete(y_test, 0, axis=1).tolist()
        errors = []
        for (true, pred) in zip(ll_true, ll_pred):
            error = utils.haversine(true[0], true[1], pred[0], pred[1])
            errors.append(error)
        errors.sort()
        errors_all.append(errors)

    print("RandomForest")
    print("Median error: {}".format(np.percentile(np.array(errors_all).mean(axis=0), 50)))
    print("Time: {}".format(datetime.datetime.now() - start))
    print("****************************")

    # 获得 c 问结果
    start = datetime.datetime.now()
    c_errors = main()
    print("Time: {}".format(datetime.datetime.now() - start))

    plt.figure('Comparision 2G DATA')
    plt.xlabel('Comparision 2G DATA - CDF figure')
    plt.ylabel('Error(meters)')

    # 绘制 c 问的结果的总体CDF曲线
    mean_errors = []
    for i in range(len(c_errors)):
        errors = np.array(c_errors[i][1])
        mean_error = errors.mean(axis=0)
        mean_errors.extend(mean_error)
    mean_errors.sort()
    plt.plot([float(i) / float(len(mean_errors)) for i in range(len(mean_errors))],
             list(mean_errors), '--', linewidth=1, alpha=0.6,
             label="c-method median error(m): %.3f" % np.percentile(mean_errors, 50))

    # 绘制 a 问的结果的总体CDF曲线
    errors = np.array(errors_all)
    mean_errors = errors.mean(axis=0)
    # print(mean_errors)
    plt.plot([float(i) / float(len(mean_errors)) for i in range(len(mean_errors))],
             list(mean_errors), '--', linewidth=1,
             alpha=0.6, label="a-method median error: %.3f" % np.percentile(mean_errors, 50))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    compare()
