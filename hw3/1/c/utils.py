import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt

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


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    reference：https://blog.csdn.net/vernice/article/details/46581361
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def gongcan_to_ll():
    """
    将原始数据中的RNCID和CellID转换为gongcan表里的经纬度
    :return:
    """
    data_2g = pd.read_csv('../data_2g.csv')
    gongcan = pd.read_csv('../2g_gongcan.csv')

    # 将RNCID和CellID合并为一个字段，用"-"隔开，之后好用来比较
    gongcan["IDs"] = gongcan[['RNCID', 'CellID']].apply(lambda x: '-'.join(str(value) for value in x), axis=1)
    # print(gongcan)
    # gongcan["LLs"] = gongcan[['Latitude', 'Longitude']].apply(lambda x: '-'.join(str(value) for value in x), axis=1)

    gongcan = gongcan[["IDs", "Latitude", "Longitude"]]
    gongcan = gongcan.as_matrix().tolist()

    IDs = list(set([g[0] for g in gongcan]))
    # print(type(IDs[0]))
    gongcan_dict = dict.fromkeys(IDs, [])

    for g in gongcan:
        gongcan_dict[g[0]] = [g[1], g[2]]
    # 表示未知的、空的，反正就不是正常的数据，都用0，-1表示
    gongcan_dict.update({"0--1":[0, -1]})

    data_2g_list = data_2g.as_matrix().tolist()
    # 在data_2g中，RNCID_i和CellID_i（i=1,2,...,7）对应的下标，打一个表方便处理
    IDs_index = [[5, 6], [10, 11], [15, 16], [20, 21], [25, 26], [30, 31], [35, 36]]

    # except_ID = set()
    # data_lls = []
    for row in data_2g_list:
        # ll = [row[0]]
        # print(row)
        for i in IDs_index:
            # 将空值附为 0 和 -1
            if math.isnan(row[i[0]]):
                row[i[0]] = 0
                row[i[1]] = -1
            # print(int(row[i[0]]))
            # print(row[i[1]])
            ID = str(int(row[i[0]])) + '-' + str(int(row[i[1]]))
            # print(ID)

            if ID in gongcan_dict.keys():
                row.append(gongcan_dict[ID][0])
                row.append(gongcan_dict[ID][1])
                # print(ll)
            else:
                row.append(0)
                row.append(-1)
                # except_ID.add(ID)
        # data_lls.append(ll)
    indexs = data_2g.columns.values.tolist()
    for i in range(1, 8):
        indexs.append('Latitude_' + str(i))
        indexs.append('Longitude_' + str(i))
        # print("miao")
    # print(indexs)
    # print(data_2g_list[0])
    new_data_2g = DataFrame(data_2g_list)
    new_data_2g.columns = indexs

    return new_data_2g


def ll_to_grid(ll_data_2g):
    """
    grid_num 是从1开始编号的
    :param ll_data_2g:
    :return:
    """

    # y_box_num = int((haversine(lb_Longitude, lb_Latitude, lb_Longitude, rt_Latitude))/20) + 1
    # X_box_num = int((haversine(lb_Longitude, lb_Latitude, rt_Longitude, lb_Latitude))/20) + 1
    # print(X_box_num)
    # print(y_box_num)
    # print(ll_data_2g)
    ll_data_2g_list = ll_data_2g.as_matrix().tolist()
    for row in ll_data_2g_list:
        lon = row[2]
        lat = row[3]
        # grid_index = calculate_grid(lb_Latitude, lb_Longitude, lat, lon)
        y_length = haversine(lb_Longitude, lb_Latitude, lb_Longitude, lat)
        X_length = haversine(lb_Longitude, lb_Latitude, lon, lb_Latitude)

        y = int(y_length / 20)
        X = int(X_length / 20)
        if y_length % 20 != 0:
            y += 1
        if X_length % 20 != 0:
            X += 1

        grid_num = X + (y-1) * X_box_num
        row.append(grid_num)

    indexs = ll_data_2g.columns.values.tolist()
    indexs.append('grid_num')
    train_data = DataFrame(ll_data_2g_list)
    train_data.columns = indexs

    # print(train_data)
    return train_data


def pos_error(y_true, y_pred):
    ll_pred = []
    for (true, pred) in zip(y_true, y_pred):
        lon = pred[0] + true[4]
        lat = pred[1] + true[5]
        ll_pred.append([lon, lat])
    ll_true = np.delete(y_true, [0, 1, 4, 5], axis=1)
    errors = []
    for (true, pred) in zip(ll_true, ll_pred):
        error = haversine(true[0], true[1], pred[0], pred[1])
        errors.append(error)
    errors.sort()
    return errors


def cdf_figure(errors_all):
    """
    绘制的是对于每一个MR所生成的模型的CDF图，以及将所有MR的模型预测结果合并在一起后的误差图
    :param errors_all:
    :return:
    """
    plt.figure('Comparision 2G DATA')
    # ax = plt.gca()
    plt.xlabel('Comparision 2G DATA - CDF figure')
    plt.ylabel('Error(meters)')

    # # 绘制的是对于每一个MR所生成的模型的CDF（画出来看都看不....）
    # for i in range(len(errors_all)):
    #     errors = np.array(errors_all[i][1])
    #     mean_errors = errors.mean(axis=0)
    #     # print(mean_errors)
    #     plt.plot([float(i)/float(len(mean_errors)) for i in range(len(mean_errors))], list(mean_errors), '--',
    #              linewidth=1, alpha=0.6)

    # 绘制的是将所有MR的模型预测结果合并在一起后的误差
    mean_errors = []
    for i in range(len(errors_all)):
        errors = np.array(errors_all[i][1])
        mean_error = errors.mean(axis=0)
        mean_errors.extend(mean_error)
    mean_errors.sort()
    plt.plot([float(i) / float(len(mean_errors)) for i in range(len(mean_errors))],
             list(mean_errors), '--', linewidth=1, alpha=0.6,
             label="overall median error(m): %.3f" % np.percentile(mean_errors, 50))

    plt.legend()
    plt.show()


def mean_figure(errors_all):
    """
    中位误差值的分布
    :param errors_all:
    :return:
    """
    labels = []
    for e in errors_all:
        labels.append(e[0])
    x = np.arange(len(labels))

    mean_errors = []
    for i in range(len(errors_all)):
        errors = np.array(errors_all[i][1])
        mean_errors.append(np.percentile(errors.mean(axis=0), 50))

    plt.xticks(range(len(labels)), range(len(labels)))
    plt.plot(mean_errors, 'x--', label='median-error')
    for a, b in zip(x, mean_errors):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    plt.legend()
    plt.show()
