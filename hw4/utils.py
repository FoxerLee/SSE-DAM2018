import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math


from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics import classification_report

# 左下角坐标
lb_Longitude = 121.20120490000001
lb_Latitude = 31.28175691
# 右上角坐标
rt_Longitude = 121.2183295
rt_Latitude = 31.29339344
# 格子个数是 82*65
y_box_num = 13
X_box_num = 17
# 每一个格子所占的经纬度
per_lon = (rt_Longitude - lb_Longitude)/X_box_num
per_lat = (rt_Latitude - lb_Latitude)/y_box_num

per_len = 100  # 每个格子的长、宽


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


def precision_recall(y_true, y_pred):
    """
    计算precision（对于所有的grid和每一个grid）和recall（对于每一个grid）
    :param y_true:
    :param y_pred:
    :return:
    """
    result = classification_report(y_true, y_pred)
    sort_d = pd.value_counts(y_true)
    top10 = sort_d.iloc[0:10].index.tolist()
    # print(top10)

    overall_pre = float(result.split('\n')[-2].split('      ')[1])
    # print(overall_pre)

    res = result.split('\n')
    # res = res[:-1]
    top10_pre = []
    top10_recall = []
    top10_f = []
    for r in res:

        r = r.lstrip().split('      ')

        try:
            if float(r[0]) in top10:
                top10_pre.append(float(r[1]))
                top10_recall.append(float(r[2]))
                top10_f.append(float(r[3]))
        except:
            continue

    return overall_pre, top10_pre, top10_recall, top10_f


def pos_error(y_true, y_pred):
    """
    计算位置误差
    :param y_true:
    :param y_pred:
    :return:
    """
    ll_pred = []
    for y in y_pred:
        # lon = lb_Longitude + (y % X_box_num) * ()
        # lat = lb_Latitude +
        X_box = int(y % X_box_num)
        y_box = int(y / X_box_num) + 1
        if X_box == 0:
            X_box = X_box_num
            y_box -= 1
        lon = lb_Longitude + per_lon * X_box - 0.5 *  per_lon
        lat = lb_Latitude + per_lat * y_box - 0.5 * per_lat

        ll_pred.append([lon, lat])
    ll_true = np.delete(y_true, 0, axis=1).tolist()
    # print(ll_true)
    # print(ll_pred)
    errors = []
    for (true, pred) in zip(ll_true, ll_pred):
        error = haversine(true[0], true[1], pred[0], pred[1])
        errors.append(error)
    errors.sort()
    return errors
    # print(errors[int(len(errors)/2)])


def gongcan_to_ll():
    """
    将原始数据中的RNCID和CellID转换为gongcan表里的经纬度
    :return:
    """
    data_2g = pd.read_csv('./data/data_2g.csv')
    gongcan = pd.read_csv('./data/2g_gongcan.csv')

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

    # print(new_data_2g)
    # print(except_ID)
    # print(len(except_ID))
    return new_data_2g


def ll_to_grid(ll_data_2g):
    """
    grid_num 是从1开始编号的
    :param ll_data_2g:
    :return:
    """

    # y_box_num = int((haversine(lb_Longitude, lb_Latitude, lb_Longitude, rt_Latitude))/per_len) + 1
    # X_box_num = int((haversine(lb_Longitude, lb_Latitude, rt_Longitude, lb_Latitude))/per_len) + 1
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

        y = int(y_length / per_len)
        X = int(X_length / per_len)

        if y_length % per_len != 0:
            y += 1
        if X_length % per_len != 0:
            X += 1

        grid_num = X + (y-1) * X_box_num
        row.append(grid_num)

    indexs = ll_data_2g.columns.values.tolist()
    indexs.append('grid_num')
    train_data = DataFrame(ll_data_2g_list)
    train_data.columns = indexs

    # print(train_data)
    return train_data


def cdf_figure(errors_all):
    plt.figure('Comparision 2G DATA')
    # ax = plt.gca()
    plt.xlabel('CDF')
    plt.ylabel('Error(meters)')
    # X_list = []
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    # for i in range(1220):
    #     X_list.append((float(i) / 1220.0))

    for i in range(len(errors_all)):
        errors = np.array(errors_all[i])
        mean_errors = errors.mean(axis=0)
        # print(mean_errors)
        plt.plot([float(i)/float(len(mean_errors)) for i in range(len(mean_errors))],
                 list(mean_errors), '--', linewidth=1,
                 alpha=0.6, label=labels[i] + " median error: %.2f" % np.percentile(mean_errors, 50))

    plt.legend()
    plt.show()


def time_figure(times):
    plt.figure('Comparision 2G DATA')
    # ax = plt.gca()
    # times = [1.16, 2.18, 2.08, 8.14, 45.75, 21.69, 133.1]
    plt.ylabel('Time(s)')
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    # X = [i for i in range(1, 8)]
    # for i in range(7):
    x = np.arange(len(labels)) + 1
    plt.bar(x, times, width=0.35, align='center', color='c', alpha=0.8)
    plt.xticks(x, labels, size='small', rotation=30)
    for a, b in zip(x, times):
        plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    # plt.barh(labels, times, linewidth=2, alpha=0.6)

    # plt.legend()
    plt.show()


def figure(overall_pres_all, top10_pres_all, top10_recalls_all, top10_fs_all):
    # overall_pres_all = [0.108, 0.335, 0.469, 0.479, 0.491, 0.504, 0.422]
    # top10_pres_all = [0.093, 0.664, 0.831, 0.835, 0.848, 0.839, 0.737]
    # top10_recalls_all = [0.055, 0.632, 0.775, 0.803, 0.788, 0.802, 0.661]
    # top10_fs_all = [0.067, 0.618, 0.786, 0.803, 0.806, 0.807, 0.670]
    plt.xlabel('Classifier')
    # plt.ylabel('Classifier')
    total_width, n = 0.9, 3
    width = total_width / n
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    x = np.arange(len(labels))
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.plot(overall_pres_all, 'x--', label='overall-precision')
    for a, b in zip(x, overall_pres_all):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(top10_pres_all, '.--', label='per-precision(top10)')
    for a, b in zip(x, top10_pres_all):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(top10_recalls_all, '*--', label='per-recall(top10)')
    for a, b in zip(x, top10_recalls_all):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(top10_fs_all, '^--', label='per-f_measurement(top10)')
    for a, b in zip(x, top10_fs_all):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.ylim([0.0, 1.0])

    plt.legend()
    plt.show()
