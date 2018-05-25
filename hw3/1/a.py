import pandas as pd
import numpy as np
import math

from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

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

    overall_pre = result.split('\n')[-2].split('      ')[1]
    print(overall_pre)

    res = result.split('\n')
    # res = res[:-1]
    top10_pre = []
    top10_recall = []
    for r in res:
        r = r.lstrip().split('      ')
        # print(type(r[0]))
        # print(type(top10[0]))
        try:
            if float(r[0]) in top10:
                top10_pre.append([r[0], r[1]])
                top10_recall.append([r[0], r[2]])
                # print(r[0])
                # print(r[1])
                # print("=====")
        except:
            continue
    print(top10_pre)
    print(top10_recall)
    print("=========")

    # for i in top10:
    # f = open('result.txt', 'w')
    # f.write(result)
    # f.close()


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
        lon = lb_Longitude + per_lon * X_box
        lat = lb_Latitude + per_lat * y_box

        ll_pred.append([lon, lat])
    ll_true = np.delete(y_true, 0, axis=1).tolist()
    # print(ll_true)
    # print(ll_pred)
    errors = []
    for (true, pred) in zip(ll_true, ll_pred):
        error = haversine(true[0], true[1], pred[0], pred[1])
        errors.append(error)
    return errors.sort()
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


def main():
    ll_data_2g = gongcan_to_ll()
    train_data = ll_to_grid(ll_data_2g)

    # print(train_data)
    # 删除原有的ID，不作为训练特征
    for i in range(1, 8):
        train_data.drop(['RNCID_'+str(i)], axis=1, inplace=True)
        train_data.drop(['CellID_'+str(i)], axis=1, inplace=True)
    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)

    # features和labels
    X = train_data.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                         'Num_connected', 'grid_num'], axis=1, inplace=False).as_matrix()
    y = train_data[['grid_num', 'Longitude', 'Latitude']].as_matrix()

    # 切分训练集和验证集
    # random_state不设置，每次的随机结果都会不一样
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 高斯朴素贝叶斯分类器
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train[:,0]).predict(X_test)
        precision_recall(y_test[:,0], y_pred)
        pos_error(y_test, y_pred)

    # print(y)
    # X.to_csv('X.csv')
    # train_data.to_csv("train_data.csv")


if __name__ == '__main__':
    main()
    # read_data('./data/data_2g.csv')
    # iris = datasets.load_iris()
    # print(iris.data)
    # print(type(iris.data))
