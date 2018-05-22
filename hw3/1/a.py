import pandas as pd
import math
from pandas.core.frame import DataFrame
from math import radians, cos, sin, asin, sqrt


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
    # 左下角坐标
    lb_Longitude = 121.20120490000001
    lb_Latitude = 31.28175691
    # 右上角坐标
    rt_Longitude = 121.2183295
    rt_Latitude = 31.29339344

    y_box_num = int((haversine(lb_Longitude, lb_Latitude, lb_Longitude, rt_Latitude))/20) + 1
    X_box_num = int((haversine(lb_Longitude, lb_Latitude, rt_Longitude, lb_Latitude))/20) + 1
    # print(X_box_num/20)
    # print(y_box_num/20)
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

        grid_num = X + y * y_box_num
        row.append(grid_num)

    indexs = ll_data_2g.columns.values.tolist()
    indexs.append('grid_num')
    train_data = DataFrame(ll_data_2g_list)
    train_data.columns = indexs

    print(train_data)

    return train_data


def main():
    ll_data_2g = gongcan_to_ll()
    train_data = ll_to_grid(ll_data_2g)


if __name__ == '__main__':
    # main()
    # read_data('./data/data_2g.csv')
