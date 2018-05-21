import pandas as pd
import math
from pandas.core.frame import DataFrame

# def read_data(path):
#     datas = pd.read_csv(path)
#     print(datas)
#
#     return datas


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

    except_ID = set()
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

    print(new_data_2g)
    # print(except_ID)
    # print(len(except_ID))
    return new_data_2g


def ll_to_grid(ll_data_2g):
    """

    :param ll_data_2g:
    :return:
    """



def main():
    ll_data_2g = gongcan_to_ll()
    ll_to_grid(ll_data_2g)


if __name__ == '__main__':
    main()
    # read_data('./data/data_2g.csv')