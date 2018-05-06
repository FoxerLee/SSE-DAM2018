import pandas as pd
import numpy as np
import math
from datetime import datetime


def combine(datas, item_no):
    # 这里与aii不同的在于，需要考虑时间序列，所以不能舍弃'sldat'字段
    # res_list = res[1:][['uid', 'vipno', item_no]].as_matrix().tolist()
    # res.set_index(['uid'], inplace=True, drop=False)
    vipnos = list(set(datas.index[1:]))
    # 这里使用两层字典，外层字典的key为vipno，value为一个字典 -- 其key为sldat，value为相同sldat的item_no集合
    merges = dict.fromkeys(vipnos, {})
    # print(res)
    for vipno in vipnos:
        # 取出某一个vipno的所有记录
        tmp = datas.loc[vipno]
        # print(tmp)
        # 该vipno只有一行数据，单独处理（因为后面不能转array）
        if type(tmp['sldat']) == str:
            # print(tmp[item_no])
            merges[tmp['vipno']] = {tmp['sldat']: tmp[item_no]}
            # print(merges[tmp['uid']])
            # print(merges[tmp['vipno']])
            continue
        # 获取该vipno购买记录的所有sldat，即时间
        times = list(set(tmp['sldat'].as_matrix().tolist()))
        tmp_list = tmp.as_matrix().tolist()
        # print(tmp_list)
        # 这里操作就和之前问类似了
        uid_dict = dict.fromkeys(times, [])
        for row in tmp_list:
            uid_dict[row[1]] = list(set([int(row[0])] + uid_dict[row[1]]))

        merges[vipno] = uid_dict

    # print(merges)
    merges_list = list(merges.items())
    return merges_list


def write_data(path, datas):
    resfile = open(path, "w")
    for merge in datas:

        miao = list(merge[1].values())
        # print(miao)
        for m in miao:
            # 判断是不是只有一个值
            if type(m) != int:
                for n in m:
                    resfile.write(str(n) + " ")
                # 加入间隔符 -1
                resfile.write(str(-1) + " ")
            else:
                # 只有一行的
                resfile.write(str(m) + " " + str(-1) + " ")
        # 换行符 -2
        resfile.write(str(-2) + "\n")

    resfile.close()


def merge_data(item_no):
    start = datetime.now()
    datas = pd.read_csv('../trade.csv', usecols=['sldat', 'uid', 'vipno', item_no])
    # datas = pd.read_csv('../trade_new.csv', usecols=['sldatime', 'uid', 'vipno', item_no])

    # datas.rename(columns={'sldatime': 'sldat'}, inplace=True)

    # 去除空值（这里主要是针对bndno）
    datas = datas[(True^datas[item_no].isin([float('nan')]))]
    # 将item_no字段的格式转换为int，方便之后处理
    datas[[item_no]] = datas[[item_no]].astype('int')
    # 设置2级索引，进行排序
    datas.set_index(['vipno', 'sldat'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    # 在完成排序后重新设置为一级索引，但不排序，这样能够加快性能
    datas.set_index(['vipno'], inplace=True, drop=False)

    X = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    y = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    indexs = set(datas.index)
    # print(indexs)
    for index in indexs:
        miao = datas.loc[index]
        train = miao[:int(len(miao) * 0.6)]
        test = miao[int(len(miao) * 0.6):]
        if type(miao) == pd.core.series.Series:
            continue
        X = pd.concat([X, train], axis=0)
        y = pd.concat([y, test], axis=0)

    X_list = combine(X, item_no)
    y_list = combine(y, item_no)
    # print(merges_list)

    write_data("input/bii_" + item_no + "_train.txt", X_list)
    write_data("input/bii_" + item_no + "_test.txt", y_list)
    print("For " + item_no + " time: {}".format(datetime.now() - start))


if __name__ == '__main__':
    item_no = ['pluno', 'dptno', 'bndno']
    for i in item_no:
        merge_data(i)