import pandas as pd
import numpy as np
import math
from datetime import datetime


def merge_data(item_no):
    # datas_old = pd.read_csv('../trade.csv', usecols=['sldat', 'vipno', item_no])
    datas = pd.read_csv('../trade_new.csv', usecols=['sldatime', 'uid', 'vipno', item_no])

    datas.rename(columns={'sldatime': 'sldat'}, inplace=True)

    # 去除空值（这里主要是针对bndno）
    datas = datas[(True ^ datas[item_no].isin([float('nan')]))]
    # 将item_no字段的格式转换为int，方便之后处理
    datas[[item_no]] = datas[[item_no]].astype('int')
    # 设置2级索引，进行排序
    datas.set_index(['vipno', 'sldat'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    # 在完成排序后重新设置为一级索引，但不排序，这样能够加快性能
    datas.set_index(['vipno'], inplace=True, drop=False)

    res = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    # print(res)
    indexs = set(datas.index)
    # print(indexs)
    for index in indexs:

        miao = datas.loc[index]
        miao = miao[:int(len(miao) * 0.6)]

        if type(miao) == pd.core.series.Series:
            continue
        res = pd.concat([res, miao], axis=0)
    # bi和ai的数据实际上是一样的，因为按照uid分类的话，每一个uid中的购买记录时间都一样的
    res = res[1:][['uid', 'vipno', item_no]].as_matrix().tolist()

    # print(res)
    # 同样的用字典合并
    uids = list(set([r[0] for r in res]))
    merges = dict.fromkeys(uids, [])
    for row in res:
        merges[row[0]] = list(set([int(row[2])] + merges[row[0]]))
    merges_list = list(merges.items())

    # print(merges_list)
    resfile = open("input/bi_" + item_no + ".txt", "w")

    res = []
    for l in merges_list:
        res.append(l[1])
        for m in l[1]:
            resfile.write(str(m) + " ")
        # 这个是SPMF中的算法所要求的输入格式
        resfile.write("-1 -2\n")

    resfile.close()


if __name__ == '__main__':
    merge_data('bndno')