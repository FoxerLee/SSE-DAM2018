import pandas as pd
import numpy as np
import math
from datetime import datetime

from fp_growth import find_frequent_itemsets
import pyfpgrowth


def merge_data(item_no):
    # datas_old = pd.read_csv('../trade.csv', usecols=['sldat', 'vipno', item_no])
    datas = pd.read_csv('../trade_new.csv', usecols=['sldatime', 'uid', 'vipno', item_no])

    datas.rename(columns={'sldatime': 'sldat'}, inplace=True)

    # 去除空值（这里主要是针对bndno）
    datas = datas[(True^datas[item_no].isin([float('nan')]))]
    datas[[item_no]] = datas[[item_no]].astype('int')
    # print(datas)
    # print(datas_new)
    # print(datas_old)

    # datas = pd.concat([datas_old, datas_new], axis=0)
    # print(datas.loc['1591015091286'])
    # print(datas)

    datas.set_index(['vipno', 'sldat'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    # 在完成排序后重新设置为一级索引，但不排序，这样能够加快性能
    datas.set_index(['vipno'], inplace=True, drop=False)

    indexs = set(datas.index)
    # print(datas)
    res = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    # print(res)
    indexs = set(datas.index)
    # print(indexs)
    for index in indexs:

        miao = datas.loc[index]
        miao = miao[:int(len(miao)*0.6)]

        if type(miao) == pd.core.series.Series:
            continue
        res = pd.concat([res, miao], axis=0)
    res = res[1:][['uid', 'vipno', item_no]].as_matrix().tolist()

    print(res)

    uids = list(set([r[0] for r in res]))
    merges = dict.fromkeys(uids, [])
    for row in res:
        merges[row[0]] = list(set([int(row[2])] + merges[row[0]]))
    merges_list = list(merges.items())

    # print(merges_list)
    resfile = open("ai_"+item_no+"_out.txt", "w")

    res = []
    for l in merges_list:
        res.append(l[1])
        for m in l[1]:
            resfile.write(str(m) + " ")
        resfile.write("\n")

    resfile.close()
    print(res)
    return res


def enaeseth_fpgrowth(minsup, item_no):
    start = datetime.now()
    transactions = merge_data(item_no)
    for itemset in find_frequent_itemsets(transactions, minsup):
        print(itemset)
    print(datetime.now() - start)


def evandempsey_fpgrowth(minsup, item_no):
    start = datetime.now()
    transactions = merge_data(item_no)
    patterns = pyfpgrowth.find_frequent_patterns(transactions, minsup)
    itemsets = patterns.items()
    for itemset in itemsets:
        print(itemset)
    print(datetime.now() - start)


if __name__ == '__main__':
    # merge_data('bndno')
    # enaeseth_fpgrowth(2, "pluno")
    evandempsey_fpgrowth(2, "pluno")


