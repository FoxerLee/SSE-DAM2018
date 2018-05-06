import pandas as pd
import numpy as np
import math
from datetime import datetime

from fp_growth import find_frequent_itemsets
import pyfpgrowth


def combine(datas):
    # 按照uid分类
    uids = list(set([r[0] for r in datas]))
    # 这里利用字典来做合并，比较方便
    merges = dict.fromkeys(uids, [])
    for row in datas:
        merges[row[0]] = list(set([int(row[2])] + merges[row[0]]))
    merges_list = list(merges.items())

    return merges_list


def write_data(path, datas):
    resfile = open(path, "w")
    res = []
    for l in datas:
        res.append(l[1])
        for m in l[1]:
            resfile.write(str(m) + " ")
        resfile.write("\n")

    resfile.close()
    return res


def merge_data(item_no):
    start = datetime.now()
    datas = pd.read_csv('../trade.csv', usecols=['sldat', 'uid', 'vipno', item_no])
    # datas = pd.read_csv('../trade_new.csv', usecols=['sldatime', 'uid', 'vipno', item_no])
    # 修改一下名字
    # datas.rename(columns={'sldatime': 'sldat'}, inplace=True)

    # 去除空值（这里主要是针对bndno）
    datas = datas[(True^datas[item_no].isin([float('nan')]))]
    # 将item_no字段的格式转换为int，方便之后处理
    datas[[item_no]] = datas[[item_no]].astype('int')
    # 设置2级索引，进行排序
    datas.set_index(['vipno', 'sldat'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    # 在完成排序后重新设置为一级索引，但不排序，这样能够加快性能，同时方便后面取值
    datas.set_index(['vipno'], inplace=True, drop=False)

    X = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    y = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', item_no])
    indexs = set(datas.index)
    # 取出前60%的数据
    for index in indexs:

        miao = datas.loc[index]
        train = miao[:int(len(miao)*0.6)]
        test = miao[int(len(miao)*0.6):]

        if type(miao) == pd.core.series.Series:
            continue
        X = pd.concat([X, train], axis=0)
        y = pd.concat([y, test], axis=0)
    # 因为a问不考虑时序，就不要'sladt'字段了
    X = X[1:][['uid', 'vipno', item_no]].as_matrix().tolist()
    y = y[1:][['uid', 'vipno', item_no]].as_matrix().tolist()

    X_list = combine(X)
    y_list = combine(y)

    X_res = write_data("input/ai_" + item_no + "_train.txt", X_list)
    y_res = write_data("input/ai_" + item_no + "_test.txt", y_list)
    print("For "+ item_no + " time: {}".format(datetime.now() - start))

    return X_res, y_res


def enaeseth_fpgrowth(minsup, item_no):
    start = datetime.now()
    transactions, y_res = merge_data(item_no)
    for itemset in find_frequent_itemsets(transactions, minsup):
        print(itemset)
    print(datetime.now() - start)


def evandempsey_fpgrowth(minsup, item_no):
    start = datetime.now()
    transactions, y_res = merge_data(item_no)
    patterns = pyfpgrowth.find_frequent_patterns(transactions, minsup)
    itemsets = patterns.items()
    for itemset in itemsets:
        print(itemset)
    print(datetime.now() - start)


if __name__ == '__main__':
    item_no = ['pluno', 'dptno', 'bndno']
    for i in item_no:
        merge_data(i)
    # enaeseth_fpgrowth(2, "pluno")
    # evandempsey_fpgrowth(2, "pluno")


