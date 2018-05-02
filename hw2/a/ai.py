import pandas as pd
import numpy as np

from datetime import datetime

from fp_growth import find_frequent_itemsets
import pyfpgrowth



def merge_data(item_no):
    # datas_old = pd.read_csv('../trade.csv', usecols=['sldat', 'vipno', item_no])
    datas = pd.read_csv('../trade_new.csv', usecols=['sldatime', 'uid', 'vipno', item_no])

    datas.rename(columns={'sldatime': 'sldat'}, inplace=True)

    # print(datas_new)
    # print(datas_old)

    # datas = pd.concat([datas_old, datas_new], axis=0)
    # print(datas.loc['1591015091286'])
    # print(datas)

    datas.set_index(['vipno', 'sldat'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    datas.set_index(['vipno'], inplace=True, drop=False)

    indexs = set(datas.index)
    # print(datas)
    res = pd.DataFrame(index=['vipno'], columns=['sldat', 'uid', 'vipno', 'pluno'])
    # print(res)
    indexs = set(datas.index)
    # print(indexs)
    for index in indexs:

        miao = datas.loc[index]
        miao = miao[:int(len(miao)*0.6)]

        if type(miao) == pd.core.series.Series:
            continue

        res = pd.concat([res, miao], axis=0)
    res = res[1:][['uid', 'vipno', 'pluno']].as_matrix().tolist()

    # print(type(res))
    # print(res)

    uids = list(set([r[0] for r in res]))
    merges = dict.fromkeys(uids, [])
    for row in res:
        merges[row[0]] = list(set([row[2]] + merges[row[0]]))
    merges_list = list(merges.items())

    print(merges_list)
    res = []
    for l in merges_list:
        res.append(l[1])

    print(res)
    return res

# def merge_data():
#
#     with open('../trade.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         trades = [row for row in reader]
#
#     with open('../trade_new.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         trades_new = [row for row in reader]
#
#     trades_new = trades_new[1:]
#     trades = trades[1:]
#     # print(datas)
#
#     uids = [row[0] for row in trades] + [row[1] for row in trades_new]
#
#     merges = dict.fromkeys(uids, [])
#
#     for row in trades:
#         merges[row[0]] = list(set([row[6]] + merges[row[0]]))
#
#     for row in trades_new:
#         merges[row[1]] = list(set([row[8]] + merges[row[1]]))
#
#     merges_list = list(merges.items())
#     # print(merges)
#
#     res = []
#
#     for l in merges_list:
#         res.append(l[1])
#
#     # with open('merges.csv', 'w') as f:
#     #     w = csv.writer(f)
#     #
#     #     for l in merges_list:
#     #         w.writerow(l[1])
#     return res


def enaeseth_fpgrowth(minsup):
    start = datetime.now()
    transactions = merge_data()
    for itemset in find_frequent_itemsets(transactions, minsup):
        print(itemset)
    print(datetime.now() - start)


def evandempsey_fpgrowth(minsup):
    start = datetime.now()
    transactions = merge_data()
    patterns = pyfpgrowth.find_frequent_patterns(transactions, minsup)
    itemsets = patterns.items()
    for itemset in itemsets:
        print(itemset)
    print(datetime.now() - start)


if __name__ == '__main__':
    merge_data('pluno')
    # enaeseth_fpgrowth(32)
    # evandempsey_fpgrowth(32)


