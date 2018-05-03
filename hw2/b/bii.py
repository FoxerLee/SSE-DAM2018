import pandas as pd
import numpy as np
import math
from datetime import datetime


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

    # indexs = set(datas.index)
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
    # res_list = res[1:][['uid', 'vipno', item_no]].as_matrix().tolist()
    # res.set_index(['uid'], inplace=True, drop=False)
    vipnos = list(set(res.index[1:]))
    merges = dict.fromkeys(vipnos, {})
    print(res)
    for vipno in vipnos:
        tmp = res.loc[vipno]

        if type(tmp['sldat']) == str:
            merges[tmp['vipno']] = {tmp['sldat']: tmp[item_no]}
            # print(merges[tmp['uid']])
            continue

        times = list(set(tmp['sldat'].as_matrix().tolist()))
        tmp_list = tmp.as_matrix().tolist()
        uid_dict = dict.fromkeys(times, [])
        for row in tmp_list:
            uid_dict[row[1]] = list(set([int(row[0])] + uid_dict[row[1]]))

        # print(tmp)
        merges[vipno] = uid_dict

    # print(merges)
    merges_list = list(merges.items())
    # print(merges_list)
    resfile = open("bii_" + item_no + "_out.txt", "w")
    for merge in merges_list:
        # print(type(merge))
        # print(merge)
        miao = list(merge[1].values())
        # print(miao)
        for m in miao:
            if type(m) != int:
                for n in m:
                    resfile.write(str(n) + " ")
                resfile.write(str(-1) + " ")
            else:
                resfile.write(str(m) + " " + str(-1) + " ")
        resfile.write(str(-2) + "\n")

    resfile.close()


if __name__ == '__main__':
    merge_data('bndno')