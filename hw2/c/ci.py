import pandas as pd
import numpy as np
import math
import collections
from operator import itemgetter
from datetime import datetime


def read_data(path):
    f = open(path, 'r')
    inputs = f.readlines()
    purchases = []
    for i in inputs:
        i = i.split(' ')
        purchases.append(i)

    return purchases


def feasibility(frequent_itemsets, test_data):
    res = []
    for fre in frequent_itemsets:
        count = 0
        for t in test_data:
            # print(fre[0])
            if set(fre[0]).issubset(set(t)):
                count += 1
        res.append((fre, count, float(count/len(test_data))))
    return res


def main():
    f = open('../miao.txt', 'r')
    datas = f.readlines()
    # print(datas)
    frequent_itemsets = []
    for data in datas:
        i = data.split(' ')
        # print(data)

        frequent_itemsets.append([i[:-2], int(i[-1][:-1])])
    # 选择支持度前10的频繁项集
    # print(frequent_itemsets)
    frequent_itemsets = sorted(frequent_itemsets, key=itemgetter(1), reverse=True)[:10]
    print("Top 10:")
    for f in frequent_itemsets:
        print(f)
    test_data = read_data('../a/input/ai_bndno_test.txt')

    res = feasibility(frequent_itemsets, test_data)
    print("Test result:")
    for r in res:
        print(r)



if __name__ == '__main__':
    main()