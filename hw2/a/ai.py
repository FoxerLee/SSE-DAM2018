import csv
from fp_growth import find_frequent_itemsets
import pyfpgrowth


def merge_data():

    with open('../trade.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        trades = [row for row in reader]

    with open('../trade_new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        trades_new = [row for row in reader]

    trades_new = trades_new[1:]
    trades = trades[1:]
    # print(datas)

    uids = [row[0] for row in trades] + [row[1] for row in trades_new]

    merges = dict.fromkeys(uids, [])

    for row in trades:
        merges[row[0]] = list(set([row[6]] + merges[row[0]]))

    for row in trades_new:
        merges[row[1]] = list(set([row[8]] + merges[row[1]]))

    merges_list = list(merges.items())
    # print(merges)

    res = []

    for l in merges_list:
        res.append(l[1])

    # with open('merges.csv', 'w') as f:
    #     w = csv.writer(f)
    #
    #     for l in merges_list:
    #         w.writerow(l[1])
    return res


def enaeseth_fpgrowth(minsup):
    transactions = merge_data()
    for itemset in find_frequent_itemsets(transactions, minsup):
        print(itemset)


def evandempsey_fpgrowth(minsup):
    transactions = merge_data()
    patterns = pyfpgrowth.find_frequent_patterns(transactions, minsup)
    itemsets = patterns.items()
    for itemset in itemsets:
        print(itemset)

if __name__ == '__main__':
    # merge_data()
    # enaeseth_fpgrowth(2)
    evandempsey_fpgrowth(2)


