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


def confidence(purchases, candidate_rules):
    correct_counts = collections.defaultdict(int)
    incorrect_counts = collections.defaultdict(int)
    for purchase in purchases:
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            # print(premise)
            if premise.issubset(set(purchase)):
                # print("miao")
                if conclusion in purchase:
                    # print("miao")
                    correct_counts[candidate_rule] += 1
                else:
                    # print("miao")
                    incorrect_counts[candidate_rule] += 1

    rule_confidence = {}
    for candidate_rule in candidate_rules:
        if correct_counts[candidate_rule] + incorrect_counts[candidate_rule] == 0:
            continue
        rule_confidence[candidate_rule] = correct_counts[candidate_rule] / \
                                          float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])

    return rule_confidence


def main():
    f = open('miao.txt', 'r')
    datas = f.readlines()
    # print(datas)
    frequent_itemsets = []
    for data in datas:
        data = data.split(' ')[:-2]
        # print(data)
        # 我们对只有一个元素的频繁项集不再感兴趣，它们对生成关联规则没有用处，生成关联规则至少需要两个项目
        if len(data) == 1:
            continue
        frequent_itemsets.append(data)

    candidate_rules = []
    # print(frequent_itemsets)
    for itemset in frequent_itemsets:
        for conclusion in itemset:
            # print(itemset)
            premise = frozenset(set(itemset) - set((conclusion,)))

            candidate_rules.append((premise, conclusion))

    # print(candidate_rules)

    purchases = read_data('../a/input/aii_pluno.txt')

    # print(purchases)
    # print(candidate_rules)
    rule_confidence = confidence(purchases, candidate_rules)
    print(rule_confidence)
    sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)[:5]
    # print(sorted_confidence)

    rule_confidence = [s[0] for s in sorted_confidence]
    test_data = read_data('../a/input/aii_pluno_test.txt')

    res = confidence(test_data, rule_confidence)
    print(res)


if __name__ == '__main__':
    main()
