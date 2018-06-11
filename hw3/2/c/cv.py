import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

from pandas import DataFrame
from scipy import interp
from sklearn.ensemble import AdaBoostRegressor


def feature_name_generator(months, overall, predict=False):
    type1 = 'U'
    type2 = 'I'
    # months = ['02', '03', '04']
    aggrs = ['mean', 'std', 'max', 'median']
    # overall = '234'
    feature_names = []
    # TYPE.1 count/ratio - count
    for m in months:
        feature_names.append(type1+'_'+type2+'_'+'month_count_'+m)
    feature_names.append(type1+'_'+type2+'_'+'overall_count_'+overall)

    for m in months:
        feature_names.append(type1+'_'+type2+'_'+'month_qty_'+m)
    feature_names.append(type1+'_'+type2+'_'+'overall_qty_'+overall)

    # TYPE.1 count/ratio - penetration
    for m in months:
        feature_names.append(type2+'_'+type1+'_'+'month_penetration_'+m)
    feature_names.append(type2 + '_' + type1 + '_' + 'overall_penetration_' + overall)

    # TYPE.1 count/ratio - product diversity
    for m in months:
        feature_names.append(type1 + '_' + type2 + '_' + 'month_diversity_' + m)
    feature_names.append(type1 + '_' + type2 + '_' + 'overall_diversity_' + overall)

    # TYPE.2 AGG feature - brand/category/item AGG
    for a in aggrs:
        feature_names.append(type1+'_'+type2+'_'+a+'_'+'count_AGG_'+overall)
    for a in aggrs:
        feature_names.append(type1+'_'+type2+'_'+a+'_'+'qty_AGG_'+overall)

    # TYPE.2 AGG feature - user AGG
    for a in aggrs:
        feature_names.append(type2+'_'+type1+'_'+a+'_'+'count_AGG_'+overall)

    # TYPE.2 AGG feature - month AGG & TYPE.4 complex feature - trend
    for a in aggrs:
        feature_names.append(type1 + '_' + type2 + '_month_count_' + a)
    # feature_names.append(type1 + '_' + type2 + '_month_count_trend')
    for a in aggrs:
        feature_names.append(type2 + '_' + type1 + '_month_penetration_' + a)
    # feature_names.append(type2 + '_' + type1 + '_month_penetration_trend')
    for a in aggrs:
        feature_names.append(type1 + '_' + type2 + '_month_diversity_' + a)
    # feature_names.append(type1 + '_' + type2 + '_month_diversity_trend')

    if not predict:
        feature_names.append('label')

    return feature_names


def train_generator(months, overall, label_month, predict=False):
    datas = pd.read_csv("../references.csv", dtype='object')
    datas = datas.fillna(0)
    indexs = datas['U_I_overall_qty_'+overall].as_matrix().tolist()

    vps = []
    for i in indexs:
        if i == 0:
            continue
        tmp = i.split('-')
        # tmp[0]是vipno，tmp[1]是pluno
        vps.append([tmp[0], tmp[1]])
    vps = np.array(vps)
    feature_names = feature_name_generator(months, overall, predict)
    train_datas = DataFrame(np.zeros(shape=(len(vps), len(feature_names))), columns=feature_names, dtype='float')
    # tmp = DataFrame(vps, columns=['vipno', 'pluno'], dtype='object')
    # print(tmp)
    train_datas = pd.concat([train_datas, DataFrame(vps, columns=['vipno', 'pluno'])], axis=1)

    # 不同的阵容，存储的格式不一样，所以分开处理
    start = datetime.datetime.now()
    train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=False)
    for f in feature_names[:8]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[(tmp[0], tmp[1]), f] = float(tmp[2])
    print(datetime.datetime.now() - start)
    print("***************")
    start = datetime.datetime.now()
    train_datas.set_index(['pluno'], inplace=True, drop=False)
    for f in feature_names[8:12]:
        # count = 0
        ds = datas[f].as_matrix().tolist()
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")
    start = datetime.datetime.now()
    for f in feature_names[24:28]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")
    train_datas.set_index(['vipno'], inplace=True, drop=False)
    # print(train_datas.index)

    start = datetime.datetime.now()
    for f in feature_names[12:24]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            # print(tmp[1])
            train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")

    # months = ['02', '03', '04']
    start = datetime.datetime.now()
    train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=False)
    for index, row in train_datas.iterrows():
        tmp = []
        for m in months:
            tmp.append(row['U_I_month_count_'+m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[28]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[29]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[30]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[31]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['I_U_month_penetration_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[32]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[33]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[34]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[35]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['U_I_month_diversity_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[36]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[37]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[38]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[39]] = tmp[1]

    print(datetime.datetime.now() - start)
    print("***************")

    if not predict:
        start = datetime.datetime.now()
        labels = datas['U_I_month_qty_'+label_month].as_matrix().tolist()
        indexs = train_datas.index
        for label in labels:
            # 0代表空值
            if label != 0:
                label = label.split('-')
                if (label[0], label[1]) in indexs:
                    train_datas.loc[(label[0], label[1]), 'label'] = float(label[2])
        print(datetime.datetime.now() - start)
        print("***************")

    return train_datas


def price():
    datas = pd.read_csv("../trade_new.csv")
    datas.drop(columns=['Unnamed: 0'], inplace=True)

    plunos = set(datas['pluno'].tolist())
    di = dict.fromkeys(plunos, 0.0)

    for row in datas.iterrows():
        per = float(row[1]['amt']) / float(row[1]['qty'])
        di[row[1]['pluno']] = per

    return di


def unit():
    datas = pd.read_csv("../trade_new.csv")
    datas.drop(columns=['Unnamed: 0'], inplace=True)

    plunos = set(datas['pluno'].tolist())
    di = dict.fromkeys(plunos, 0)

    for row in datas.iterrows():
        if row[1]['pkunit'] != '千克':
            # 1 代表这个商品不是千克作为单价，数量必须是整数
            di[row[1]['pluno']] = 1

    return di


def prob():
    res = []
    f = open('predict/1552674_2b_RandomForestClassifier.txt', 'r')
    line = f.readline()
    while line:
        line = line[:-1].split(',')
        if line[2] == 'Yes':
            res.append([line[0], line[1]])
        line = f.readline()
    return res


def money():
    tmp = []
    f = open('predict/1552674_2civ_AdaBoostRegressor.txt', 'r')
    line = f.readline()
    while line:
        line = line[:-1].split(',')

        tmp.append([line[0], line[1]])
        line = f.readline()
    users = np.array(tmp)[:, 0]

    res = dict.fromkeys(users, 0.0)
    for t in tmp:
        print(t)
        res[t[0]] = t[1]
        print(res[t[0]])

    return res


def main():
    print("Generator train data!")
    months = ['02', '03', '04']
    overall = '234'
    label_month = '05'
    train_datas = train_generator(months, overall, label_month)
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X_train = np.delete(train, train.shape[1] - 1, axis=1)
    y_train = train[:, train.shape[1] - 1]

    print("Generator test data!")
    months = ['05', '06', '07']
    overall = '567'
    label_month = '08'
    test_datas = train_generator(months, overall, label_month, predict=True)
    label_ = test_datas.index.tolist()
    X_ = test_datas.drop(columns=['vipno', 'pluno'], inplace=False).as_matrix()

    # 每件商品的单价
    prices = price()
    # 每件商品的单位，有一些商品根据单位可以判断只可能为整数的购买数量
    units = unit()

    # 使用效果最好的 Adaboost
    reg = AdaBoostRegressor(n_estimators=90, learning_rate=0.02)
    y_pred = reg.fit(X_train, y_train).predict(X_)

    pred = dict.fromkeys(label_, 0.0)
    for (i, j) in zip(label_, y_pred):

        if units[int(i[1])] == 1:
            pred[i] = int(j) + 1
        else:
            pred[i] = j

    # 加入前几个月的数量，预测结果综合考虑
    months = ['05', '06', '07']
    datas = pd.read_csv("../references.csv", dtype='object')
    datas = datas.fillna(0)

    for m in months:
        ds = datas['U_I_month_qty_' + m].as_matrix().tolist()
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            if (tmp[0], tmp[1]) in pred.keys():
                pred[(tmp[0], tmp[1])] += float(tmp[2])

    users = np.array(list(pred.keys()))[:, 0]
    pred_U = dict.fromkeys(users, [])

    for key, value in pred.items():
        pred_U[key[0]] = list(pred_U[key[0]] + [key[1], int(value) / 4])

    p = prob()
    moneys = money()

    res = dict.fromkeys(users, [])
    for key, value in pred_U.items():
        true_money = float(moneys[key])
        pred_money = 0.0
        for v in value:
            # 利用预测得到的金额进行修正
            if v[1] != 0 and [key, v[0]] in p:
                pred_money += float(v[1]) * float(prices[int(v[0])])
                res[key] = list(res[key] + [v[0], v[1]])
            if true_money < pred_money:
                break

    f = open('predict/1552674_2cv.txt', 'w')
    for key, value in res.items():
        f.write(str(key) + "::")
        miao = ""
        while i < len(value) - 1:
            print(value[i])
            tmp = int(value[i + 1])
            if units[int(value[i])] == 1:
                tmp = int(tmp)
            if tmp != 0:
                miao += str(value[i]) + ":" + str(value[i + 1]) + ","
            i += 2
        if miao != "":
            f.write(miao[:-1] + '\n')
        else:
            f.write("\n")
    f.close()


if __name__ == '__main__':
    main()