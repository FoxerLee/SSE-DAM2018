import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

from pandas import DataFrame


def figure(pres, recalls, aucs):
    # ax = plt.gca()
    # pres = [0.810, 0.655, 0.655, 0.712, 0.721, 0.717, 0.728]
    # racalls = [0.426, 0.644, 0.633, 0.624, 0.668, 0.672, 0.686]
    # aucs = [0.769, 0.688, 0.648, 0.754, 0.783, 0.775, 0.789]
    plt.xlabel('Classifier')
    # plt.ylabel('Classifier')
    total_width, n = 0.9, 3
    width = total_width / n
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    x = np.arange(len(labels))
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.plot(pres, 'x--', label='precision')
    for a, b in zip(x, pres):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(recalls, '.--', label='recall')
    for a, b in zip(x, recalls):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(aucs, '*--', label='auc')
    for a, b in zip(x, aucs):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.ylim([0.0, 1.0])

    plt.legend()
    plt.show()


def least_squares(x, y):
    """
    最小二乘法计算出拟合直线点斜率
    :param x:
    :param y:
    :return:
    """
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in range(len(x)):
        k = (x[i]-x_) * (y[i]-y_)
        m += k
        p = np.square(x[i]-x_)
        n = n + p
    a = m/n
    b = y_ - a*x_
    return a


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

    # feature_names.append(type1 + '_' + type2 + '_repeat_' + overall)
    # feature_names.append(type2 + '_' + type1 + '_repeat_' + overall)
    if not predict:
        feature_names.append('label')

    return feature_names


def train_generator(months, overall, label_month, predict=False):
    datas = pd.read_csv("../references.csv", dtype='object')
    datas = datas.fillna(0)
    indexs = datas['U_I_overall_count_'+overall].as_matrix().tolist()

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
    for f in feature_names[:4]:
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
    for f in feature_names[4:8]:
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
    for f in feature_names[16:20]:
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
    for f in feature_names[8:16]:
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
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[20]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[21]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[22]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[23]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['I_U_month_penetration_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[24]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[25]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[26]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[27]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['U_I_month_diversity_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[28]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[29]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[30]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['pluno']), feature_names[31]] = tmp[1]

    print(datetime.datetime.now() - start)
    print("***************")

    # start = datetime.datetime.now()
    # train_datas.set_index(['vipno'], inplace=True, drop=False)
    # ds = datas[feature_names[32]].as_matrix().tolist()
    # for row in ds:
    #     if row == 0:
    #         continue
    #     tmp = row.split('-')
    #     train_datas.loc[tmp[0], feature_names[32]] = float(tmp[1])
    #
    # train_datas.set_index(['pluno'], inplace=True, drop=False)
    # ds = datas[feature_names[33]].as_matrix().tolist()
    # for row in ds:
    #     if row == 0:
    #         continue
    #     tmp = row.split('-')
    #     train_datas.loc[tmp[0], feature_names[33]] = float(tmp[1])
    # start = datetime.datetime.now()
    # print(datetime.datetime.now() - start)
    # print("***************")
    # train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=False)

    if not predict:
        start = datetime.datetime.now()
        labels = datas['U_I_month_count_'+label_month].as_matrix().tolist()
        indexs = train_datas.index
        for label in labels:
            # 0代表空值
            if label != 0:
                label = label.split('-')
                if (label[0], label[1]) in indexs:
                    train_datas.loc[(label[0], label[1]), 'label'] = 1
        print(datetime.datetime.now() - start)
        print("***************")
    return train_datas


def draw_roc(fprs, tprs, thresholds, aucs):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    for i in range(len(aucs)):
        plt.plot(fprs[i], tprs[i], lw=lw,
                 label=labels[i] + ' ROC curve (auc = %0.3f)' % aucs[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.show()