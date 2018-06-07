import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from pandas import DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


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


def feature_name_generator(months, overall):
    type1 = 'U'
    type2s = ['I', 'B', 'C']
    # months = ['02', '03', '04']
    aggrs = ['mean', 'std', 'max', 'median']
    # overall = '234'
    feature_names = []
    # TYPE.1 count/ratio - count
    for month in months:
        feature_names.append('U_month_count_'+month)
    feature_names.append('U_overall_count_'+overall)

    for month in months:
        feature_names.append('U_month_money_'+month)
    feature_names.append('U_overall_money_'+overall)

    # for type2 in type2s:
    #     # for m in months:
    #     #     feature_names.append(type1+'_'+type2+'_'+'month_count_'+m)
    #     # feature_names.append(type1+'_'+type2+'_'+'overall_count_'+overall)
    #     # # TYPE.1 count/ratio - penetration
    #     # for m in months:
    #     #     feature_names.append(type2+'_'+type1+'_'+'month_penetration_'+m)
    #     # feature_names.append(type2 + '_' + type1 + '_' + 'overall_penetration_' + overall)
    #
    #     # TYPE.1 count/ratio - product diversity
    #     for m in months:
    #         feature_names.append(type1 + '_' + type2 + '_' + 'month_diversity_' + m)
    #     feature_names.append(type1 + '_' + type2 + '_' + 'overall_diversity_' + overall)
    #     # TYPE.2 AGG feature - brand/category/item AGG
    #     for a in aggrs:
    #         feature_names.append(type1+'_'+type2+'_'+a+'_'+'count_AGG_'+overall)
    #     # # TYPE.2 AGG feature - user AGG
    #     # for a in aggrs:
    #     #     feature_names.append(type2+'_'+type1+'_'+a+'_'+'count_AGG_'+overall)
    #
    #     # TYPE.2 AGG feature - month AGG
    #     # for a in aggrs:
    #     #     feature_names.append(type1 + '_' + type2 + '_month_count_' + a)
    #     # for a in aggrs:
    #     #     feature_names.append(type2 + '_' + type1 + '_month_penetration_' + a)
    #     for a in aggrs:
    #         feature_names.append(type1 + '_' + type2 + '_month_diversity_' + a)
    for a in aggrs:
        feature_names.append(type1+'_month_count_'+a)
    feature_names.append(type1 + '_month_count_trend')
    for a in aggrs:
        feature_names.append(type1+'_month_money_'+a)
    feature_names.append(type1 + '_month_money_trend')
    for type2 in type2s:
        for a in aggrs:
            feature_names.append(type1 + '_' + type2 + '_' + a + '_' + 'money_AGG_' + overall)

    for type2 in type2s:
        feature_names.append(type1 + '_' + type2 + '_' + 'repeat_' + overall)
    feature_names.append('label')
    # print(feature_names)
    # print(len(feature_names))
    return feature_names


def train_generator(months, overall, label_month):
    datas = pd.read_csv("../references.csv", dtype='object')
    datas = datas.fillna(0)
    indexs = datas['U_overall_money_'+overall].as_matrix().tolist()

    vps = []
    for i in indexs:

        if i != 0:
            tmp = i.split('-')
            # tmp[0]是vipno
            vps.append(tmp[0])
    vps = np.array(vps)
    feature_names = feature_name_generator()
    train_datas = DataFrame(np.zeros(shape=(len(vps), len(feature_names))), columns=feature_names, dtype='float')
    # tmp = DataFrame(vps, columns=['vipno', 'pluno'], dtype='object')
    # print(tmp)
    train_datas = pd.concat([train_datas, DataFrame(vps, columns=['vipno'])], axis=1)
    # print(train_datas)
    # print(train_datas.loc['22102005'])

    # 不同的阵容，存储的格式不一样，所以分开处理
    train_datas.set_index(['vipno'], inplace=True, drop=False)
    # print(train_datas.index)
    start = datetime.datetime.now()
    for f in feature_names[0:8]:
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

    start = datetime.datetime.now()
    for f in feature_names[18:32]:
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
    # train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=False)
    for index, row in train_datas.iterrows():

        tmp = []
        for m in months:
            tmp.append(row['U_month_count_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'],), feature_names[8]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'],), feature_names[9]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'],), feature_names[10]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'],), feature_names[11]] = tmp[1]
        train_datas.loc[(row['vipno'],), feature_names[12]] = least_squares(np.array([i for i in range(1, 4)]),
                                                                            np.array(tmp))

        tmp = []
        for m in months:
            tmp.append(row['U_month_money_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'],), feature_names[13]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'],), feature_names[14]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'],), feature_names[15]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'],), feature_names[16]] = tmp[1]
        train_datas.loc[(row['vipno'],), feature_names[17]] = least_squares(np.array([i for i in range(1, 4)]),
                                                                            np.array(tmp))

    print(datetime.datetime.now() - start)
    print("***************")

    start = datetime.datetime.now()
    labels = datas['U_month_money_'+label_month].as_matrix().tolist()
    indexs = train_datas.index
    for label in labels:
        # 0代表空值
        if label != 0:
            label = label.split('-')
            if label[0] in indexs:
                train_datas.loc[(label[0],), 'label'] = float(label[1])
    print(datetime.datetime.now() - start)
    print("***************")

    return train_datas


def money_error(y_true, y_pred):
    res = []
    for (t, p) in zip(y_true, y_pred):
        res.append(t-p)
    res.sort()
    return res


def cdf_figure(all_errors):
    """
    绘制的是对于每一个MR所生成的模型的CDF图
    :param all_errors:
    :return:
    """
    plt.figure('Comparision 2G DATA')
    # ax = plt.gca()
    plt.xlabel('CDF')
    plt.ylabel('Error(¥)')
    labels = ['KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    i = 0
    for errors in all_errors:
        mean_errors = np.array(errors).mean(axis=0)

        plt.plot([float(i)/float(len(mean_errors)) for i in range(len(mean_errors))],
                 list(mean_errors), '--', linewidth=1, alpha=0.6,
                 label=labels[i] + " median error: %.2f" % np.percentile(mean_errors, 50))
        i += 1
    plt.legend()
    plt.show()


def main():
    print("Generator train data!")
    months = ['02', '03', '04']
    overall = '234'
    label_month = '05'
    train_datas = train_generator(months, overall, label_month)
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X_train_all = np.delete(train, train.shape[1] - 1, axis=1)
    y_train_all = train[:, train.shape[1] - 1]

    print("Generator test data!")
    months = ['03', '04', '05']
    overall = '345'
    label_month = '06'
    test_datas = train_generator(months, overall, label_month)
    test_datas.set_index(['vipno'], inplace=True, drop=True)
    test = test_datas.as_matrix()
    X_test_all = np.delete(test, test.shape[1] - 1, axis=1)
    y_test_all = test[:, test.shape[1] - 1]

    # 用于做降采样，以确保正负样本的数量相近
    # rus = RandomUnderSampler(return_indices=True)
    # X_train_all, y_train_all, idx_resampled = rus.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all, idx_resampled = rus.fit_sample(X_test_all, y_test_all)
    # 同理测试过采样效果
    # ros = RandomOverSampler(random_state=0)
    # X_train_all, y_train_all = ros.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = ros.fit_sample(X_test_all, y_test_all)

    # X_train_all, y_train_all = SMOTE(kind='borderline1').fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = SMOTE(kind='borderline1').fit_sample(X_test_all, y_test_all)

    # smote_enn = SMOTEENN(random_state=0)
    # X_train_all, y_train_all = smote_enn.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = smote_enn.fit_sample(X_test_all, y_test_all)

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    all_errors = []

    # K临近
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = KNeighborsRegressor()
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("KNeighbors median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))


    # 决策树
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = DecisionTreeRegressor(max_depth=20)
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("DecisionTree median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))

    # 随机森林
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = RandomForestRegressor(max_depth=20, random_state=0)
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("RandomForest median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))

    # AdaBoost
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = AdaBoostRegressor(n_estimators=90, learning_rate=0.02)
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("AdaBoost median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))

    # Bagging
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = BaggingRegressor(n_estimators=20)
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("Bagging median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))

    # GradientBoosting
    errors = []
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        regr = GradientBoostingRegressor(learning_rate=0.02, n_estimators=50,
                                         max_depth=13, max_features=19, subsample=0.6)
        y_pred = regr.fit(X_train, y_train).predict(X_test)
        errors.append(money_error(y_test, y_pred))
    all_errors.append(errors)
    print("GradientBoosting median error: {}".format(np.percentile(np.array(errors).mean(axis=0), 50)))

    cdf_figure(all_errors)


if __name__ == '__main__':
    main()