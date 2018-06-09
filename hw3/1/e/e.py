import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import utils # 减少代码重复

# 左下角坐标
lb_Longitude = 121.20120490000001
lb_Latitude = 31.28175691
# 右上角坐标
rt_Longitude = 121.2183295
rt_Latitude = 31.29339344


def main():
    train_data = utils.gongcan_to_ll()
    # 删除原有的ID，不作为训练特征
    for i in range(1, 8):
        train_data.drop(['RNCID_' + str(i)], axis=1, inplace=True)
        train_data.drop(['CellID_' + str(i)], axis=1, inplace=True)

    # 将空余的信号强度，用0补填补
    train_data = train_data.fillna(0)
    rel_lon = []
    rel_lat = []
    # print(train_data)
    for index, row in train_data.iterrows():
        rel_lon.append(row['Longitude'] - row['Longitude_1'])
        rel_lat.append(row['Latitude'] - row['Latitude_1'])

    train_data['rel_Longitude'] = np.array(rel_lon)
    train_data['rel_Latitude'] = np.array(rel_lat)

    # features和labels
    train_data.set_index(['Longitude_1', 'Latitude_1'], inplace=True, drop=False)
    train_data.sort_index(inplace=True)
    ids = list(set(train_data.index.tolist()))

    # 利用 KMeans 聚类，将不同的基站通过距离进行聚类
    y_pred = KMeans(n_init=1, random_state=0).fit_predict(ids)
    # print(y_pred)

    # 做出聚类后的结果
    plt.title("Kmeans Result")
    x = [id[0] for id in ids]
    y = [id[1] for id in ids]
    plt.scatter(x, y, c=y_pred)
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # plt.xlim([lb_Longitude, rt_Longitude])
    # plt.ylim([lb_Latitude, rt_Latitude])
    plt.show()

    ids = [(id, cluster) for (id, cluster) in zip(ids, y_pred)]
    # print(ids)
    errors_all = []
    median_errors = []
    for id in ids:
        MS_datas = train_data.loc[id[0]]
        X = MS_datas.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                           'Num_connected'], axis=1, inplace=False).as_matrix()
        y = MS_datas[
            ['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()

        # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
        random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        # 随机森林
        print("MS {}".format(id))
        errors = []
        for i in range(10):
            # 切分训练集和验证集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

            regr = RandomForestRegressor(max_depth=20, random_state=0)
            y_pred = regr.fit(X_train, np.delete(y_train, [2, 3, 4, 5], axis=1)).predict(X_test)
            error = utils.pos_error(y_test, y_pred)
            errors.append(error)

            # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
            # errors.append(utils.pos_error(y_test, y_pred))
        median_error = np.percentile(np.array(errors).mean(axis=0), 50)
        print("Median error: {}".format(median_error))
        median_errors.append([id[0], median_error, id[1]])
        errors_all.append([id, errors])
        print("****************************")
    median_errors = DataFrame(median_errors, columns=['id', 'median_error', 'cluster'])
    median_errors.set_index(['median_error'], inplace=True, drop=False)
    median_errors.sort_index(inplace=True)

    MS_number = median_errors.shape[0]
    topk_worst = median_errors.iloc[int(MS_number * 0.8):][['id', 'cluster']].as_matrix()

    old_errors = []  # 用于存储没有修正前的 top k- 的所有 error
    for error in errors_all:
        if error[0][0] in topk_worst[:,0].tolist():
            old_errors.append([error[0], error[1]])

    print("\n")
    print("Start correction")
    print("\n")

    new_errors = []  # 用于存储修正后的 top k- 的所有 error
    for worst in topk_worst:
        similars = median_errors[median_errors['cluster'] == worst[1]].as_matrix().tolist()

        MS_datas = worst_data = train_data.loc[worst[0]]
        X_worst = worst_data.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                                   'Num_connected'], axis=1, inplace=False).as_matrix()
        y_worst = worst_data[
            ['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()
        for similar in similars:
            MS_datas = pd.concat([MS_datas, train_data.loc[similar[0]]])

        # 随机抽样
        # MS_datas = MS_datas.sample(frac=0.8)

        X = MS_datas.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                           'Num_connected'], axis=1, inplace=False).as_matrix()
        y = MS_datas[
            ['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()

        # X = []
        # y = []
        #
        # # 筛选，删掉距离原始数据集过远的数据
        # for i, j in zip(X_, y_):
        #     error = utils.haversine(j[4], j[5], worst[0][0], worst[0][1])
        #     if error > 500:
        #         continue
        #     X.append(i)
        #     y.append(j)
        # X = np.array(X)
        # y = np.array(y)

        # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的，同时每一次的实验的数据集也是一样的，从而提升结果的可信度
        random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        # 随机森林
        print("MS {}".format(worst))
        errors = []
        for i in range(10):
            # 切分训练集和验证集
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=random_states[i])
            _, X_test, _, y_test = train_test_split(X_worst, y_worst, test_size=0.2, random_state=random_states[i])
            regr = RandomForestRegressor(max_depth=20, random_state=0)
            y_pred = regr.fit(X_train, np.delete(y_train, [2, 3, 4, 5], axis=1)).predict(X_test)
            error = utils.pos_error(y_test, y_pred)
            errors.append(error)

        # 做出每个基站用于加入了新的数据集后的所有训练数据点位置和原始该 MS 基站的数据点位置
        plt.title("Median error: %.3f" % np.percentile(np.array(errors).mean(axis=0), 50))
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.scatter(y[:, 2], y[:, 3], label='new data')
        plt.scatter(y_worst[:, 2], y_worst[:, 3], label='old data')

        plt.xlim([lb_Longitude, rt_Longitude])
        plt.ylim([lb_Latitude, rt_Latitude])
        plt.legend()
        plt.show()

        median_error = np.percentile(np.array(errors).mean(axis=0), 50)
        print("Median error: {}".format(median_error))
        new_errors.append([worst, errors])
        print("****************************")

    utils.cdf_figure(old_errors, new_errors)
    utils.mean_figure(old_errors, new_errors)


if __name__ == '__main__':
    main()
