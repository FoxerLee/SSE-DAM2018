import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import utils # 减少代码重复


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

    y_pred = KMeans(n_init=1, random_state=0).fit_predict(ids)

    # x = [id[0] for id in ids]
    # y = [id[1] for id in ids]
    # plt.scatter(x, y, c=y_pred)
    plt.show()

    ids = [(id, cluster) for (id, cluster) in zip(ids, y_pred)]
    # print(ids)

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
        # errors_all.append([id, errors])
        print("****************************")
    median_errors = DataFrame(median_errors, columns=['id', 'median_error', 'cluster'])
    median_errors.set_index(['median_error'], inplace=True, drop=False)
    median_errors.sort_index(inplace=True)
    print(median_errors)

    MS_number = median_errors.shape[0]
    topk_worst = median_errors.iloc[int(MS_number * 0.8):][['id', 'cluster']].as_matrix().tolist()
    for worst in topk_worst:
        similars = median_errors[median_errors['cluster'] == worst[1]].as_matrix().tolist()

        MS_datas = worst_data = train_data.loc[worst[0]]
        X_worst = worst_data.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                                   'Num_connected'], axis=1, inplace=False).as_matrix()
        y_worst = worst_data[
            ['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()
        for similar in similars:
            MS_datas = pd.concat([MS_datas, train_data.loc[similar[0]]])

        X = MS_datas.drop(['IMSI', 'MRTime', 'Longitude', 'Latitude',
                           'Num_connected'], axis=1, inplace=False).as_matrix()
        y = MS_datas[
            ['rel_Longitude', 'rel_Latitude', 'Longitude', 'Latitude', 'Longitude_1', 'Latitude_1']].as_matrix()

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

            # overall_pre, top10_pre, top10_recall = utils.precision_recall(y_test[:, 0], y_pred)
            # errors.append(utils.pos_error(y_test, y_pred))
        median_error = np.percentile(np.array(errors).mean(axis=0), 50)
        print("Median error: {}".format(median_error))
        # median_errors.append([worst, median_error])
        # errors_all.append([id, errors])
        print("****************************")


if __name__ == '__main__':
    main()
