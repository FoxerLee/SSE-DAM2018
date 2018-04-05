from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from q1 import q1
np.set_printoptions(threshold=np.inf)

def get_data():
    """
    利用pandas包，从csv文件中提取所需数据，聚合后转换为矩阵格式。这里使用的函数和q1问一样
    :return: datas_set为DataFrame格式的矩阵和datas_matrix为array格式的矩阵
    """
    datas = pd.read_csv('/Users/liyuan/Documents/数据分析与数据挖掘/SSE-DAM2018/hw1/trade.csv', usecols=['vipno', 'pluno', 'amt'])

    # 利用pandas的groupby函数做聚合运算
    amts_set = datas.groupby([datas['vipno'], datas['pluno']], as_index = False).agg({'amt': sum})
    # 修改DataFrame的列中数据格式为object，否则后面会将vipno和pluno转换为float64
    amts_set[['vipno', 'pluno']] = amts_set[['vipno', 'pluno']].astype('object')

    # 将vipno和pluno提取出来然后去重，新建一个矩阵
    datas_set = pd.DataFrame(0, index=list(set(datas['pluno'])), columns=list(set(datas['vipno'])))

    # 根据行列索引将获得的聚合值放入矩阵当中，并且在放入前进行四舍五入
    for index, row in amts_set.iterrows():
        af = np.floor(row['amt'])
        ad = row['amt'] - af
        if ad >= 0.5:
            af = af + 1
        # 使用at会比loc快很多
        datas_set.at[row['pluno'], row['vipno']] = int(af)

    # 将DataFrame格式的数据转换为numpy中的array格式
    datas_matrix = datas_set.as_matrix()
    return datas_set, datas_matrix


def gmm_kmeans(n_cluster=2):
    # 数据获取，预处理做标准化
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T
    X = StandardScaler().fit_transform(datas_matrix_T)

    # 数据利用KMeans训练
    clusterer = KMeans(n_clusters=n_cluster)
    y_train = clusterer.fit_predict(X)

    estimator = GaussianMixture(n_components=n_cluster, covariance_type='diag')

    # 我们假定KMeans是真实的聚类结果，那么我们可以预先确定部分GMM参数
    estimator.means_init = np.array([X[y_train == i].mean(axis=0)
                                    for i in range(n_cluster)])
    estimator.fit(X)

    y_train_pred = estimator.predict(X)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print(y_train)
    print(y_train_pred)

    print("Comparing with KMeans, the accuracy of GMM is:", train_accuracy, "%")


def gmm_dbscan(minPts=5, e=1300):
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T

    X = datas_matrix_T

    db = DBSCAN(eps=e, min_samples=minPts).fit(X)
    y_train = db.labels_
    n_cluster = len(set(y_train)) - (1 if -1 in y_train else 0)

    estimator = GaussianMixture(n_components=n_cluster, covariance_type='tied')

    # 我们假定KMeans是真实的聚类结果，那么我们可以预先确定部分GMM参数
    estimator.means_init = np.array([X[y_train == i].mean(axis=0)
                                     for i in range(n_cluster)])
    estimator.fit(X)

    y_train_pred = estimator.predict(X)

    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print(y_train)
    print(y_train_pred)

    print("Comparing with KMeans, the accuracy of GMM is:", train_accuracy, "%")


if __name__ == '__main__':
    gmm_kmeans()
    # gmm_dbscan()


