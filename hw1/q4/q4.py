from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from lshash.lshash import LSHash
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def get_data():
    """
    利用pandas包，从csv文件中提取所需数据，聚合后转换为矩阵格式。这里使用的函数和q1问一样
    :return: datas_set为DataFrame格式的矩阵和datas_matrix为array格式的矩阵
    """
    datas = pd.read_csv('trade.csv', usecols=['vipno', 'pluno', 'amt'])

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


def lsh(p_hash_size, distance_func):
    """
    实现局部敏感哈希模拟KNN的具体函数
    :param p_hash_size: 与vipno的总数（去重后）相乘构成最终的hash_size
    :param distance_funcs: 可选择的距离计算函数
    :return: 去除自身之后的该vipno对应knn的输出vipno
    """
    datas_set, datas_matrix = get_data()
    # vipno_nums 为vipno去重后的总数
    vipno_nums = len(datas_matrix[0])

    # 随机取一个vipno（这里是vipno对应的下标）
    random_vipno = random.randint(0, vipno_nums - 1)

    # 初始化lshash
    lsh = LSHash(int(vipno_nums * p_hash_size), len(datas_matrix[:, 0]))
    for i in range(vipno_nums):
        # extra_data为当前列对应的vipno值，作为之后输出的时候所想要的knn的输出vipno
        lsh.index(datas_matrix[:, i], extra_data=datas_set.columns[i])

    vipno_res = []
    # num_results可以限制输出的结果个数，这里取前6个，因为第一个为输入列本身
    for res in lsh.query(datas_matrix[:, random_vipno], num_results=6, distance_func=distance_func):
        vipno_res.append(res[0][1])

    print("distance func:", distance_func)
    print("knn output(from 1 to 5): {}".format(vipno_res[1:]))

    return vipno_res[1:], datas_set.columns[random_vipno]


def gmm_kmeans(n_cluster=2):
    """
    gmm与kmeans的比较
    :param n_cluster: q2取到的最优值，默认为最优
    :return: 无
    """
    # 数据获取
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T
    X = datas_matrix_T
    res_vipno, random_vipno = lsh(0.01, "cosine")

    # 数据利用KMeans训练
    clusterer = KMeans(n_clusters=n_cluster)
    y_train = clusterer.fit_predict(X)

    accs = []
    types = ['full', 'tied', 'diag', 'spherical']
    # 比较了四种协方差矩阵对应的acc值
    for type in types:
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
        accs.append(train_accuracy)
        res = 0
        # pos为q1中输入的随机vipno在gmm中的分类结果
        pos = y_train_pred[datas_set.columns.get_loc(random_vipno)]
        # 逐个获取q1中输出的knn对应在gmm中的分类结果，和pos比较
        for i in res_vipno:
            if y_train_pred[datas_set.columns.get_loc(i)] == pos:
                res += 1

        print("For k =", len(res_vipno),
              "There are", res, "in the same cluster as gmm predicted")

    # 做四种协方差的acc值图
    plt.bar(types, accs, alpha=0.9, width=0.35, facecolor='lightskyblue', edgecolor='white', label='time',
            lw=1)
    plt.title("four covariances` acc")
    plt.legend(loc="upper left")
    plt.show()


def gmm_dbscan(minPts=5, e=1300):
    """
    gmm算法与dbscan算法比较
    :param minPts: q3取到的最优值，默认为最优
    :param e: q3取到的最优半径，默认为最优
    :return: 无
    """
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T

    X = datas_matrix_T
    res_vipno, random_vipno = lsh(0.01, "cosine")

    db = DBSCAN(eps=e, min_samples=minPts).fit(X)
    y_train = db.labels_
    n_cluster = len(set(y_train)) - (1 if -1 in y_train else 0)

    accs = []
    types = ['full', 'tied', 'diag', 'spherical']
    # 比较了四种协方差矩阵对应的acc值
    for type in types:
        estimator = GaussianMixture(n_components=n_cluster, covariance_type='tied')

        # 我们假定KMeans是真实的聚类结果，那么我们可以预先确定部分GMM参数
        estimator.means_init = np.array([X[y_train == i].mean(axis=0)
                                         for i in range(n_cluster)])
        estimator.fit(X)

        y_train_pred = estimator.predict(X)

        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        print(y_train)
        print(y_train_pred)

        print("Comparing with DBScan, the accuracy of GMM is:", train_accuracy, "%")
        accs.append(train_accuracy)
        res = 0
        # pos为q1中输入的随机vipno在gmm中的分类结果
        pos = y_train_pred[datas_set.columns.get_loc(random_vipno)]
        # 逐个获取q1中输出的knn对应在gmm中的分类结果，和pos比较
        for i in res_vipno:
            if y_train_pred[datas_set.columns.get_loc(i)] == pos:
                res += 1

        print("For k =", len(res_vipno),
              "There are", res, "in the same cluster as GMM predicted")
    # 做四种协方差的acc值图
    plt.bar(types, accs, alpha=0.9, width=0.35, facecolor='lightskyblue', edgecolor='white', label='acc',
            lw=1)
    plt.title("four covariances` acc")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    # gmm_kmeans()
    gmm_dbscan()


