from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lshash.lshash import LSHash
import pandas as pd
import numpy as np
import random
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


def eps(minPts):
    """
    利用k-距离曲线图法，求到最好的Eps值
    :param minPts: 自定义的minPts值
    :return: 获取到的较好的Eps值
    """
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T

    # print(type(datas_matrix_T[0]))
    # 尝试做降维操作，但是效果不好
    # pca = PCA(n_components=2)
    # datas_matrix_T = pca.fit_transform(datas_matrix_T)
    res = []
    # 计算每个点的k-距离值
    for row1 in datas_matrix_T:
        temp = []
        for row2 in datas_matrix_T:
            temp.append(np.linalg.norm(row1-row2))
        temp.sort()
        res.append(temp[minPts])
    # 对所有点的k-距离集合进行升序排序
    res.sort()

    x_ = []
    for i in range(len(res)):
        x_.append(i)

    plt.plot(x_, res, 'ro')
    plt.show()
    # print(res[len(res)-10:])
    return res[len(res)-10:-1]
    # return res


def dbscan(minPts):
    """
    具体实现dbscan算法
    :param minPts: 自定义的minPts值
    :return: 无
    """
    datas_set, datas_matrix = get_data()
    res_vipno, random_vipno = lsh(0.01, "cosine")

    datas_matrix_T = datas_matrix.T
    X = datas_matrix_T
    # 尝试做降维操作
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(X)

    # X = StandardScaler().fit_transform(datas_matrix_T)
    # print(len(X))
    # print(len(datas_matrix_T))
    all_eps = eps(minPts)

    # 对于每一个k值，求得silhouette系数
    range_silhouette_avg = []
    for e in all_eps:
    # for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        db = DBSCAN(eps=e, min_samples=minPts).fit(X)

        cluster_labels = db.labels_
        print(len(cluster_labels))
        n_cluster = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        print("when eps =", e,
              "there are", n_cluster, "clusters")

        # 获得silhouette分数
        silhouette_avg = silhouette_score(X, db.labels_)
        range_silhouette_avg.append(silhouette_avg)
        print("For n_clusters =", n_cluster,
              "The average silhouette_score is :", silhouette_avg)

        res = 0
        # pos为q1中输入的随机vipno在kmeans中的分类结果
        pos = cluster_labels[datas_set.columns.get_loc(random_vipno)]
        # 逐个获取q1中输出的knn对应在kmeans中的分类结果，和pos比较
        for i in res_vipno:
            if cluster_labels[datas_set.columns.get_loc(i)] == pos:
                res += 1

        print("For k =", len(res_vipno),
              "There are", res, "in the same cluster as KMeans predicted")

    # 做Silhouette系数值-k值的函数图
    plt.plot(all_eps, range_silhouette_avg, 'ro-')
    plt.title('Silhouette-k line chart')
    plt.xlabel('eps values')
    plt.ylabel('The silhouette coefficient values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # eps(5)
    dbscan(5)
