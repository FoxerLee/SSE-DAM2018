from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lshash.lshash import LSHash
import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


def k_means():
    """
    KMeans的具体实现
    :return: 无
    """
    datas_set, datas_matrix = get_data()
    res_vipno, random_vipno = lsh(0.01, "cosine")

    datas_matrix_T = datas_matrix.T
    # 对于输入的数据做标准化，但是之后发现效果不好
    # X = StandardScaler().fit_transform(datas_matrix_T)
    X = datas_matrix_T
    # 尝试做降维操作，之后发现效果并不突出
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(X)


    # vipno_nums 为vipno去重后的总数
    vipno_nums = len(datas_matrix[0])
    # 初始的的聚类数量k ≈ √(n/2)
    start_n_cluster = int(math.sqrt(vipno_nums/2))

    range_n_clusters = []
    # 做k的遍历取值，这里的取值是参考的常见k值取2-10而定。详细说明请见文档
    for i in range(2, start_n_cluster + 3):
        range_n_clusters.append(i)

    # 对于每一个k值，求得silhouette系数
    range_silhouette_avg = []
    for n_cluster in range_n_clusters:
        fig, (ax1) = plt.subplots(1, 1)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
        clusterer = KMeans(n_clusters=n_cluster, init='k-means++', algorithm="full")
        cluster_labels = clusterer.fit_predict(X)
        # 获得silhouette分数
        silhouette_avg = silhouette_score(X, cluster_labels)
        range_silhouette_avg.append(silhouette_avg)
        print("For n_clusters =", n_cluster,
              "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_cluster):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_cluster)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.suptitle(("Silhouette analysis for KMeans "
                      "with n_clusters = %d" % n_cluster),
                     fontsize=14, fontweight='bold')
        plt.show()

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
    plt.plot(range_n_clusters, range_silhouette_avg, 'ro-')
    plt.title('Silhouette-k line chart')
    plt.xlabel('k values')
    plt.ylabel('The silhouette coefficient values')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    k_means()
