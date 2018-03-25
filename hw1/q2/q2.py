from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
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

def K_Means():
    datas_set, datas_matrix = get_data()
    datas_matrix_T = datas_matrix.T
    # print(type(datas_matrix_T[0][0]))
    # vipno_nums 为vipno去重后的总数
    vipno_nums = len(datas_matrix[0])
    # 初始的的聚类数量k ≈ √n/2
    start_n_cluster = int(math.sqrt(vipno_nums)/2)

    range_n_clusters = []
    # 做k的遍历取值
    for i in range(2, 11):
        # if i == start_n_cluster:
        #     continue
        range_n_clusters.append(i)

    # 对于每一个k值，求得silhouette系数
    range_silhouette_avg = []
    for n_cluster in range_n_clusters:

        clusterer = KMeans(n_clusters=n_cluster)
        cluster_labels = clusterer.fit_predict(datas_matrix_T)

        silhouette_avg = silhouette_score(datas_matrix_T, cluster_labels)
        range_silhouette_avg.append(silhouette_avg)
        print("For n_clusters =", n_cluster,
              "The average silhouette_score is :", silhouette_avg)

    # sample_silhouette_values = silhouette_samples(datas_matrix_T, cluster_labels)
    # print(sample_silhouette_values)

    # 做得Silhouette系数值-k值的函数图
    plt.plot(range_n_clusters, range_silhouette_avg, 'ro-')
    plt.title('Silhouette-k line chart')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    K_Means()
