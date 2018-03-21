from lshash.lshash import LSHash
import pandas as pd
import numpy as np
import random
np.set_printoptions(threshold=np.inf)


def get_data():
    """

    :return:
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
        datas_set.loc[row['pluno'], [row['vipno']]] = int(af)

    # 将DataFrame格式的数据转换为numpy中的array格式
    datas_matrix = datas_set.as_matrix()
    return datas_set, datas_matrix


def lsh(p_hash_size, distance_func):
    """

    :param p_hash_size:
    :param distance_func:
    :return:
    """
    datas_set, datas_matrix = get_data()
    vipno_nums = len(datas_matrix[0])
    lsh = LSHash(int(vipno_nums * p_hash_size), len(datas_matrix[:, 0]))
    for i in range(vipno_nums):
        lsh.index(datas_matrix[:, i], extra_data=datas_set.columns[i])

    random_vipno = random.randint(0, vipno_nums-1)

    print("hash size: {}".format(vipno_nums*p_hash_size))
    print("input vipno: {}".format(datas_set.columns[random_vipno]))
    vipno_res = []
    for res in lsh.query(datas_matrix[:, random_vipno], num_results=6, distance_func=distance_func):
        vipno_res.append(res[0][1])
    print("knn output(from 1 to 5): {}".format(vipno_res[1:]))


if __name__ == '__main__':
    # p_hash_size可选参数 0.01, 0.05, 0.1, 0.2, 0.3, 0.5
    # distance_func可选函数 "euclidean", "hamming", "true_euclidean", "centred_euclidean", "cosine", "l1norm" 详细介绍会在文档当中
    p_hash_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    distance_func = ["euclidean", "hamming", "true_euclidean", "centred_euclidean", "cosine", "l1norm"]

    lsh(p_hash_size[0], distance_func[5])