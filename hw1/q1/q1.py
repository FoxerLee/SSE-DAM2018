from lshash.lshash import LSHash
import pandas as pd
import numpy as np
import random
np.set_printoptions(threshold=np.inf)


def get_data():
    """
    利用pandas包，从csv文件中提取所需数据，聚合后转换为矩阵格式
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
    :param distance_func: 可选择的距离计算函数
    :return: 去除自身之后的该vipno对应knn的输出vipno
    """
    datas_set, datas_matrix = get_data()
    # vipno_nums 为vipno去重后的总数
    vipno_nums = len(datas_matrix[0])
    # 初始化lshash
    lsh = LSHash(int(vipno_nums * p_hash_size), len(datas_matrix[:, 0]))
    for i in range(vipno_nums):
        # extra_data为当前列对应的vipno值，作为之后输出的时候所想要的knn的输出vipno
        lsh.index(datas_matrix[:, i], extra_data=datas_set.columns[i])

    # 随机取一个vipno（这里是vipno对应的下标）
    random_vipno = random.randint(0, vipno_nums-1)

    print("hash size: {}".format(vipno_nums*p_hash_size))
    print("input vipno: {}".format(datas_set.columns[random_vipno]))
    vipno_res = []
    # num_results可以限制输出的结果个数，这里取前6个，因为第一个为输入列本身
    for res in lsh.query(datas_matrix[:, random_vipno], num_results=6, distance_func=distance_func):
        vipno_res.append(res[0][1])
    print("knn output(from 1 to 5): {}".format(vipno_res[1:]))
    return vipno_res[1:], datas_set.columns[random_vipno]


if __name__ == '__main__':
    # p_hash_size可选参数 0.01, 0.05, 0.1, 0.2, 0.3, 0.5
    # distance_func可选函数 "euclidean", "hamming", "true_euclidean", "centred_euclidean", "cosine", "l1norm" 详细介绍会在文档当中
    p_hash_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    distance_func = ["euclidean", "hamming", "true_euclidean", "centred_euclidean", "cosine", "l1norm"]

    res, random_vipno = lsh(p_hash_size[1], distance_func[5])


    