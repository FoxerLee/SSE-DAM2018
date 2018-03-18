from lshash.lshash import LSHash
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)

datas = pd.read_csv('trade.csv', usecols=['vipno', 'pluno', 'amt'])

# 利用pandas的groupby函数做聚合运算
amts_set = datas.groupby([datas['vipno'], datas['pluno']], as_index = False).agg({'amt': sum})
# 修改DataFrame的列中数据格式为object，否则后面输出会将vipno和pluno转换为float64
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

# datas_set.to_csv('test2.csv')

print(datas_set)





