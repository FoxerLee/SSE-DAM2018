import pandas as pd
import numpy as np
import datetime

from pandas import DataFrame


def add_month_day():
    datas = pd.read_csv("../trade_new.csv")
    datas.drop(columns=['Unnamed: 0'], inplace=True)
    columns = datas.columns.tolist()
    columns.append('month')
    columns.append('day')
    # print(columns)

    # datas.set_index(['sldatime'], inplace=True, drop=False)
    # datas.sort_index(inplace=True)
    datas = datas.as_matrix().tolist()
    for data in datas:
        sldatime = data[1]
        month = sldatime.split(' ')[0].split('-')[1]
        day = sldatime.split(' ')[0].split('-')[2]
        data.append(month)
        data.append(day)
        # print(day)
    datas = DataFrame(datas, columns=columns)

    datas.set_index(['month'], inplace=True, drop=False)
    datas.sort_index(inplace=True)
    # print(datas)
    return datas


def X_Y_count(datas, type1, type2):
    """
    对于type1字段（如vipno），其在当前数据集中，购买的type2字段（如pluno）的次数
    :param datas:
    :param type1:
    :param type2:
    :return:
    """
    types1 = list(set(datas[type1].as_matrix().tolist()))
    types2 = list(set(datas[type2].as_matrix().tolist()))

    indexs = [(x, y) for x in types1 for y in types2]
    aggr = dict.fromkeys(indexs, 0)
    for index, data in datas.iterrows():
        t1 = data[type1]
        t2 = data[type2]
        aggr[tuple((t1, t2))] += 1

    res = []
    for (key, value) in aggr.items():
        # -1代表是空值
        if value != 0 and key[1] != -1:
            row = str(key[0]) + '-' + str(key[1]) + '-' + str(value)
            res.append(row)
    return res


def X_Y_pen_div(datas, type1, type2):
    """
    对于type1字段（如pluno），其在当前数据集中，包含的不同type2字段（如vipno）个数
    :param datas:
    :param type1:
    :param type2:
    :return:
    """
    types1 = list(set(datas[type1].as_matrix().tolist()))

    # print(types1)
    aggr = dict.fromkeys(types1, [])
    # print(aggr)
    # print(aggr['14721027'])
    # aggr['14721027'] = list(aggr['14721027'] + ['1'])
    # print(aggr['14721027'])
    # print(aggr['14721033'])
    for index, data in datas.iterrows():
        aggr[data[type1]] = list(aggr[data[type1]] + [data[type2]])

    res = []
    # print(aggr)
    for (key, value) in aggr.items():
        # -1代表是空值
        if len(set(value)) != 0 and key != -1:
            row = str(key) + '-' + str(len(set(value)))
            res.append(row)
    return res


def X_Y_AGG(datas, type1, type2):
    """
    对于type1字段，针对type2字段统计购买的次数，再进行aggregation，agg操作包含mean、std、max、median
    :param datas:
    :param type1:
    :param type2:
    :return:
    """
    types1 = list(set(datas[type1].as_matrix().tolist()))
    aggr = dict.fromkeys(types1, [])
    for index, data in datas.iterrows():
        aggr[data[type1]] = list(aggr[data[type1]] + [data[type2]])

    res = []
    for (key, value) in aggr.items():
        # -1代表是空值
        if len(set(value)) != 0 and key != -1:
            tmp = []
            for i in set(value):
                tmp.append(value.count(i))
            tmp.sort()
            res.append([key, tmp])
    mean = []
    std = []
    max = []
    median = []
    for r in res:
        mean.append(str(r[0]) + '-' + str(np.array(r[1]).mean()))
        std.append(str(r[0]) + '-' + str(np.array(r[1]).std()))
        max.append(str(r[0]) + '-' + str(np.array(r[1]).max()))
        median.append(str(r[0]) + '-' + str(r[1][int(len(r[1])/2)]))
    return mean, std, max, median


def main():
    month_datas = add_month_day()
    # 填补bndno的空值
    month_datas['bndno'] = month_datas['bndno'].fillna(-1)
    months = list(set(month_datas.index.tolist()))
    months.sort()
    references = DataFrame()

    print("TYPE.1 count/ratio - count")
    start = datetime.datetime.now()
    # 对于每个vipno，其在每一个月所购买的某种pluno的次数
    for month in months:
        res = X_Y_count(month_datas.loc[month], 'vipno', 'pluno')
        U_I_month_counts = DataFrame(res, columns=['U_I_month_count_'+month])
        references = pd.concat([references, U_I_month_counts], axis=1)
    # 对于每个vipno，其总体购买的某种pluno的次数
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_count(datas, 'vipno', 'pluno')
    U_I_234_counts = DataFrame(res, columns=['U_I_overall_count_234'])
    references = pd.concat([references, U_I_234_counts], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_count(datas, 'vipno', 'pluno')
    U_I_567_counts = DataFrame(res, columns=['U_I_overall_count_567'])
    references = pd.concat([references, U_I_567_counts], axis=1)

    # 对于每个vipno，其在每一个月所购买的某种dptno的次数
    for month in months:
        res = X_Y_count(month_datas.loc[month], 'vipno', 'dptno')
        U_C_month_counts = DataFrame(res, columns=['U_C_month_count_' + month])
        references = pd.concat([references, U_C_month_counts], axis=1)
    # 对于每个vipno，其总体购买的某种dptno的次数
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_count(datas, 'vipno', 'dptno')
    U_C_234_counts = DataFrame(res, columns=['U_C_overall_count_234'])
    references = pd.concat([references, U_C_234_counts], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_count(datas, 'vipno', 'dptno')
    U_C_567_counts = DataFrame(res, columns=['U_C_overall_count_567'])
    references = pd.concat([references, U_C_567_counts], axis=1)

    # 对于每个vipno，其在每一个月所购买的某种bndno的次数
    for month in months:
        res = X_Y_count(month_datas.loc[month], 'vipno', 'bndno')
        U_B_month_counts = DataFrame(res, columns=['U_B_month_count_' + month])
        references = pd.concat([references, U_B_month_counts], axis=1)
    # 对于每个vipno，其总体购买的某种bndno的次数
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_count(datas, 'vipno', 'bndno')
    U_B_234_counts = DataFrame(res, columns=['U_B_overall_count_234'])
    references = pd.concat([references, U_B_234_counts], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_count(datas, 'vipno', 'bndno')
    U_B_567_counts = DataFrame(res, columns=['U_B_overall_count_567'])
    references = pd.concat([references, U_B_567_counts], axis=1)

    print(datetime.datetime.now() - start)
    print("*************************")

    print("TYPE.1 count/ratio - penetration")
    start = datetime.datetime.now()
    # 对于每个pluno，其在每一个月被多少个vipno所购买
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'pluno', 'vipno')
        I_U_month_penetration = DataFrame(res, columns=['I_U_month_penetration_'+month])
        references = pd.concat([references, I_U_month_penetration], axis=1)
    # 对于每个pluno，其总体被多少个vipno所购买
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'pluno', 'vipno')
    I_U_234_penetration = DataFrame(res, columns=['I_U_overall_penetration_234'])
    references = pd.concat([references, I_U_234_penetration], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'pluno', 'vipno')
    I_U_567_penetration = DataFrame(res, columns=['I_U_overall_penetration_567'])
    references = pd.concat([references, I_U_567_penetration], axis=1)

    # 对于每个dptno，其在每一个月被多少个vipno所购买
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'dptno', 'vipno')
        C_U_month_penetration = DataFrame(res, columns=['C_U_month_penetration_' + month])
        references = pd.concat([references, C_U_month_penetration], axis=1)
    # 对于每个dptno，其总体被多少个vipno所购买
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'dptno', 'vipno')
    C_U_234_penetration = DataFrame(res, columns=['C_U_overall_penetration_234'])
    references = pd.concat([references, C_U_234_penetration], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'dptno', 'vipno')
    C_U_567_penetration = DataFrame(res, columns=['C_U_overall_penetration_567'])
    references = pd.concat([references, C_U_567_penetration], axis=1)

    # 对于每个bndno，其在每一个月被多少个vipno所购买
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'bndno', 'vipno')
        B_U_month_penetration = DataFrame(res, columns=['B_U_month_penetration_' + month])
        references = pd.concat([references, B_U_month_penetration], axis=1)
    # 对于每个bndno，其总体被多少个vipno所购买
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'bndno', 'vipno')
    B_U_234_penetration = DataFrame(res, columns=['B_U_overall_penetration_234'])
    references = pd.concat([references, B_U_234_penetration], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'bndno', 'vipno')
    B_U_567_penetration = DataFrame(res, columns=['B_U_overall_penetration_567'])
    references = pd.concat([references, B_U_567_penetration], axis=1)

    print(datetime.datetime.now() - start)
    print("*************************")

    print("TYPE.1 count/ratio - product diversity")
    start = datetime.datetime.now()
    # 对于每个vipno，其在每一个月买过多少个pluno
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'vipno', 'pluno')
        U_I_month_diversity = DataFrame(res, columns=['U_I_month_diversity_' + month])
        references = pd.concat([references, U_I_month_diversity], axis=1)
    # 对于每个pluno，其总体买过多少个vipno
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'vipno', 'pluno')
    U_I_234_diversity = DataFrame(res, columns=['U_I_overall_diversity_234'])
    references = pd.concat([references, U_I_234_diversity], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'vipno', 'pluno')
    U_I_567_diversity = DataFrame(res, columns=['U_I_overall_diversity_567'])
    references = pd.concat([references, U_I_567_diversity], axis=1)

    # 对于每个vipno，其在每一个月买过多少个dptno
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'vipno', 'dptno')
        U_C_month_diversity = DataFrame(res, columns=['U_C_month_diversity_' + month])
        references = pd.concat([references, U_C_month_diversity], axis=1)
    # 对于每个dptno，其总体买过多少个vipno
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'vipno', 'dptno')
    U_C_234_diversity = DataFrame(res, columns=['U_C_overall_diversity_234'])
    references = pd.concat([references, U_C_234_diversity], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'vipno', 'dptno')
    U_C_567_diversity = DataFrame(res, columns=['U_C_overall_diversity_567'])
    references = pd.concat([references, U_C_567_diversity], axis=1)

    # 对于每个vipno，其在每一个月买过多少个bndno
    for month in months:
        res = X_Y_pen_div(month_datas.loc[month], 'vipno', 'bndno')
        U_B_month_diversity = DataFrame(res, columns=['U_B_month_diversity_' + month])
        references = pd.concat([references, U_B_month_diversity], axis=1)
    # 对于每个bndno，其总体买过多少个vipno
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    res = X_Y_pen_div(datas, 'vipno', 'bndno')
    U_B_234_diversity = DataFrame(res, columns=['U_B_overall_diversity_234'])
    references = pd.concat([references, U_B_234_diversity], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    res = X_Y_pen_div(datas, 'vipno', 'bndno')
    U_B_567_diversity = DataFrame(res, columns=['U_B_overall_diversity_567'])
    references = pd.concat([references, U_B_567_diversity], axis=1)

    print(datetime.datetime.now() - start)
    print("*************************")

    print("TYPE.2 AGG feature - brand/category/item AGG")
    start = datetime.datetime.now()
    # 对于某一个vipno，先针对pluno进行统计，然后进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'pluno', 'vipno')
    tmp = DataFrame({'U_I_mean_count_AGG_234': mean, 'U_I_std_count_AGG_234': std, 'U_I_max_count_AGG_234': max,
                     'U_I_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'pluno', 'vipno')
    tmp = DataFrame({'U_I_mean_count_AGG_567': mean, 'U_I_std_count_AGG_567': std, 'U_I_max_count_AGG_567': max,
                     'U_I_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    # 对于某一个vipno，先针对dptno进行统计，然后进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'dptno', 'vipno')
    tmp = DataFrame({'U_C_mean_count_AGG_234': mean, 'U_C_std_count_AGG_234': std, 'U_C_max_count_AGG_234': max,
                     'U_C_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'dptno', 'vipno')
    tmp = DataFrame({'U_C_mean_count_AGG_567': mean, 'U_C_std_count_AGG_567': std, 'U_C_max_count_AGG_567': max,
                     'U_C_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    # 对于某一个vipno，先针对bndno进行统计，然后进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'bndno', 'vipno')
    tmp = DataFrame({'U_B_mean_count_AGG_234': mean, 'U_B_std_count_AGG_234': std, 'U_B_max_count_AGG_234': max,
                     'U_B_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'bndno', 'vipno')
    tmp = DataFrame({'U_B_mean_count_AGG_567': mean, 'U_B_std_count_AGG_567': std, 'U_B_max_count_AGG_567': max,
                     'U_B_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    print(datetime.datetime.now() - start)
    print("*************************")

    print("TYPE.2 AGG feature - user AGG")
    start = datetime.datetime.now()
    # 对于某个pluno，针对单个vipno进行统计总时间内购买的次数，再进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'pluno')
    tmp = DataFrame({'I_U_mean_count_AGG_234': mean, 'I_U_std_count_AGG_234': std, 'I_U_max_count_AGG_234': max,
                     'I_U_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'pluno')
    tmp = DataFrame({'I_U_mean_count_AGG_567': mean, 'I_U_std_count_AGG_567': std, 'I_U_max_count_AGG_567': max,
                     'I_U_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    # 对于某个dptno，针对单个vipno进行统计总时间内购买的次数，再进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'dptno')
    tmp = DataFrame({'C_U_mean_count_AGG_234': mean, 'C_U_std_count_AGG_234': std, 'I_U_max_count_AGG_234': max,
                     'C_U_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'dptno')
    tmp = DataFrame({'C_U_mean_count_AGG_567': mean, 'C_U_std_count_AGG_567': std, 'C_U_max_count_AGG_567': max,
                     'C_U_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    # 对于某个bndno，针对单个vipno进行统计总时间内购买的次数，再进行aggregation
    # 2,3,4月
    datas = pd.concat([month_datas.loc['02'], month_datas.loc['03'], month_datas.loc['04']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'bndno')
    tmp = DataFrame({'B_U_mean_count_AGG_234': mean, 'B_U_std_count_AGG_234': std, 'B_U_max_count_AGG_234': max,
                     'B_U_median_count_AGG_234': median})
    references = pd.concat([references, tmp], axis=1)
    # 5,6,7月
    datas = pd.concat([month_datas.loc['05'], month_datas.loc['06'], month_datas.loc['07']])
    mean, std, max, median = X_Y_AGG(datas, 'vipno', 'bndno')
    tmp = DataFrame({'B_U_mean_count_AGG_567': mean, 'B_U_std_count_AGG_567': std, 'B_U_max_count_AGG_567': max,
                     'B_U_median_count_AGG_567': median})
    references = pd.concat([references, tmp], axis=1)

    print(datetime.datetime.now() - start)
    print("*************************")
    # print(references)

    references.to_csv("../references.csv")
    # print(month_datas)
    # overall_datas = pd.read_csv("../trade_new.csv")


if __name__ == '__main__':
    main()