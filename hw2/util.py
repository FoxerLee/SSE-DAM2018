from operator import itemgetter


def max_sup(path):
    """
    用来统计支持度前十的频繁项集
    :param path:
    :return:
    """
    f = open(path, 'r')
    inputs = f.readlines()
    datas = []
    for i in inputs:
        i = i.split(' ')
        datas.append([i[:-2], int(i[-1][:-1])])

    print(datas)

    top10 = sorted(datas, key=itemgetter(1), reverse=True)
    # print(top5)
    for t in top10[:10]:
        print(t)


if __name__ == '__main__':
    max_sup('miao.txt')
