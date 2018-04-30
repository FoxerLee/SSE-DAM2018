import csv



def merge_data():

    with open('../trade.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        trades = [row for row in reader]

    with open('../trade_new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        trades_new = [row for row in reader]

    trades_new = trades_new[1:]
    trades = trades[1:]
    # print(datas)

    uids = [row[0] for row in trades] + [row[1] for row in trades_new]

    merges = dict.fromkeys(uids, [])

    for row in trades:
        merges[row[0]] = list(set([row[6]] + merges[row[0]]))

    for row in trades_new:
        merges[row[1]] = list(set([row[8]] + merges[row[1]]))

    print(merges)
    a = list(merges.items())
    print(type(a[0]))


if __name__ == '__main__':
    merge_data()



