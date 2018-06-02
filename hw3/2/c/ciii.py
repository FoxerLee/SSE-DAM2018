import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from pandas import DataFrame
from scipy import interp
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score, precision_score, classification_report
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


def feature_name_generator():
    type1 = 'U'
    type2 = 'C'
    months = ['02', '03', '04']
    aggrs = ['mean', 'std', 'max', 'median']
    overall = '234'
    feature_names = []
    # TYPE.1 count/ratio - count
    for m in months:
        feature_names.append(type1+'_'+type2+'_'+'month_count_'+m)
    feature_names.append(type1+'_'+type2+'_'+'overall_count_'+overall)
    # TYPE.1 count/ratio - penetration
    for m in months:
        feature_names.append(type2+'_'+type1+'_'+'month_penetration_'+m)
    feature_names.append(type2 + '_' + type1 + '_' + 'overall_penetration_' + overall)
    # TYPE.1 count/ratio - product diversity
    for m in months:
        feature_names.append(type1 + '_' + type2 + '_' + 'month_diversity_' + m)
    feature_names.append(type1 + '_' + type2 + '_' + 'overall_diversity_' + overall)
    # TYPE.2 AGG feature - brand/category/item AGG
    for a in aggrs:
        feature_names.append(type1+'_'+type2+'_'+a+'_'+'count_AGG_'+overall)
    # TYPE.2 AGG feature - user AGG
    for a in aggrs:
        feature_names.append(type2+'_'+type1+'_'+a+'_'+'count_AGG_'+overall)
    # TYPE.2 AGG feature - month AGG
    for a in aggrs:
        feature_names.append(type1 + '_' + type2 + '_month_count_' + a)
    for a in aggrs:
        feature_names.append(type2 + '_' + type1 + '_month_penetration_' + a)
    for a in aggrs:
        feature_names.append(type1 + '_' + type2 + '_month_diversity_' + a)

    feature_names.append('label')
    # print(feature_names)
    # print(len(feature_names))
    return feature_names


def train_generator():
    datas = pd.read_csv("../references.csv", dtype='object')
    datas = datas.fillna(0)
    indexs = datas['U_C_overall_count_234'].as_matrix().tolist()

    vps = []
    for i in indexs:
        if i == 0:
            continue
        tmp = i.split('-')
        # tmp[0]是vipno，tmp[1]是vptno
        vps.append([tmp[0], tmp[1]])
    vps = np.array(vps)
    feature_names = feature_name_generator()

    train_datas = DataFrame(np.zeros(shape=(len(vps), len(feature_names))), columns=feature_names, dtype='float')
    # tmp = DataFrame(vps, columns=['vipno', 'vptno'], dtype='object')
    # print(tmp)
    train_datas = pd.concat([train_datas, DataFrame(vps, columns=['vipno', 'vptno'])], axis=1)
    # print(train_datas)
    # print(train_datas.loc['22102005'])

    # 不同的阵容，存储的格式不一样，所以分开处理
    start = datetime.datetime.now()
    train_datas.set_index(['vipno', 'vptno'], inplace=True, drop=False)
    for f in feature_names[:4]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[(tmp[0], tmp[1]), f] = float(tmp[2])
    print(datetime.datetime.now() - start)
    print("***************")
    start = datetime.datetime.now()
    train_datas.set_index(['vptno'], inplace=True, drop=False)
    for f in feature_names[4:8]:
        # count = 0
        ds = datas[f].as_matrix().tolist()
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")
    start = datetime.datetime.now()
    for f in feature_names[16:20]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            # print(count)
            # count += 1
            train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")
    train_datas.set_index(['vipno'], inplace=True, drop=False)

    start = datetime.datetime.now()
    for f in feature_names[8:16]:
        ds = datas[f].as_matrix().tolist()
        # count = 0
        for row in ds:
            if row == 0:
                continue
            tmp = row.split('-')
            if tmp[0] in train_datas.index.tolist():
                train_datas.loc[tmp[0], f] = float(tmp[1])
    print(datetime.datetime.now() - start)
    print("***************")

    months = ['02', '03', '04']
    start = datetime.datetime.now()
    train_datas.set_index(['vipno', 'vptno'], inplace=True, drop=False)
    for index, row in train_datas.iterrows():
        tmp = []
        for m in months:
            tmp.append(row['U_C_month_count_'+m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[20]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[21]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[22]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[23]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['C_U_month_penetration_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[24]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[25]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[26]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[27]] = tmp[1]

        tmp = []
        for m in months:
            tmp.append(row['U_C_month_diversity_' + m])
        tmp.sort()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[28]] = np.array(tmp).mean()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[29]] = np.array(tmp).std()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[30]] = np.array(tmp).max()
        train_datas.loc[(row['vipno'], row['vptno']), feature_names[31]] = tmp[1]

    print(datetime.datetime.now() - start)
    print("***************")

    start = datetime.datetime.now()
    labels = datas['U_C_month_count_05'].as_matrix().tolist()
    indexs = train_datas.index
    for label in labels:
        # 0代表空值
        if label != 0:
            label = label.split('-')
            if (label[0], label[1]) in indexs:
                train_datas.loc[(label[0], label[1]), 'label'] = 1
    print(datetime.datetime.now() - start)
    print("***************")
    return train_datas


def draw_roc(fprs, tprs, thresholds, aucs):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    labels = ['Gaussian', 'Kmeans', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    for i in range(len(aucs)):
        plt.plot(fprs[i], tprs[i], lw=lw,
                 label=labels[i] + ' ROC curve (auc = %0.3f)' % aucs[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.show()


def main():
    train_datas = train_generator()
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno', 'vptno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X = np.delete(train, train.shape[1] - 1, axis=1)
    y = train[:, train.shape[1] - 1]
    # 用于做降采样，以确保正负样本的数量相近
    # rus = RandomUnderSampler(return_indices=True)
    # X, y, idx_resampled = rus.fit_sample(X, y)
    # 同理测试过采样效果
    # ros = RandomOverSampler(random_state=0)
    # X, y = ros.fit_sample(X, y)
    X, y = SMOTE(kind='borderline1').fit_sample(X, y)

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    all_fprs = []
    all_tprs = []
    all_thresholds = []
    all_aucs = []

    # 高斯朴素贝叶斯分类器
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        y_pred_proba = gnb.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:,1], pos_label=1)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("Gaussian")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # K近邻分类器
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        y_pred_proba = neigh.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("KNeighbors")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # 决策树分类器
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("DecisionTree")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # 随机森林
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("RandomForest")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # AdaBoost
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = AdaBoostClassifier(base_estimator=None)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("AdaBoost")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # Bagging
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = BaggingClassifier(n_estimators=20)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("Bagging")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    # GradientBoosting
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[i])

        clf = GradientBoostingClassifier(n_estimators=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        overall_pres.append(precision_score(y_test, y_pred))
        overall_recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= 10
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)
    all_fprs.append(mean_fpr)
    all_tprs.append(mean_tpr)
    all_aucs.append(mean_auc)
    print("GradientBoosting")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    draw_roc(all_fprs, all_tprs, all_thresholds, all_aucs)


if __name__ == '__main__':
    main()
