import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

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
from imblearn.combine import SMOTEENN

import util


def main():
    print("Generator train data!")
    months = ['02', '03', '04']
    overall = '234'
    label_month = '05'
    train_datas = util.train_generator(months, overall, label_month)
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X_train_all = np.delete(train, train.shape[1] - 1, axis=1)
    y_train_all = train[:, train.shape[1] - 1]

    print("Generator test data!")
    months = ['03', '04', '05']
    overall = '345'
    label_month = '06'
    test_datas = util.train_generator(months, overall, label_month)
    test_datas.set_index(['vipno', 'pluno'], inplace=True, drop=True)
    test = test_datas.as_matrix()
    X_test_all = np.delete(test, test.shape[1] - 1, axis=1)
    y_test_all = test[:, test.shape[1] - 1]

    # 用于做降采样，以确保正负样本的数量相近
    # rus = RandomUnderSampler(return_indices=True)
    # X_train_all, y_train_all, idx_resampled = rus.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all, idx_resampled = rus.fit_sample(X_test_all, y_test_all)
    # 同理测试过采样效果
    # ros = RandomOverSampler(random_state=0)
    # X_train_all, y_train_all = ros.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = ros.fit_sample(X_test_all, y_test_all)

    # X_train_all, y_train_all = SMOTE(kind='borderline1').fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = SMOTE(kind='borderline1').fit_sample(X_test_all, y_test_all)

    smote_enn = SMOTEENN(random_state=0)
    X_train_all, y_train_all = smote_enn.fit_sample(X_train_all, y_train_all)
    # X_test_all, y_test_all = smote_enn.fit_sample(X_test_all, y_test_all)

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    all_fprs = []
    all_tprs = []
    all_thresholds = []
    all_aucs = []
    all_pres = []
    all_recalls = []

    # 高斯朴素贝叶斯分类器
    overall_pres = []
    overall_recalls = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        # 切分训练集和验证集
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        y_pred_proba = gnb.predict_proba(X_test)
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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

        clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, algorithm='SAMME.R')
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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
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
        X_train, _, y_train, _ = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                  random_state=random_states[i])
        _, X_test, _, y_test = train_test_split(X_test_all, y_test_all, test_size=0.2, random_state=random_states[i])

        clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_features=19, subsample=0.85,
                                         max_depth=3, min_samples_split=300)
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
    all_pres.append(np.mean(np.array(overall_pres)))
    all_recalls.append(np.mean(np.array(overall_recalls)))
    print("GradientBoosting")
    print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
    print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
    print("Overall auc: %.3f" % mean_auc)
    print("*********************")

    util.draw_roc(all_fprs, all_tprs, all_thresholds, all_aucs)
    util.figure(all_pres, all_recalls, all_aucs)


if __name__ == '__main__':
    main()
    # util.figure()
