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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score, precision_score, classification_report
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

import b


def main():
    train_datas = b.train_generator()
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno', 'pluno'], inplace=True, drop=True)
    features = train_datas.columns[:-1]

    train = train_datas.as_matrix()
    X = np.delete(train, train.shape[1] - 1, axis=1)

    y = train[:, train.shape[1] - 1]
    # 用于做降采样，以确保正负样本的数量相近
    rus = RandomUnderSampler(return_indices=True, random_state=128)
    X, y, idx_resampled = rus.fit_sample(X, y)

    # 通过设置每一次的随机数种子，保证不同分类器每一次的数据集是一样的
    random_states = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    all_pres = []
    all_recalls = []
    all_aucs = []

    for i in range(X.shape[1]):
        overall_pres = []
        overall_recalls = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for j in range(10):
            # 切分训练集和验证集
            X_train, X_test, y_train, y_test = train_test_split(np.delete(X, i, axis=1), y, test_size=0.2, random_state=random_states[j])

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
        all_pres.append(np.array(overall_pres))
        all_recalls.append(np.array(overall_recalls))
        all_aucs.append(mean_auc)
        print(features[i])
        print("Overall precision: %.3f" % np.mean(np.array(overall_pres)))
        print("Overall recall: %.3f" % np.mean(np.array(overall_recalls)))
        print("Overall auc: %.3f" % mean_auc)
        print("*********************")

    plt.figure('Leave out precision/recall/auc')
    # ax = plt.gca()
    # x = list(range(len(features)))
    # plt.plot(x, np.array(all_pres).mean(axis=1), 'x--', label='leave-out precision')
    # plt.plot(x, [0.712 for i in range(len(features))], 'x--', label='auc')
    # plt.plot(x, np.array(all_recalls).mean(axis=1), '*--', label='leave-out recall')
    # plt.plot(x, [0.624 for i in range(len(features))], '*--', label='auc')
    plt.xticks(range(len(features)), features, rotation=90)
    plt.plot( all_aucs, '.--', label='leave-out auc')
    plt.plot([0.754 for i in range(len(features))], '.--', label='auc')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    # 使用 sklearn 的 feature_importance
    feature_importance = []
    for j in range(10):
        # 切分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=random_states[j])
        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(X_train, y_train)
        feature_importance.append(clf.feature_importances_)
        print(feature_importance)

    mean_importance = np.array(feature_importance).mean(axis=0)
    print("Feature importance: ")
    for i in range(len(mean_importance)):
        print(features[i]+": %.3f" % mean_importance[i])


if __name__ == '__main__':
    main()
