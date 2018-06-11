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
from imblearn.combine import SMOTEENN

import util
import cii


def main():
    print("Generator train data!")
    months = ['02', '03', '04']
    overall = '234'
    label_month = '05'
    train_datas = cii.train_generator(months, overall, label_month)
    # train_datas.to_csv('X.csv')
    train_datas.set_index(['vipno', 'bndno'], inplace=True, drop=True)
    train = train_datas.as_matrix()
    X_train = np.delete(train, train.shape[1] - 1, axis=1)
    y_train = train[:, train.shape[1] - 1]

    print("Generator test data!")
    months = ['05', '06', '07']
    overall = '567'
    label_month = '08'
    test_datas = cii.train_generator(months, overall, label_month, predict=True)
    y_test = test_datas.index.tolist()
    X_test = test_datas.drop(columns=['vipno', 'bndno'], inplace=False).as_matrix()

    smote_enn = SMOTEENN(random_state=0)
    X_train, y_train = smote_enn.fit_sample(X_train, y_train)

    # 高斯朴素贝叶斯分类器
    f = open('predict/cii/1552674_2cii_GaussianNB.txt', 'w')
    clf = GaussianNB()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0])+','+str(i[1])+',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # K近邻分类器
    f = open('predict/cii/1552674_2cii_KNeighborsClassifier.txt', 'w')
    clf = KNeighborsClassifier()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # 决策树分类器
    f = open('predict/cii/1552674_2cii_DecisionTreeClassifier.txt', 'w')
    clf = DecisionTreeClassifier()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # 随机森林
    f = open('predict/cii/1552674_2cii_RandomForestClassifier.txt', 'w')
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # AdaBoost
    f = open('predict/cii/1552674_2cii_AdaBoostClassifier.txt', 'w')
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, algorithm='SAMME.R')
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # Bagging
    f = open('predict/cii/1552674_2cii_BaggingClassifier.txt', 'w')
    clf = BaggingClassifier(n_estimators=20)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()

    # GradientBoosting
    f = open('predict/cii/1552674_2cii_GradientBoostingClassifier.txt', 'w')
    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_features=19, subsample=0.85,
                                     max_depth=3, min_samples_split=300)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    for (i, j) in zip(y_test, y_pred):
        if j == 1:
            f.write(str(i[0]) + ',' + str(i[1]) + ',Yes\n')
        else:
            f.write(str(i[0]) + ',' + str(i[1]) + ',No\n')
    f.close()


if __name__ == '__main__':
    main()


