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
    train = train_datas.as_matrix()
    X = np.delete(train, train.shape[1] - 1, axis=1)
    y = train[:, train.shape[1] - 1]
    # 用于做降采样，以确保正负样本的数量相近
    rus = RandomUnderSampler(return_indices=True)
    X, y, idx_resampled = rus.fit_sample(X, y)

    param_test1 = {'n_estimators': range(10, 61, 10),
                   'learning_rate': np.arange(0.01, 0.1, 10)}
    param_test2 = {'max_depth': range(3, 14, 2)}
    param_test3 = {'max_features': range(7, 20, 2),
                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.01, n_estimators=60,
                                                                 max_depth=3, min_samples_split=300),
                            param_grid=param_test2, scoring='precision', cv=5)

    gsearch1.fit(X, y)

    print("Best param: {}".format(gsearch1.best_params_))
    print("Best score: {}".format(gsearch1.best_score_))
    print("Best estimator: {}".format(gsearch1.best_estimator_))

    print("****************************")


if __name__ == '__main__':
    main()