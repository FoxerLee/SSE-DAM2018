import matplotlib.pyplot as plt
import numpy as np


def figure(pres, recalls, aucs):
    # ax = plt.gca()
    # pres = [0.810, 0.655, 0.655, 0.712, 0.721, 0.717, 0.728]
    # racalls = [0.426, 0.644, 0.633, 0.624, 0.668, 0.672, 0.686]
    # aucs = [0.769, 0.688, 0.648, 0.754, 0.783, 0.775, 0.789]
    plt.xlabel('Classifier')
    # plt.ylabel('Classifier')
    total_width, n = 0.9, 3
    width = total_width / n
    labels = ['Gaussian', 'KNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging', 'GradientBoosting']
    x = np.arange(len(labels))
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.plot(pres, 'x--', label='precision')
    for a, b in zip(x, pres):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(recalls, '.--', label='recall')
    for a, b in zip(x, recalls):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.plot(aucs, '*--', label='auc')
    for a, b in zip(x, aucs):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)

    plt.ylim([0.0, 1.0])

    plt.legend()
    plt.show()
