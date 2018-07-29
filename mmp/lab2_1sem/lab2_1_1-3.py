import numpy as np
import sys

from matplotlib import pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator

__author__ = 'andreykoloskov'

def main():
    students = pd.read_csv(sys.argv[1]);

    res = np.array(list(map(lambda a: find_best_split(np.array(students[a]),
                   np.array(students[students.columns[-1]])),
                   students.columns[:-1])))

    f_plot(res[:, 0], res[:, 1], labels = students.columns[:-1])
    plt.title('gini')
    plt.show()

    for column in students.columns[:-1]:
        plt.scatter(students[column], students[students.columns[-1]])
        plt.title(column)
        plt.grid()
        plt.show()

    print("feature | max_thresholds | max_gini")
    for a in list(zip(students.columns[:-1], res[:, 2], res[:, 3])):
        print("{:s}                 {:02.2f}      {:02.2f}"
              .format(a[0], a[1], a[2]))

def find_best_split(feature_vector, target_vector):
    thresholds = np.array(list(set(feature_vector)))
    if len(thresholds) <= 1:
        return [], [], -1, -1
    thresholds.sort()
    thresholds = thresholds[:-1]
    feature_target = pd.DataFrame(np.vstack((feature_vector, target_vector)).T)
    df = pd.DataFrame(feature_target)
    ginis = np.array(list(map(lambda v: gini(feature_target, v), thresholds)))
    thresholds_ginis = pd.DataFrame(np.vstack((thresholds, ginis)).T)
    max_gini = thresholds_ginis[1].max()
    max_thresholds = np.array(thresholds_ginis[thresholds_ginis[1]
                              == max_gini])[0][0]

    return thresholds, ginis, max_thresholds, max_gini

def gini(feature_target, value):
    r = len(feature_target)
    zl = feature_target[feature_target[0] <= value]
    rl = len(zl)
    rl0 = len(zl[zl[1] == 0])
    rl1 = len(zl[zl[1] == 1])
    hrl = 1 - (rl0 / rl) ** 2 - (rl1 / rl) ** 2

    zr = feature_target[feature_target[0] > value]
    rr = len(zr)
    rr0 = len(zr[zr[1] == 0])
    rr1 = len(zr[zr[1] == 1])
    hrr = 1 - (rr0 / rr) ** 2 - (rr1 / rr) ** 2

    qr = -1 * (rl / r) * hrl - (rr / r) * hrr

    return qr

def f_plot(xarray, yarray, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for x, y, label in zip(xarray, yarray, labels):
        ax.plot(x, y, label = label)

    ax.grid(True)
    ax.legend()

if __name__ == '__main__':
    main()
