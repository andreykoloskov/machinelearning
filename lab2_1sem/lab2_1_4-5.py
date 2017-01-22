import numpy as np
import sys

from matplotlib import pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

__author__ = 'andreykoloskov'

def main():
    columns = [str(x) for x in range(23)]
    mushrooms = np.array(pd.read_csv(sys.argv[1], header = None,
                                     names = columns))
    for x in range(mushrooms.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(mushrooms[:, x])
        mushrooms[:, x] = le.transform(mushrooms[:, x])

    X_train, X_test, y_train, y_test = train_test_split(mushrooms[:, 1:],
                                                        mushrooms[:, 0],
                                                        test_size=0.5,
                                                        random_state=241)
    #types = np.zeros(mushrooms.shape[1])
    types = np.ones(mushrooms.shape[1])
    tree = DecisionTree(types)
    tree.fit(X_train, y_train)
    predicted = tree.predict(X_test)
    accur = accuracy_score(list(y_test), list(predicted))
    print("accuracy = ", accur)

class DecisionTree(BaseEstimator):
    def __init__(self, types):
        self._tree = []
        self.types = types

    def fit(self, X, y):
        def fit_node(subX, suby, node):
            if len(suby) == 0:
                return

            if len(set(suby)) == 1:
                node.append(suby[0])
                return

            max_gini = -1
            max_threshold = -1
            j = -1
            left_features = None
            for i in range(subX.shape[1]):
                feature_vector = None
                ind = None

                if self.types[i] == 0:
                    feature_vector = subX[:, i]
                else:
                    ind = np.array(pd.DataFrame(subX[:, i]).groupby(0).size()
                                   .sort_values().index)

                    num = [i for i, a in enumerate(ind)]
                    dct = dict(zip(ind, num))
                    dct2 = dict(zip(num, ind))
                    feature_vector = np.array([dct[x] for x in subX[:, i]])

                thresholds_array, ginis_array, threshold, gini = \
                    find_best_split(feature_vector, suby)
                if gini > max_gini:
                    max_gini = gini
                    max_threshold = threshold
                    j = i
                    if self.types[j] == 1:
                        left_features = set([dct2[x] \
                                for x in range(int(max_threshold) + 1)])

            node.append(j)
            if self.types[j] == 0:
                node.append(max_threshold)
            else:
                node.append(left_features)

            nodel, noder, subXl, subyl, subXr, subyr = [], [], [], [], [], []

            if self.types[j] == 0:
                for i, x in enumerate(subX[:, j]):
                    if x <= max_threshold:
                        subXl.append(subX[i])
                        subyl.append(suby[i])
                    else:
                        subXr.append(subX[i])
                        subyr.append(suby[i])
            elif left_features != None:
                for i, x in enumerate(subX[:, j]):
                    if x in left_features:
                        subXl.append(subX[i])
                        subyl.append(suby[i])
                    else:
                        subXr.append(subX[i])
                        subyr.append(suby[i])

            fit_node(np.array(subXl), np.array(subyl), nodel)
            fit_node(np.array(subXr), np.array(subyr), noder)

            node.append(nodel)
            node.append(noder)

        fit_node(X, y, self._tree)

    def predict(self, X):
        def predict_node(x, node):
            res = 0
            if len(node) == 1:
                return node[0]

            if self.types[node[0]] == 0:
                if x[node[0]] <= node[1]:
                    res = predict_node(x, node[2])
                else:
                    res = predict_node(x, node[3])
            else:
                if x[node[0]] in node[1]:
                    res = predict_node(x, node[2])
                else:
                    res = predict_node(x, node[3])

            return res

        predicted = []
        for x in X:
            predicted.append(predict_node(x, self._tree))
        return np.array(predicted)

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

if __name__ == '__main__':
    main()
