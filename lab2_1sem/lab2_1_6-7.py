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
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

__author__ = 'andreykoloskov'

def main():
    dct = {}

    #mashrooms
    mushrooms = np.array(pd.read_csv(sys.argv[1], header = None,
                                     names = [str(x) for x in range(23)]))
    for x in range(mushrooms.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(mushrooms[:, x])
        mushrooms[:, x] = le.transform(mushrooms[:, x])

    accur_m_r = cross_val_score(DecisionTree(np.zeros(mushrooms.shape[1])),
                mushrooms[:, 1:], list(mushrooms[:, 0]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real'] = accur_m_r

    accur_m_c = cross_val_score(DecisionTree(np.ones(mushrooms.shape[1])),
                mushrooms[:, 1:], list(mushrooms[:, 0]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['categor'] = accur_m_c

    data = OneHotEncoder(sparse=False).fit(mushrooms[:, 1:]) \
                                      .transform(mushrooms[:, 1:])
    accur_m_o = cross_val_score(DecisionTree(np.zeros(data.shape[1])),
                data, list(mushrooms[:, 0]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real_one-hot-encoding'] = accur_m_o

    accur_m_d = cross_val_score(DecisionTreeClassifier(random_state=0),
                mushrooms[:, 1:], list(mushrooms[:, 0]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['DecisionTreeClassifier'] = accur_m_d

    #tic_tac
    tic_tacs = np.array(pd.read_csv(sys.argv[2], header = None,
                                    names = [str(x) for x in range(10)]))
    for x in range(tic_tacs.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(tic_tacs[:, x])
        tic_tacs[:, x] = le.transform(tic_tacs[:, x])

    accur_t_r = cross_val_score(DecisionTree(np.zeros(tic_tacs.shape[1])),
                tic_tacs[:, :-1], list(tic_tacs[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real'] = np.append(dct['real'], accur_t_r)

    accur_t_c = cross_val_score(DecisionTree(np.ones(tic_tacs.shape[1])),
                tic_tacs[:, :-1], list(tic_tacs[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['categor'] = np.append(dct['categor'], accur_t_c)

    data = OneHotEncoder(sparse=False).fit(tic_tacs[:, :-1]) \
                                      .transform(tic_tacs[:, :-1])
    accur_t_o = cross_val_score(DecisionTree(np.zeros(data.shape[1])),
                data, list(tic_tacs[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real_one-hot-encoding'] = np.append(dct['real_one-hot-encoding'],
                                             accur_t_o)

    accur_t_d = cross_val_score(DecisionTreeClassifier(random_state=0),
                tic_tacs[:, :-1], list(tic_tacs[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['DecisionTreeClassifier'] = np.append(dct['DecisionTreeClassifier'],
                                              accur_t_d)

    #cars
    cars = pd.read_csv(sys.argv[3], header = None,
                       names = [str(x) for x in range(7)])
    cars['6'] = cars['6'].replace('unacc', 'acc')
    cars['6'] = cars['6'].replace('vgood', 'good')
    cars = np.array(cars)

    for x in range(cars.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(cars[:, x])
        cars[:, x] = le.transform(cars[:, x])

    accur_c_r = cross_val_score(DecisionTree(np.zeros(cars.shape[1])),
                cars[:, :-1], list(cars[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real'] = np.append(dct['real'], accur_c_r)

    accur_c_c = cross_val_score(DecisionTree(np.ones(cars.shape[1])),
                cars[:, :-1], list(cars[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['categor'] = np.append(dct['categor'], accur_c_c)

    data = OneHotEncoder(sparse=False).fit(cars[:, :-1]) \
                                      .transform(cars[:, :-1])
    accur_c_o = cross_val_score(DecisionTree(np.zeros(data.shape[1])),
                data, list(cars[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real_one-hot-encoding'] = np.append(dct['real_one-hot-encoding'],
                                             accur_c_o)

    accur_c_d = cross_val_score(DecisionTreeClassifier(random_state=0),
                cars[:, :-1], list(cars[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['DecisionTreeClassifier'] = np.append(dct['DecisionTreeClassifier'],
                                              accur_c_d)

    #nursery
    nursery = pd.read_csv(sys.argv[4], header = None,
                          names = [str(x) for x in range(9)])

    nursery['8'] = nursery['8'].replace('not_recom', '0')
    nursery['8'] = nursery['8'].replace('recom', '0')
    nursery['8'] = nursery['8'].replace('recommend', '0')
    nursery['8'] = nursery['8'].replace('very_recom', '1')
    nursery['8'] = nursery['8'].replace('priority', '1')
    nursery['8'] = nursery['8'].replace('spec_prior', '1')
    nursery = np.array(nursery)

    for x in range(nursery.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(nursery[:, x])
        nursery[:, x] = le.transform(nursery[:, x])

    accur_n_r = cross_val_score(DecisionTree(np.zeros(nursery.shape[1])),
                nursery[:, :-1], list(nursery[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real'] = np.append(dct['real'], accur_n_r)

    accur_n_c = cross_val_score(DecisionTree(np.ones(nursery.shape[1])),
                nursery[:, :-1], list(nursery[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['categor'] = np.append(dct['categor'], accur_n_c)

    data = OneHotEncoder(sparse=False).fit(nursery[:, :-1]) \
                                      .transform(nursery[:, :-1])
    accur_n_o = cross_val_score(DecisionTree(np.zeros(data.shape[1])),
                data, list(nursery[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['real_one-hot-encoding'] = np.append(dct['real_one-hot-encoding'],
                                             accur_n_o)

    accur_n_d = cross_val_score(DecisionTreeClassifier(random_state=0),
                nursery[:, :-1], list(nursery[:, -1]), cv = 10,
                scoring = make_scorer(accuracy_score))
    dct['DecisionTreeClassifier'] = np.append(dct['DecisionTreeClassifier'],
                                              accur_n_d)

    indexes = ['mushrooms_' + str(i) for i in range(10)]
    indexes.extend(['tic-tac-toe_' + str(i) for i in range(10)])
    indexes.extend(['cars_' + str(i) for i in range(10)])
    indexes.extend(['nursery_' + str(i) for i in range(10)])
    accure = pd.DataFrame(dct, index = indexes)
    print(accure)

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
