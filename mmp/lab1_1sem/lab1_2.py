import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

__author__ = 'andreykoloskov'

def main():
    data = pd.read_csv(sys.argv[1])

    print(data.head())
    print(data.shape)
    print(data.ACTION.mean())

    for col_name in data.columns:
        print(col_name, len(data[col_name].unique()))

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:],
                                                        data.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    #2.1
    print('#2.1')
    knn(X_train, X_test, y_train, y_test, 10, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 10, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 10, 'distance', 'chebyshev')

    #2.2
    print('#2.2')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'chebyshev')

    #2.3
    print('#2.3')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :],
                                                        data.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter(X_train, X_test)

    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'chebyshev')

    #folding
    print('#folding')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :],
                                                        data.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter_folder(X_train, X_test, y_train, y_test)

    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'chebyshev')

    #2.4
    print('#2.4')
    data_conc = add_features(data)

    X_train, X_test, y_train, y_test = train_test_split(data_conc.iloc[:, :],
                                                        data_conc.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter(X_train, X_test)

    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'chebyshev')

    #folding
    print('#folding')
    data_conc = add_features(data)

    X_train, X_test, y_train, y_test = train_test_split(data_conc.iloc[:, :],
                                                        data_conc.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter_folder(X_train, X_test, y_train, y_test)

    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'manhattan')
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'minkowski', 2)
    knn(X_train, X_test, y_train, y_test, 35, 'distance', 'chebyshev')

    #3.1
    print('#3.1')
    data_conc = add_features(data)

    X_train, X_test, y_train, y_test = train_test_split(data_conc.iloc[:, :],
                                                        data_conc.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter(X_train, X_test)

    cv = KFold(X_train.shape[0], shuffle=True, random_state=241)
    gs = GridSearchCV(DecisionTreeClassifier(random_state=241),
                      param_grid={'max_features': [None, 'log2', 'sqrt'],
                                  'max_depth': [2, 4, 6, 8, 10, 20, 50],
                                  'min_samples_leaf': [x for x in range(1, 20)]},
                      cv=cv)
    ft = gs.fit(X_train, y_train)
    for z in gs.grid_scores_:
        if z.mean_validation_score == gs.best_score_:
            ft = np.array(z)[0]['max_features']
            dp = np.array(z)[0]['max_depth']
            lf = np.array(z)[0]['min_samples_leaf']
            tree_classif(X_train, X_test, y_train, y_test, 241, ft, dp, lf)

    tree_classif(X_train, X_test, y_train, y_test, 100, 'sqrt', 20, 10)

    #3.2
    print('#3.2')
    data_conc = add_features(data)

    X_train, X_test, y_train, y_test = train_test_split(data_conc.iloc[:, :],
                                                        data_conc.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter(X_train, X_test)

    rnd_forest(X_train, X_test, y_train, y_test, 1000)

    #3.3
    print('#3.3')
    print('#folding')
    data_conc = add_features(data)

    X_train, X_test, y_train, y_test = train_test_split(data_conc.iloc[:, :],
                                                        data_conc.iloc[:, 0],
                                                        test_size=0.3,
                                                        random_state=241)

    X_train, X_test = create_counter_folder(X_train, X_test, y_train, y_test)

    rnd_forest(X_train, X_test, y_train, y_test, 1000)

def knn(X_train, X_test, y_train, y_test, n, w, m, r = -1):
    if r == -1:
        neigh = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m)
    else:
        neigh = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m, p=r)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    res_test = neigh.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)
    fpr, tpr, thresholds = roc_curve(y_test, res_test[:, 1])
    roc_auc  = auc(fpr, tpr)
    print('roc_auc = ', roc_auc, '( metric = ', m, ')')

def cnt_succ(X_train):
    cnt = []
    succ = []
    for col_name in X_train.columns:
        if col_name != 'ACTION':
            cnt.append(dict(np.array(X_train.groupby(col_name).count()
               .reset_index()[[col_name, 'ACTION']])))
            succ.append(dict(np.array(X_train[X_train['ACTION'] == 1]
                .groupby(col_name).count().reset_index()[[col_name, 'ACTION']])))
    return cnt, succ

def rewrite_arr(X_train, cnt, succ):
    X_train2 = []
    for row in np.array(X_train):
        arr = []
        for i, v in enumerate(row):
            c = cnt[i][v] if v in cnt[i] else 0
            arr.append(c)
            s = succ[i][v] if v in succ[i] else 0
            arr.append(s)
            arr.append((s + 1.0) / (c + 2.0))
        X_train2.append(arr)
    return np.array(X_train2)

def tree_classif(X_train, X_test, y_train, y_test, rs, ft, dp, lf):
    model = DecisionTreeClassifier(random_state=rs, max_features=ft,
                                   max_depth=dp, min_samples_leaf=lf)
    res_test = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, res_test[:, 1])
    roc_auc  = auc(fpr, tpr)
    print('treeClassifier roc_auc = ', roc_auc)

def rnd_forest(X_train, X_test, y_train, y_test, estim):
    model = RandomForestClassifier(n_estimators=estim)
    res_test = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, res_test[:, 1])
    roc_auc  = auc(fpr, tpr)
    print('randomForest roc_auc = ', roc_auc)

def add_features(data):
    data_conc = data.copy()
    for i, col_name in enumerate(data.iloc[:, 1:].columns):
        for j, col_name2 in enumerate(data.iloc[:, 1:].columns):
            if i < j:
                col = []
                for k, v in enumerate(data[col_name]):
                    col.append(int(str(data[col_name].iloc[k])
                                   + str(data[col_name2].iloc[k])))
                data_conc[data.columns[i + 1] + '_' + data.columns[j + 1]] = col
    return data_conc

def create_counter(X_train, X_test):
    cnt_tr, succ_tr = cnt_succ(X_train)
    X_train_c = X_train.iloc[:, 1:]
    X_test_c = X_test.iloc[:, 1:]
    X_train_c = rewrite_arr(X_train_c, cnt_tr, succ_tr)
    X_test_c = rewrite_arr(X_test_c, cnt_tr, succ_tr)
    return X_train_c, X_test_c

def create_counter_folder(X_train, X_test, y_train, y_test):
    X_train1 = X_train[: int(len(X_train) / 3 + 1)]
    X_train2 = X_train[int(len(X_train) / 3 + 1) : int(2 * len(X_train) / 3 + 1)]
    X_train3 = X_train[int(2 * len(X_train) / 3 + 1) : ]

    y_train1 = y_train[: int(len(y_train) / 3 + 1)]
    y_train2 = y_train[int(len(y_train) / 3 + 1) : int(2 * len(y_train) / 3 + 1)]
    y_train3 = y_train[int(2 * len(y_train) / 3 + 1) : ]

    X_train1_2 = X_train1.append(X_train2)
    X_train1_3 = X_train1.append(X_train3)
    X_train2_3 = X_train2.append(X_train3)

    y_train1_2 = y_train1.append(y_train2)
    y_train1_3 = y_train1.append(y_train3)
    y_train2_3 = y_train2.append(y_train3)

    cnt_tr1, succ_tr1 = cnt_succ(X_train2_3)
    X_train1 = X_train1.iloc[:, 1:]
    X_train1 = rewrite_arr(X_train1, cnt_tr1, succ_tr1)

    cnt_tr2, succ_tr2 = cnt_succ(X_train1_3)
    X_train2 = X_train2.iloc[:, 1:]
    X_train2 = rewrite_arr(X_train2, cnt_tr2, succ_tr2)

    cnt_tr3, succ_tr3 = cnt_succ(X_train1_2)
    X_train3 = X_train3.iloc[:, 1:]
    X_train3 = rewrite_arr(X_train3, cnt_tr3, succ_tr3)

    cnt_tr, succ_tr = cnt_succ(X_train)
    X_train4 = X_train.iloc[:, 1:]
    X_train4 = np.concatenate((X_train1, X_train2), axis=0)
    X_train4 = np.concatenate((X_train4, X_train3), axis=0)

    X_test4 = X_test.iloc[:, 1:]
    X_test4 = rewrite_arr(X_test4, cnt_tr, succ_tr)

    return X_train4, X_test4

if __name__ == '__main__':
    main()
