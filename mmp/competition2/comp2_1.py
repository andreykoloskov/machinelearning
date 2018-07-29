import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

def main():
    data = np.array(pd.read_csv(sys.argv[1], sep = ';').dropna())

    for x in [1, 3, 4, 9, 10]:
        le = preprocessing.LabelEncoder()
        le.fit(data[:, x])
        data[:, x] = le.transform(data[:, x])

    for k in [5, 8]:
        for i, x in enumerate(data[: , k]):
            data[i, k] = float(x.replace(',', '.'))

    for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        mx = max(data[:, x])
        if (mx != 0.0):
            data[:, x] = (data[:, x] * 1.0) / mx

    '''
    for i, x in enumerate(data[:, -1]):
        if (data[i, -1] == 0):
            data[i, -1] = -1
        else:
            data[i, -1] = 1
    '''
    #data[:, -1] = data[:, -1] + 1.0
    print np.unique(data[:, -1])

    X_train, X_test, y_train, y_test = train_test_split(data[:, 1:-2],
            data[:, -1], test_size=0.3, random_state=241)

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #clf = RandomForestClassifier(n_estimators=25)
    clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, clf_probs[:, 1])
    roc_auc  = auc(fpr, tpr)
    print roc_auc

if __name__ == '__main__':
    main()
