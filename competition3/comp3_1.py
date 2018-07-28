import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier

def main():
    x = pd.read_csv(sys.argv[1], sep = ';')
    del x['doReturnOnLowerLevels']
    del x['averageNumOfTurnsPerCompletedLevel']
    x = np.array(x)
    y = np.array(pd.read_csv(sys.argv[2], header = None))[:, 0]

    '''
    x = np.column_stack((x, x[:, -1] * x[:, 0]))
    x = np.column_stack((x, x[:, -2] * x[:, 1]))
    x = np.column_stack((x, x[:, -3] * x[:, 3]))
    x = np.column_stack((x, x[:, -4] * x[:, -4]))
    x = np.column_stack((x, x[:, -5] * x[:, -5] * x[:, -5]))
    x = np.column_stack((x, x[:, -6] * x[:, -6] * x[:, -6] * x[:, -6]))
    x = np.column_stack((x, x[:, -7] * x[:, 0] * x[:, 1]))
    x = np.column_stack((x, x[:, -8] * x[:, 0] * x[:, 1] * x[:, 3]))
    x = np.column_stack((x, x[:, -10] * x[:, -11]))
    x = np.column_stack((x, x[:, -10] * x[:, -11]))
    '''

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
            random_state=241)

    #clf = LogisticRegression()
    clf = GradientBoostingClassifier(loss='exponential',min_samples_split=100, min_samples_leaf=100)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf.fit(X_train_scaled, y_train)
    clf_probs = clf.predict_proba(X_test_scaled)
    score = log_loss(y_test, clf_probs)

    print score

    ####################################################3
    train = pd.read_csv(sys.argv[1], sep = ';')
    del train['doReturnOnLowerLevels']
    del train['averageNumOfTurnsPerCompletedLevel']
    train = np.array(train)

    '''
    train = np.column_stack((train, train[:, -1] * train[:, 0]))
    train = np.column_stack((train, train[:, -2] * train[:, 1]))
    train = np.column_stack((train, train[:, -3] * train[:, 3]))
    train = np.column_stack((train, train[:, -4] * train[:, -4]))
    train = np.column_stack((train, train[:, -5] * train[:, -5] * train[:, -5]))
    train = np.column_stack((train, train[:, -6] * train[:, -6] * train[:, -6] * train[:, -6]))
    train = np.column_stack((train, train[:, -7] * train[:, 0] * train[:, 1]))
    train = np.column_stack((train, train[:, -8] * train[:, 0] * train[:, 1] * train[:, 3]))
    train = np.column_stack((train, train[:, -10] * train[:, -11]))
    train = np.column_stack((train, train[:, -10] * train[:, -11]))
    '''

    y2 = np.array(pd.read_csv(sys.argv[2], header = None))[:, 0]

    test = pd.read_csv(sys.argv[3], sep = ';')
    del test['doReturnOnLowerLevels']
    del test['averageNumOfTurnsPerCompletedLevel']
    test = np.array(test)

    '''
    test = np.column_stack((test, test[:, -1] * test[:, 0]))
    test = np.column_stack((test, test[:, -2] * test[:, 1]))
    test = np.column_stack((test, test[:, -3] * test[:, 3]))
    test = np.column_stack((test, test[:, -4] * test[:, -4]))
    test = np.column_stack((test, test[:, -5] * test[:, -5] * test[:, -5]))
    test = np.column_stack((test, test[:, -6] * test[:, -6] * test[:, -6] * test[:, -6]))
    test = np.column_stack((test, test[:, -7] * test[:, 0] * test[:, 1]))
    test = np.column_stack((test, test[:, -8] * test[:, 0] * test[:, 1] * test[:, 3]))
    test = np.column_stack((test, test[:, -10] * test[:, -11]))
    test = np.column_stack((test, test[:, -10] * test[:, -11]))
    '''

    #clf2 = LogisticRegression()
    clf2 = GradientBoostingClassifier(loss='exponential',min_samples_split=100, min_samples_leaf=100)

    scaler2 = StandardScaler()
    train_scaled = scaler2.fit_transform(train)
    test_scaled = scaler2.transform(test)

    clf2.fit(train_scaled, y2)
    clf_probs2 = clf2.predict_proba(test_scaled)

    clf_probs2[:, 1].tofile(sys.argv[4], sep="\n", format='%10.8f')

if __name__ == '__main__':
    main()
