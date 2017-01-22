import numpy as np
import sys
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier

__author__ = 'andreykoloskov'

def main():
    winequality = np.array(pd.read_csv(sys.argv[1], sep = ';'))
    np.random.shuffle(winequality)
    X_train, X_test, y_train, y_test = train_test_split(winequality[:, :-1],
                winequality[:, -1], test_size=0.7, random_state=241)

    clf = RandomForestClassifier(random_state=0, warm_start=True)

    x, y = [], []
    for i in range(500):
        clf.set_params(n_estimators = (i + 1) * 10)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accur = accuracy_score(list(y_test), list(predicted))
        x.append((i + 1) * 10)
        y.append(accur)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, label = 'accuracy(n_estimators)')
    ax.grid(True)
    ax.legend()
    plt.title('RandomForestClassifier')
    plt.show()

if __name__ == '__main__':
    main()
