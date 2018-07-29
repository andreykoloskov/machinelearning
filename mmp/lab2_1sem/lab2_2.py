import numpy as np
import sys
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

__author__ = 'andreykoloskov'

def main():
    winequality = np.array(pd.read_csv(sys.argv[1], sep = ';'))

    accur = cross_val_score(DecisionTreeClassifier(random_state=0),
                winequality[:, :-1], list(winequality[:, -1]), cv = 3,
                scoring = make_scorer(accuracy_score))
    print("DecisionTreeClassifier = ", accur)

    accur = cross_val_score(BaggingClassifier(random_state=0, n_estimators=100),
                winequality[:, :-1], list(winequality[:, -1]), cv = 3,
                scoring = make_scorer(accuracy_score))
    print("BaggingClassifier = ", accur)

    accur = cross_val_score(BaggingClassifier(random_state=0, n_estimators=100,
                            max_features=0.5),
                winequality[:, :-1], list(winequality[:, -1]), cv = 3,
                scoring = make_scorer(accuracy_score))
    print("BaggingClassifier max_samples(0.5) = ", accur)

    accur = cross_val_score(RandomForestClassifier(random_state=0,
                            n_estimators=100),
                winequality[:, :-1], list(winequality[:, -1]), cv = 3,
                scoring = make_scorer(accuracy_score))
    print("RandomForestClassifier = ", accur)

if __name__ == '__main__':
    main()
