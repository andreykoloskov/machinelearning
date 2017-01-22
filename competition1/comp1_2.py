import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

__author__ = 'andreykoloskov'

def main():
    data = pd.read_csv(sys.argv[1])

    del data['lobby_type']
    del data['start_time']

    X_pick = np.zeros((data.shape[0], 113))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    for p in range(5):
        del data['r%d_hero' % (p+1)]
        del data['d%d_hero' % (p+1)]

    for i in range(len(X_pick[0])):
        data.insert(1, 'hero_%d' % i, X_pick[:, i])

    X_data = data.iloc[:, :-6]
    y_data = data.iloc[:, -5]


    data_test = pd.read_csv(sys.argv[2])

    del data_test['lobby_type']
    del data_test['start_time']

    X_pick = np.zeros((data_test.shape[0], 113))

    for i, match_id in enumerate(data_test.index):
        for p in range(5):
            X_pick[i, data_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    for i, match_id in enumerate(data_test.index):
        for p in range(5):
            X_pick[i, data_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    for p in range(5):
        del data_test['r%d_hero' % (p+1)]
        del data_test['d%d_hero' % (p+1)]

    for i in range(len(X_pick[0])):
        data_test.insert(1, 'hero_%d' % i, X_pick[:, i])


    log_regr = LogisticRegression()

    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data)
    data_test_scaled = scaler.transform(data_test)

    res_test = log_regr.fit(X_data_scaled, y_data) \
            .predict_proba(data_test_scaled)

    tmp = dict([('match_id', np.array(data_test['match_id'])),
               ('radiant_win', res_test[:, 1])])
    res = pd.DataFrame(tmp).set_index('match_id')
    res.to_csv(sys.argv[3])


if __name__ == '__main__':
    main()
