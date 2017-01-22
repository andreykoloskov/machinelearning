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

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-6],
                                                        data.iloc[:, -5],
                                                        test_size=0.3,
                                                        random_state=241)

    log_regr = LogisticRegression()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    res_test = log_regr.fit(X_train_scaled, y_train) \
            .predict_proba(X_test_scaled)

    tmp = dict([('match_id', np.array(X_test['match_id'])),
               ('radiant_win', res_test[:, 1])])
    res = pd.DataFrame(tmp).set_index('match_id')
    res.to_csv(sys.argv[2])

    log_loss_res = log_loss(y_test, res_test[:, 1])
    print('log_loss = ', log_loss_res)

if __name__ == '__main__':
    main()
