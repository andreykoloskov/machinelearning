import numpy as np
from matplotlib import pyplot

'''
    integral cos(x^2) from 0 to 0.5
    method montecarlo
'''

X = np.random.rand(10000, 2)
X[:, 0] = X[:, 0] / 2
print(X)

inside = X[:, 1] <= np.cos(X[:, 0] ** 2)
print(inside)

sq = 1 * 0.5
res = np.cumsum(inside) / np.arange(1, X.shape[0] + 1) * sq
print(res)

pyplot.plot(res)
pyplot.show()
