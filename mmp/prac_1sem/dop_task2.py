import numpy as np

'''
    Подсчитать в векторе x среднее значение, проигнорировав значения inf и nan.
    Т.е. для x = np.array([1, 2, np.nan]) ответ 1.5
'''

x = np.array([1, 2, np.nan, np.inf, 3])
l = np.isnan(x)
m = np.isinf(x)
x[l] = 0
x[m] = 0
res = x.sum() / (np.size(x) - np.sum(np.logical_or(l, m)))
print(res)
