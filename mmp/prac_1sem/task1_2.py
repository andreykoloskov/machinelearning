import numpy as np

'''
Дана матрица X и два вектора одинаковой длины i и j.
Построить вектор
np.array([X[i[0], j[0]], X[i[1], j[1]], ..., X[i[N-1], j[N-1]]]).
'''

a = 4
X = np.random.random(a * a).reshape(a, a)
print(X)

i = np.random.randint(0, a, a)
j = np.random.randint(0, a, a)

print(i)
print(j)

print(X[i, j])
