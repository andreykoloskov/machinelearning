import numpy as np

'''
Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы.
Для X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]) ответ 3.
'''

X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])
d = np.diag(X)
p = np.nonzero(d)
p = d[p]

y = 1
for x in p:
    y *= x

print(y)
