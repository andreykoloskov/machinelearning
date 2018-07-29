import numpy as np

'''
Найти максимальный элемент в векторе x среди элементов,
перед которыми стоит нулевой.
Для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответ 5.
'''
x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
mx = np.array([x[i + 1] for i, a in \
        enumerate(x) if a == 0 and i < x.size - 1 and x[i + 1] != 0]).max()

print(x)
print(mx)
