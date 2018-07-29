import numpy as np

'''
Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество.
Для x = np.array([1, 2, 2, 4]), y = np.array([4, 2, 1, 2]) ответ True.
'''

x = np.array([1, 2, 2, 4])
y = np.array([4, 2, 1, 2])

print(x)
print(y)

x.sort()
y.sort()

print(x)
print(y)

print(sum(x == y) == x.size == y.size)


