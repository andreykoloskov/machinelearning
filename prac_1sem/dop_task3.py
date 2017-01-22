import numpy as np

'''
    В матрице H заменить все значения, которые больше maxH, на maxH, а все
    значения, которые меньше minH, на minH. Решите задачу двумя способами:
    с использованием индексации по матрице, и с использованием операций взятия
    максимума и минимума
'''
H = np.random.rand(10, 5)
minH = 0.2
maxH = 0.8

print(H)
print("minH = ", minH, " ", "maxH = ", maxH, "\n")

#1
H_res1 = H.reshape(H.shape[0] * H.shape[1])
H_res1 = np.array(list(map(lambda x: \
        maxH if x > maxH else minH if x < minH else x, H_res1))) \
        .reshape(H.shape)
print(H_res1, '\n')

#2
H_res2 = np.minimum(np.maximum(H, minH), maxH)
print(H_res2)
