import numpy as np
from scipy import misc

'''
Дан трёхмерный массив, содержащий изображение, размера (height, width,
numChannels), а также вектор длины numChannels. Сложить каналы изображения
с указанными весами, и вернуть результат в виде матрицы размера
(height, width). Считать реальное изображение можно при помощи функции
scipy.misc.imread (если изображение не в формате png, установите пакет pillow:
conda install pillow). Преобразуйте цветное изображение в оттенки серого,
использовав коэффициенты np.array([0.299, 0.587, 0.114]).
'''

img = misc.imread('image1.jpg')

#misc.imsave('image_out.jpg', img)
#img2 = np.sum(img, axis=2)

#print(img)
#print(img2)

#img2 = img[:,:,0] + img[:,:,1] + img[:,:,2]

#print(img2)

misc.imshow(img)
misc.imshow(img2)

print(img.shape)
print(img[:,:,0].T)
