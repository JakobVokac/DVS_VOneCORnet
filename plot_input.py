import numpy as np
from PIL import Image, ImageShow
import os
from sklearn.preprocessing import normalize

img_path = "d:/Datasets/cifar10dvs/train/airplane/cifar10_airplane_0.npz"

vox = np.load(img_path)["arr_0"][0]

print(vox.shape)
max = vox.max()
min = vox.min()

def func(x,max,min):
    return int(np.floor(((x - min)/(max-min))*255))

for i in range(128):
    for j in range(128):
        vox[i,j] = func(vox[i,j],max,min)

print(vox.max())
print(vox.min())


im = Image.fromarray(vox,mode='L')
im.show()