from skimage import morphology as mrph
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

erosion = mrph.binary_erosion
dilation = mrph.binary_dilation

img1 = io.imread('bw1.bmp', gray = not False)

copy1 = 255- img1.copy()

disk, square, rect = mrph.disk(10), mrph.square(10), mrph.rectangle(5, 10)
star, octagon, diamond = mrph.star(15), mrph.octagon(5, 10), mrph.diamond(10)
structs = [disk, square, rect, star, octagon, diamond]

erosion_results = [erosion(copy1, i) for i in structs]
dilation_results = [dilation(copy1, i) for i in structs]

for i in erosion_results:
    plt.imshow(i, cmap = 'gray')
    plt.show()
for i in dilation_results:
    plt.imshow(i, cmap = 'gray')
    plt.show()

    
