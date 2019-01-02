from skimage import morphology as morph
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray as czary_mary_op_już_szary

_images = [(255 - io.imread('horse.png')), (255 - io.imread('psio.png')), (255 - io.imread('bw.png'))]
images = [i.astype(np.bool) for i in _images]
grey = [czary_mary_op_już_szary(i) for i in images]
sp00ky_scary_skeletons = [morph.skeletonize(i) for i in grey]
medial = [morph.medial_axis(i) for i in [czary_mary_op_już_szary(j) for j in _images]]

for i in sp00ky_scary_skeletons:
    plt.imshow(i, cmap='gray')
    plt.show()
for i in medial:
    plt.imshow(i, cmap='gray')
    plt.show()

for c, i in enumerate(sp00ky_scary_skeletons, start = 1):
    io.imsave('skeletonized' + str(c) + '.png', i.astype(np.uint8) * 255)

for c, i in enumerate(medial, start = 1):
    io.imsave('medial_axis' + str(c) + '.png', i.astype(np.uint8) * 255)
