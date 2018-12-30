from skimage import io, color
from skimage import segmentation as seg
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

img = cv.imread('apples.jpg')

copied_img = img.copy()

grey = cv.cvtColor(copied_img, cv.COLOR_BGR2GRAY)

res_g = copied_img[:,:,1].copy()

apple_thresh = cv.threshold(res_g, 20, 150, cv.THRESH_BINARY_INV)[1]
outside_thresh = cv.threshold(res_g, 250, 150, cv.THRESH_TOZERO)[1]

outside_thresh[outside_thresh >= 200] = 100
outside_thresh += apple_thresh

random_walker_res = seg.random_walker(grey, outside_thresh, beta = 1000, mode = 'bf')

plt.imshow(random_walker_res, cmap='gray')
plt.show()

