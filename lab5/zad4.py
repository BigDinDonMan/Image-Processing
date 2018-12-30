import skimage.morphology as morph
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray as czary_mary_op_jest_szary
import skimage.filters
import cv2 as cv
from skimage.util import invert

teeth_img = cv.imread('teeth.jpg', 0)
grey = teeth_img.copy()
threshed_teeth = cv.threshold(grey, 140, 170, cv.THRESH_BINARY_INV)[1]

eroded = morph.binary_erosion(threshed_teeth)

for i in range(5):
    eroded = morph.binary_dilation(eroded, morph.disk(3 + i))
eroded = invert(eroded)
plt.imshow(eroded, cmap='gray')
plt.show()

