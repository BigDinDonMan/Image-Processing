import skimage.morphology as morph
from skimage import io
from scipy import ndimage
from skimage.color import rgb2gray as czary_mary_op_jest_szary
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from skimage import transform

#havent actually managed to do it, kept getting like 1 or 2 too many or too less rectangles

grey_img = czary_mary_op_jest_szary(io.imread('zad32.png'))
mask = czary_mary_op_jest_szary(io.imread('m.png'))


res = ndimage.morphology.binary_hit_or_miss(grey_img, structure1 = mask)

count = len(np.nonzero(res)[0])
print(count)
plt.imshow(res, cmap = 'gray')
plt.show()
'''plt.imshow(res, cmap='gray')
plt.show()'''

