from skimage import filters, feature, io
import cv2 as cv
import numpy as np
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import invert

image = cv.imread('gears.bmp', 0)

threshed_image = cv.threshold(image, 160, 255, cv.THRESH_BINARY_INV)[1]

canny = feature.canny(threshed_image, sigma = 1)
sk_filters = [filters.sobel(threshed_image), filters.prewitt(threshed_image)]
h = filters.apply_hysteresis_threshold(image, 130, 180)
h = invert(h)

filled = [canny] + sk_filters + [h]
_filled = [nd.binary_fill_holes(x) for x in filled]

stack1 = np.hstack((tuple(filled)))
stack2 = np.hstack((tuple(_filled)))

plt.imshow(stack1, cmap = 'gray')
plt.show()
plt.imshow(stack2, cmap = 'gray')
plt.show()

