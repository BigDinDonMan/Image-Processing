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
#hysteresis = [filters.apply_hysteresis_threshold(im, 0.05, 0.35) for im in sk_filters]
h = filters.apply_hysteresis_threshold(image, 130, 180)
h = invert(h)

filled = [canny] + sk_filters + [h]# + hysteresis
_filled = [nd.binary_fill_holes(x) for x in filled]

stack1 = np.hstack((tuple(filled)))
stack2 = np.hstack((tuple(_filled)))

plt.imshow(stack1, cmap = 'gray')
plt.show()
plt.imshow(stack2, cmap = 'gray')
plt.show()

'''plt.subplot(2, 1, 1)
plt.imshow(stack1, cmap = 'gray')
plt.subplot(2, 1, 2)
plt.imshow(stack2, cmap = 'gray')
plt.show()'''


#najlepszy wynik daje oczywiście algorytm canny'ego, sobel i prewitt muszą zostać jeszcze poddane histerezie
#histereza zapobiega "dzieleniu się" krawędzi (jeśli użyjemy wypełniania dziur przed histerezą to algorytm "rozleje" wypełnienie po większości obrazu bo nie jest ograniczany)

