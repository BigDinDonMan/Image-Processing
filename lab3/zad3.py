from skimage import segmentation, io
from skimage import feature
from skimage import filters
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color

oh_hi_mark = io.imread('fish.bmp')
its_not_true = filters.sobel(skimage.color.rgb2gray(oh_hi_mark.copy()))
#watershed
watershed = segmentation.watershed(its_not_true, markers = 512)
watershed = segmentation.mark_boundaries(oh_hi_mark, watershed)
#slic
slic = segmentation.slic(oh_hi_mark, n_segments=256, sigma=1, compactness=10)
slic = segmentation.mark_boundaries(oh_hi_mark, slic)
#quickshift
quickshift = segmentation.quickshift(oh_hi_mark, kernel_size=10, max_dist=15, ratio=1)
quickshift = segmentation.mark_boundaries(oh_hi_mark, quickshift)
#felzenszwalb
felzenszwalb = segmentation.felzenszwalb(oh_hi_mark, scale = 25, sigma = 1.5, min_size = 40)
felzenszwalb = segmentation.mark_boundaries(oh_hi_mark, felzenszwalb)
#display
'''segmented = {'watershed': watershed, 'slic': slic, 'quickshift' : quickshift, 'felzenszwalb' : felzenszwalb}
for key in segmented.keys():
    plt.subplot(1, 1, 1, title = key)
    plt.imshow(segmented[key])
    plt.axis('off')
    plt.show()'''
stack1, stack2 = np.hstack((watershed, slic)), np.hstack((quickshift, felzenszwalb))
plt.subplot(2, 1, 1)
plt.imshow(stack1)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(stack2)
plt.axis('off')
plt.show()
