from skimage.segmentation import active_contour
from skimage.filters import gaussian
import numpy as np
import skimage.color as clr
import matplotlib.pyplot as plt
from skimage import io

cat_img = io.imread('cat.jpg')
copy = cat_img.copy()

#butterfly part

s = np.linspace(0, 2 * np.pi, 400)
x, y = 120 + 70 * np.cos(s), 140 + 70 * np.sin(s)
init = np.array([x, y]).T

butterfly_contour = active_contour(gaussian(copy, sigma = 3), init, alpha = 0.015, beta = 10, gamma = 0.001 / 2)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(cat_img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(butterfly_contour[:, 0], butterfly_contour[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, cat_img.shape[1], cat_img.shape[0], 0])
plt.show()

#cat part

coords = (450, 115)
eliptoid_size = (250, 160)
x, y = coords[0] + eliptoid_size[0] * np.cos(s), coords[1] + eliptoid_size[1] * np.sin(s)
init = np.array([x,y]).T

oh_hi_mark = active_contour(gaussian(copy, sigma = 3), init, alpha = 0.0015, beta = 0.001, gamma = 0.00005)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(cat_img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(oh_hi_mark[:, 0], oh_hi_mark[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, cat_img.shape[1], cat_img.shape[0], 0])
plt.show()
