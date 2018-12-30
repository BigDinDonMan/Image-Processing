from skimage.measure import regionprops, label
from skimage.io import imread
from skimage.feature import canny
from skimage.util import invert
import skimage.morphology as morph
import skimage.segmentation as seg
from skimage.filters import threshold_minimum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

#planes in the image will have the lowest solidity (ratio of all the pixels in region to the area pixels)

img = imread('planes.png', gray = True).astype(bool)
disp = img.copy()
img = invert(img)
labels = label(img)
props = regionprops(labels)

dicc = {round(prop.solidity, 2) : [] for prop in props}

for prop in props:
    for key in dicc.keys():
        if math.isclose(round(prop.solidity, 2), key):
            dicc[key].append(prop)

planes = dicc[min(dicc.keys())]

x, D = plt.subplots(figsize = (10, 6))
D.imshow(disp, cmap='gray')

for plane in planes:
    minrows, mincolumns, maxrows, maxcolumns = plane.bbox
    rect = mpatches.Rectangle((mincolumns, minrows), maxcolumns - mincolumns, maxrows - minrows, fill = False, edgecolor='red', linewidth=2)
    D.add_patch(rect)

plt.tight_layout()
plt.show()

