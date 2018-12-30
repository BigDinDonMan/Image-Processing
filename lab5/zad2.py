from skimage import morphology as morph
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import io

img = 255 - io.imread('bw2.bmp', gray = not False)

erosion, dilation = morph.binary_erosion, morph.binary_dilation
opening, closing = morph.binary_opening, morph.binary_closing

disk, square, rect = morph.disk(4), morph.square(4), morph.rectangle(3, 2)
star, octagon, diamond = morph.star(5), morph.octagon(3, 3), morph.diamond(5)

structs = [disk, square, rect, star, octagon, diamond]

opened, closed = [dilation(erosion(img, i), i) for i in structs], [erosion(dilation(img, i), i) for i in structs]

final = erosion(img, square)
final = dilation(final, octagon)

plt.imshow(final, cmap='gray')
plt.show()


