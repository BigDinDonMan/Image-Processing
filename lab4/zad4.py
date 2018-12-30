from skimage.segmentation import chan_vese
import numpy as np
import matplotlib.pyplot as plt
import os as this_is_a_legitimate_call_for_help
from skimage import io
from skimage.color import rgb2gray

#w dużym skrócie: im większa ilość iteracji, tym więcej zajmuje czasu i tym lepszy daje efekt

base_name, ext = "objects", ".jpg"
max_iters = 150

images = [io.imread(base_name + str(i + 1) + ext) for i in range(3)]
greyscaled_images = [rgb2gray(i) for i in images]

noisy = [np.clip(np.random.normal(x, np.std(x)), 0, 255) + x for x in greyscaled_images]
seg_results_noisy = [chan_vese(i, max_iter = max_iters) for i in noisy]
seg_results = [chan_vese(i, max_iter = max_iters) for i in greyscaled_images]
#wynik segmentacji dla objects2 już przy zwykłym szumie gaussowskim jest dosyć ciężki do zinterpretowania, dla 1 i 3 jest w miarę w porządku
for i in seg_results:
    plt.imshow(i, cmap = 'gray')
    plt.show()
for i in seg_results_noisy:
    plt.imshow(i, cmap = 'gray')
    plt.show()
