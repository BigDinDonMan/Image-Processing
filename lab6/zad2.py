from skimage import measure as msr
from skimage import segmentation as seg
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.morphology as morph
from skimage.filters import threshold_minimum, threshold_otsu
import matplotlib.patches as mpatches

def is_similar(area1, area2) -> bool:
    return True if (abs(area1.area - area2.area) / area1.area * 100) <= 10 else False

def sum_area(areas : list) -> int:
    return sum([x.area for x in areas])

def count_coins(threshold_method):
    coins = io.imread('coins.png', gray = True)
    thresh = morph.binary_closing(coins > threshold_method(coins))
    cleared = seg.clear_border(thresh)
    labels = msr.label(cleared)
    areas = msr.regionprops(labels)
    print('total area: {0}'.format(sum_area(areas)))

    areas_dict = {obj.area : obj for obj in areas}
    max_area = max(areas_dict.keys())
    max_tup = (max_area, areas_dict[max_area])

    del areas_dict[max_area]

    result_image = coins.copy()

    x, D = plt.subplots(figsize=(10, 6))
    D.imshow(coins, cmap = 'gray')

    for area in areas:
        minrows, mincolumns, maxrows, maxcolumns = area.bbox
        rect = mpatches.Rectangle((mincolumns, minrows), maxcolumns - mincolumns, maxrows - minrows, fill = False, edgecolor='red' if is_similar(max_tup[1], area) else 'green', linewidth=2)
        D.add_patch(rect)
    
    plt.tight_layout()
    plt.show()

count_coins(threshold_minimum)
count_coins(threshold_otsu)

