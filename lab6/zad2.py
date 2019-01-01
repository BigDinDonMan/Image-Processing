from skimage import measure as msr
from skimage import segmentation as seg
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.morphology as morph
from skimage.filters import threshold_minimum, threshold_otsu
import matplotlib.patches as mpatches
from skimage.feature import canny
from skimage.filters import sobel, prewitt
from scipy.ndimage import binary_fill_holes
from enum import Enum

class segmentation_method(Enum):
    threshold = 0
    edge_detection = 1

def is_similar(area1, area2) -> bool:
    return True if (abs(area1.area - area2.area) / area1.area * 100) <= 10 else False

def sum_area(areas : list) -> int:
    return sum([x.area for x in areas])

def count_coins(choice, **kwargs) -> None:
    coins = io.imread('coins.png', gray = True)
    img = coins.copy()
    if choice == segmentation_method.threshold:
        if kwargs["thresh"]: coins = coins > kwargs["thresh"](coins)
        else: raise TypeError("Error: no segmentation method given")
    elif choice == segmentation_method.edge_detection:
        if kwargs["edge"]:
            if kwargs["edge"] == canny and kwargs["sigma"]: coins = kwargs["edge"](coins, sigma = kwargs["sigma"])
            else: coins = kwargs["edge"](coins > threshold_minimum(coins))
    else: raise ValueError("wrong segmentation method")
    coins = seg.clear_border(binary_fill_holes(coins))
    areas = list(filter(lambda x: x.area > 1000, msr.regionprops(msr.label(coins))))
    print('total area: {0}'.format(sum_area(areas)))

    areas_dict = {obj.area : obj for obj in areas}
    max_area = max(areas_dict.keys())
    max_tup = (max_area, areas_dict[max_area])

    del areas_dict[max_area]

    x, D = plt.subplots(figsize=(10, 6))
    D.imshow(img, cmap = 'gray')

    for area in areas:
        minrows, mincolumns, maxrows, maxcolumns = area.bbox
        rect = mpatches.Rectangle((mincolumns, minrows), maxcolumns - mincolumns, maxrows - minrows, fill = False, edgecolor='red' if is_similar(max_tup[1], area) else 'green', linewidth=2)
        D.add_patch(rect)
    
    plt.tight_layout()
    plt.show()

count_coins(segmentation_method.threshold, thresh = threshold_minimum)
count_coins(segmentation_method.threshold, thresh = threshold_otsu)
count_coins(segmentation_method.edge_detection, edge = canny, sigma = 1)


