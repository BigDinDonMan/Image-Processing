from skimage import io
from skimage import segmentation as seg
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = io.imread('lungs_lesion.bmp')

samples = ['lungs_lesion_seeds1.bmp', 'lungs_lesion_seeds2.bmp', 'lungs_lesion_seeds3.bmp']

images = [io.imread(sample) for sample in samples]

results = [seg.random_walker(img, mask, beta = 0, mode = 'bf') * 255 for mask in images]
results_50k = [seg.random_walker(img, mask, beta = 50000, mode = 'bf') * 255 for mask in images]
results_100k = [seg.random_walker(img, mask, beta = 100000, mode = 'bf') * 255 for mask in images]

stacks = [np.hstack((img, images[0], results[0])), np.hstack((img, images[1], results[1])), np.hstack((img, images[2], results[2]))]
stacks_50k = [np.hstack((img, images[0], results_50k[0])), np.hstack((img, images[1], results_50k[1])), np.hstack((img, images[2], results_50k[2]))]
stacks_100k = [np.hstack((img, images[0], results_100k[0])), np.hstack((img, images[1], results_100k[1])), np.hstack((img, images[2], results_100k[2]))]

for stack in stacks:
    plt.imshow(stack, cmap = 'gray')
    plt.show()
for stack in stacks_50k:
    plt.imshow(stack, cmap = 'gray')
    plt.show()
for stack in stacks_100k:
    plt.imshow(stack, cmap = 'gray')
    plt.show()

    
