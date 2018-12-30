import cv2
import numpy as np
import skimage.filters as f
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import invert

image = cv2.cvtColor(cv2.imread('brain_tumor.bmp'), cv2.COLOR_BGR2GRAY)
thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

#progowanie z ręcznym doborem progu
imgs = [cv2.threshold(image, 220, 255, thresh)[1] for thresh in thresholds]
stack = np.hstack(tuple(imgs))
cv2.imshow('trial and error threshold values', stack)
cv2.waitKey()
cv2.destroyAllWindows()

#progowanie z automatycznym doborem progu
imgs = [cv2.threshold(image, 0, 255, thresh | cv2.THRESH_OTSU)[1] for thresh in thresholds]
stack = np.hstack(tuple(imgs))
cv2.imshow('auto threshold values', stack)
cv2.waitKey()
cv2.destroyAllWindows()

#progowanie adaptacyjne
    
thresholds.clear()
thresholds = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
imgs = [cv2.adaptiveThreshold(image, 255, thresh, cv2.THRESH_BINARY, 2551, 8) for thresh in thresholds]
stack = np.hstack(tuple(imgs))
cv2.imshow('adaptive threshold values', stack)
cv2.waitKey()
cv2.destroyAllWindows()


    
    

#wartości progu dla progowania z ręcznym doborem progu: 220, 255 dały najlepsze rezultaty
#najlepsze progowanie z automatycznym doborem daje THRESH_TOZERO
#ostatni parametr adaptiveThreshold odpowiada za zmniejszenie o jego wartość średniej liczonej z sąsiedztwa, którego wielkość określamy przedostatnim parametrem
#przedostatni parametr to musi być liczba nieparzysta
#wyniki adaptiveThreshold są porównywalne, aczkolwiek MEAN dał nieco lepszy wynik od GAUSSIAN (gaussowski dał trochę więcej "szumów" (tj. niepożądanych białych elementów))

