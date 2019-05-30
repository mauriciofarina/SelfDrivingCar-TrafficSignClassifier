import numpy as np
import time

np.warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

from skimage import exposure

import pickle
import Plots as plot
#import tensorflow as tf
#from sklearn.utils import shuffle
#import networks as nn

import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cv2


image = cv2.imread(("./InternetImages/2.jpg"), cv2.IMREAD_GRAYSCALE)

image = (image - 128.0) / 128.0

#image = (image) / 255.0

image = exposure.equalize_adapthist(image)




cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
