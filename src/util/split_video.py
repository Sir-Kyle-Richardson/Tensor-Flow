from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.transform import resize   # for resizing images
from keras.utils import np_utils
import numpy as np    # for mathematical operations
from keras.preprocessing import image   # for preprocessing the images
import pandas as pd
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib
import matplotlib.pyplot as plt    # for plotting the images

# Splits video by frames into src/tmp folder
def splitVideo(video):
    try:

    except Error: 
        return False