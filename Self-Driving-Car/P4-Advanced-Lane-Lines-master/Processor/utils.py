import cv2
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def save_image(image, image_path):
    mpimg.imsave(image_path, image)


def read_image(image):
    return mpimg.imread(image)


def get_image_list(input_path):
    images = glob.glob("{}/*.jpeg".format(input_path))
    return list(map(lambda x: (x, os.path.split(x)), images))