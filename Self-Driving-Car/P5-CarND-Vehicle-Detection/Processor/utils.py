import os
import matplotlib.image as mpimg

def save_image(image, image_path):
    mpimg.imsave(image_path, image)


def read_image(image):
    return mpimg.imread(image)

def file_name(path):
    return os.path.split(path)[-1]