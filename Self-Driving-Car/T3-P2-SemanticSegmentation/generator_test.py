from glob import glob
import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import scipy.misc
import imgaug as ia
from imgaug import augmenters as iaa
import time

ia.seed(int(time.time()))

ROTATION = 10
SHIFT = 10
IMAGE_ROWS = 160
IMAGE_COLS = 576
SHEAR = IMAGE_ROWS * math.sin(20 * math.pi / 180.0)


def load_resize_image(image_file, image_shape):
    return scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)


def gen_batch_function(data_folder, image_shape):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    random.shuffle(image_paths)
    image_file = image_paths[0]
    gt_image_file = label_paths[os.path.basename(image_file)]

    image = load_resize_image(image_file=image_file, image_shape=image_shape)
    gt_image = load_resize_image(image_file=gt_image_file, image_shape=image_shape)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.imshow(image)
    ax2.imshow(gt_image)

    ia.seed(int(time.time()))

    augmentation_seq = iaa.Sequential([
        iaa.Crop(px=(0, 30), name="Cropper"),  # crop images from each side by 0 to 30px (randomly chosen)
        iaa.Fliplr(0.5, name="Flipper"),
        # iaa.GaussianBlur((0, 1.0), name="GaussianBlur"),
        iaa.Add((-30, 30), name="Brighness"),
        iaa.ContrastNormalization((0.5, 2.0), name="Contrast"),
        iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise"),
        iaa.Affine(translate_px={"x": (-30, 30), "y": (-30, 30)},
                   shear=(-20, 20),
                   rotate=(-10, 10),
                   name="Affine")
    ])

    # change the activated augmenters for binary masks,
    # we only want to execute horizontal crop, flip and affine transformation
    def activator_binmasks(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Brighness", "GaussianNoise", "Contrast"]:
            return False
        else:
            # default value for all other augmenters
            return default

    hooks_binmasks = ia.HooksImages(activator=activator_binmasks)

    augmentation_seq_deterministic = augmentation_seq.to_deterministic()
    batch_inputs = augmentation_seq_deterministic.augment_images([image])
    batch_targets = augmentation_seq_deterministic.augment_images([gt_image], hooks=hooks_binmasks)


    ax3.imshow(batch_inputs[0])
    ax4.imshow(batch_targets[0])
    plt.show()

if __name__ == '__main__':
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '../data'
    runs_dir = './runs'
    model_checkpoint = './model.ckpt'

    gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
