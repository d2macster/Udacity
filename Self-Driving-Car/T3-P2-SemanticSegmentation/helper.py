import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
import time
import cv2

ia.seed(int(time.time()))

# augmentation ideas were inspired after reading
# https://github.com/aleju/imgaug
augmentation_seq = iaa.Sequential([
    iaa.Crop(px=(0, 30), name="Cropper"),  # crop images from each side by 0 to 30px (randomly chosen)
    iaa.Fliplr(0.5, name="Flipper"),
    # iaa.GaussianBlur((0, 1.0), name="GaussianBlur"),
    iaa.Add((-10, 10), name="Brighness"),
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


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_kitti(data_dir):
    # http://kitti.is.tue.mpg.de/kitti/data_road.zip
    """
    :param data_dir: Directory to download the model to
    """
    kitti_filename = 'data_road.zip'
    kitti_path = os.path.join(data_dir, 'data_road')
    kitti_files = [
        os.path.join(kitti_path, 'testing/image_2/uu_000000.png'),
        os.path.join(kitti_path, 'training/image_2/uu_000000.png'),
        os.path.join(kitti_path, 'training/gt_image_2/um_road_000000.png')]

    missing_kitti_files = [f for f in kitti_files if not os.path.exists(f)]
    if missing_kitti_files:
        # Clean vgg dir
        if os.path.exists(kitti_path):
            shutil.rmtree(kitti_path)
        os.makedirs(kitti_path)

        # Download vgg
        print('Downloading kitti data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'http://kitti.is.tue.mpg.de/kitti/data_road.zip',
                os.path.join(kitti_path, kitti_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(kitti_path, kitti_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(kitti_path, kitti_filename))



def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []

            ia.seed(int(time.time()))

            augmentation_seq_deterministic = augmentation_seq.to_deterministic()


            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                image = augmentation_seq_deterministic.augment_images([image])[0]
                gt_image = augmentation_seq_deterministic.augment_images([gt_image], hooks=hooks_binmasks)[0]

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def make_a_movie(data_path, video_output):
    images = os.listdir(data_path)
    # Determine the width and height from the first image
    image_path = os.path.join(data_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(video_output, fourcc, 20.0, (width, height))

    for image in images:
        image_path = os.path.join(data_path, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()