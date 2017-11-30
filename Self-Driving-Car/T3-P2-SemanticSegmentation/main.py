import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import glob

L2_REG = 1e-6
STDEV = 1e-2
KEEP_PROB = 0.5
LEARNING_RATE = 0.5e-4
EPOCHS = 60
BATCH_SIZE = 5

IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2

DATA_DIR = '../data'
RUNS_DIR = '../runs'
MODELS_DIR = '../models'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    # print the graph , thus later we can visualize it with
    # tensorboard - -logdir = logs
    # tf.summary.FileWriter("logs", graph)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_layer = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob_layer, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output

    """
    # for this work i re-use a tutorial on upsampling
    # http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
    #
    # Here is the summary of the article aplicable to our problem
    #
    # The factor of upsampling is equal to the stride of transposed convolution.
    # The kernel of the upsampling operation is determined by the identity: 2 * factor - factor % 2.

    kernel_initializer = tf.truncated_normal_initializer(stddev=STDEV)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG)

    # 1x1 convolution
    l_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='SAME',
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer)
    l_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='SAME',
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer)
    l_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='SAME',
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer)

    # upsampling result = l_7 -> 2x
    l_7_up = tf.layers.conv2d_transpose(l_7, num_classes, 4, 2, padding='SAME',
                                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    skip_1 = tf.add(l_4, l_7_up)

    # upsampling result = l_7 -> 4x, l_4 -> 2x
    skip_1_up = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding='SAME',
                                           kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    skip_2 = tf.add(skip_1_up, l_3)

    # upsampling result = l_7 -> 32x, l_4 -> 16x, l_3 -> 8x
    nn_last_layer = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding='SAME',
                                               kernel_initializer=kernel_initializer,
                                               kernel_regularizer=kernel_regularizer)
    return nn_last_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # tf.losses.softmax_cross_entropy() takes labels and logits,
    # performs softmax activation on logits, calculates cross-entropy,
    # and returns our loss as a scalar Tensor
    soft_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(soft_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, logits, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, cv=False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # For this work we will be also doing validation
    # the idea is inspired by
    # https://medium.com/@RecastAI/towards-an-augmented-dataset-cdf3b9b605ec
    # Given that we perform constant data augmentation, we can get almost
    # infinite sample of non-repeating images

    print("training")
    for e in range(epochs):
        batch_loss = []
        cv_iou = []

        # TRAINING
        for images, gt_images in get_batches_fn(batch_size):
            opt_, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: images,
                correct_label: gt_images,
                learning_rate: LEARNING_RATE,
                keep_prob: KEEP_PROB
            })
            batch_loss.append(loss)

        print("Epoch {} loss {}".format(e, np.average(batch_loss)))

        # CV
        if cv and (e <= 3 or e == (epochs - 1) or (e + 1) % 10 == 0):
            for images, gt_images in get_batches_fn(1):
                im_softmax = sess.run(
                    [tf.nn.softmax(logits)],
                    {keep_prob: 1.0, input_image: [images[0]]})
                im_softmax = im_softmax[0]
                vfunc = np.vectorize(lambda x: 1 if x > 1.0/NUM_CLASSES else 0)
                im_softmax = vfunc(im_softmax)

                intersection = 0
                union = 0
                for c_id in range(NUM_CLASSES):
                    predictions = im_softmax[:,c_id].reshape(-1)
                    labels = gt_images[0][:,:,c_id].reshape(-1)

                    i_set = np.logical_and(predictions, labels)
                    u_set = np.logical_or(predictions, labels)
                    intersection += (i_set == True).sum()
                    union += (u_set == True).sum()
                iou = 1.0*intersection / union
                cv_iou.append(iou)

            print("Epoch {} cv {}".format(e, np.average(cv_iou)))


tests.test_train_nn(train_nn)


def run():
    # Download kitti data set
    helper.maybe_download_kitti(DATA_DIR)
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        correct_label = tf.placeholder(tf.int32, [None, None, None, NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Create function to get batches
        # returns images and labels
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # added augmentation using imgaug library
        image_input, keep_prob_layer, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, NUM_CLASSES)

        sess.run(tf.global_variables_initializer())

        train_nn(sess, logits, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob_layer, learning_rate, False)

        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob_layer, image_input)

    # generating movie from test images
    for r_data in glob.glob("{}/*".format(RUNS_DIR)):
        movie_path = "{}.mp4".format(r_data)
        helper.make_a_movie(r_data, movie_path)
        print("processed {}".format(r_data))

if __name__ == '__main__':
    run()
