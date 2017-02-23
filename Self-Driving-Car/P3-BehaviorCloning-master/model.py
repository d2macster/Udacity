import cv2
import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

input_img_shape = (160, 320, 3)
output_img_shape = (66, 200, 3)

trans_range = 100
angle_delta = 0.25

window_size = 10
batch_size = 128
dropout = .25


def parse_cvs(data_dir, window_size):
    # center,left,right,steering,throttle,brake,speed
    data = pd.read_csv('{}/driving_log.csv'.format(data_dir))
    steering = data['steering'].values
    L = len(steering)

    center_angle = np.zeros(L)
    left_angle = np.zeros(L)
    right_angle = np.zeros(L)

    center_angle[:] = steering[:]

    for i in range(0, L - window_size):
        v = 1.0 * sum(steering[i + 1:i + 1 + window_size]) / window_size
        if center_angle[i] == 0.0:
            center_angle[i] = v

    left_angle[:] = 1.2 * center_angle[:] + angle_delta
    right_angle[:] = 1.2 * center_angle[:] - angle_delta

    data['center'] = data_dir + '/' + data['center']
    data['left'] = data_dir + '/' + data['left']
    data['right'] = data_dir + '/' + data['right']

    data['center_angle'] = center_angle
    data['left_angle'] = left_angle
    data['right_angle'] = right_angle

    df1 = data[['center', 'center_angle']]
    df1.columns = ['img', 'angle']

    df2 = data[['left', 'left_angle']]
    df2.columns = ['img', 'angle']

    df3 = data[['right', 'right_angle']]
    df3.columns = ['img', 'angle']

    return (df1, df2, df3)

def image_brightness(img):
    # courtesy https://chatbotslife.com/@vivek.yadav

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    hsv[::2] = hsv[::2] * random_bright

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def trans_image(img, angle):
    rows, cols, _ = img.shape
    tr_x = 1.0 * trans_range * (np.random.uniform() - 0.5)
    angle += 3.0 * angle_delta * tr_x / trans_range
    tr_y = 40 * (np.random.uniform() - 0.5)
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(img, Trans_M, (cols, rows))

    return image_tr, angle


def flip_image_at_random(img, angle):
    choice = np.random.randint(2)
    if choice == 1:
        img = cv2.flip(img, 1)
        angle = -1.0 * angle
    return (img, angle)


def crop_reshape(img):
    img = img[60:120, :, :]
    img = cv2.resize(img, (output_img_shape[1], output_img_shape[0]), interpolation=cv2.INTER_AREA)
    return img


def preprocess_image(img, angle):
    img, angle = flip_image_at_random(img, angle)
    img, angle = trans_image(img, angle)
    img = image_brightness(img)
    # img = add_random_shadow(img)
    img = crop_reshape(img)

    return img, angle


def generator_sequential_tuple(data):
    while 1:
        for ind in range(len(data)):
            pairs = data.ix[ind]
            x = cv2.imread(pairs['img'])
            y = pairs['angle']
            x, y = preprocess_image(img=x, angle=y)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = np.array([[y]])
            yield x, y


def generator_batch(data, batch_size):
    batch_x = np.zeros(shape=(
        batch_size,
        output_img_shape[0],
        output_img_shape[1],
        output_img_shape[2]))
    batch_y = np.zeros(shape=batch_size)
    L = len(data)

    while 1:

        for i in range(0, batch_size):
            ind = random.randint(0, L - 1)
            pairs = data.ix[ind]
            x = cv2.imread(pairs['img'])
            y = pairs['angle']
            x, y = preprocess_image(img=x, angle=y)

            batch_x[i] = x
            batch_y[i] = y

        yield batch_x, batch_y


def generate_model_pooling():
    model = Sequential()
    model.add(Lambda(lambda x: x * 1. / 127.5 - 1.,
                     input_shape=output_img_shape,
                     output_shape=output_img_shape))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Activation(activation='elu'))
    model.add(Dropout(dropout))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Activation(activation='elu'))
    model.add(Dropout(dropout))

    model.add(Convolution2D(48, 5, 5, border_mode='valid', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Activation(activation='elu'))
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', activation='elu'))
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', activation='elu'))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == '__main__':
    # data_prefix = "/Users/andriicherniak/Desktop/"
    # img = cv2.imread("{}/1.jpg".format(data_prefix))
    # img = image_brightness(img)
    # cv2.imwrite("{}/out.jpg".format(data_prefix), img)

    data_prefix = "/home/carnd/data"
    data_prefix_2 = "/home/carnd/data_2"
    data_prefix_3 = "/home/carnd/data_3"
    data_prefix_4 = "/home/carnd/data_4"

    # data_prefix = "/Users/andriicherniak/Desktop/data"

    [tuples_center_1, tuples_left_1, tuples_right_1] = parse_cvs(
        data_dir=data_prefix,
        window_size=window_size)

    # [tuples_center_2, tuples_left_2, tuples_right_2] = parse_cvs(
    #     data_dir=data_prefix_2,
    #     window_size=window_size)
    #
    # [tuples_center_3, tuples_left_3, tuples_right_3] = parse_cvs(
    #     data_dir=data_prefix_3,
    #     window_size=window_size)
    #
    # [tuples_center_4, tuples_left_4, tuples_right_4] = parse_cvs(
    #     data_dir=data_prefix_4,
    #     window_size=window_size)

    tuples_union_1 = tuples_center_1.append([tuples_left_1, tuples_right_1])
    # tuples_union_2 = tuples_center_2.append([tuples_left_2, tuples_right_2])
    # tuples_union_3 = tuples_center_3.append([tuples_left_3, tuples_right_3])
    # tuples_union_4 = tuples_center_4.append([tuples_left_4, tuples_right_4])
    #
    # tuples_union = tuples_union_1.append([tuples_union_2, tuples_union_3, tuples_union_4])
    tuples_union = tuples_union_1

    L = len(tuples_union)
    val_size = int(0.1 * L)

    val_tuples = tuples_union.sample(val_size, replace=False).reset_index()
    val_tuples = val_tuples[['img', 'angle']]
    val_images = set(val_tuples['img'])

    train_tuples = tuples_union[~tuples_union['img'].isin(val_images)].reset_index()
    train_tuples = train_tuples[['img', 'angle']]

    L = len(train_tuples)
    n_batches = int(L / batch_size)

    g_train = generator_batch(
        data=train_tuples,
        batch_size=batch_size)
    g_val = generator_sequential_tuple(
        data=val_tuples)

    model = generate_model_pooling()
    model.fit_generator(g_train,
                        samples_per_epoch=2 * n_batches * batch_size,
                        nb_epoch=2,
                        validation_data=g_val,
                        nb_val_samples=2 * val_size)

    model.save("model.h5")

    # print(model.summary())
    # plot(model, to_file="model.png", show_shapes=True)
