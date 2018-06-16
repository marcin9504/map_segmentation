import os

import cv2
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential, load_model
from sklearn.utils import shuffle

model_filename = "model.h5"

train_loc_output = "train/map/"
train_loc_input = "train/sat/"
test_loc_output = "test/map/"
test_loc_input = "test/sat/"
window_size = 20


# IS_ROAD = np.array([1, 0])  # TODO unnecessary
# IS_NOT_ROAD = np.array([0, 1])  # TODO unnecessary
# THRESHOLD = 120  # TODO unnecessary


def get_model():
    try:
        return load_model(model_filename)
    except:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(window_size, window_size, 3), padding='same', activation='relu',
                         kernel_constraint=maxnorm(3)))  # TODO change input layer(?)
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(2, activation='softmax'))  # TODO change output layer

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


def save_model(net):
    net.save(model_filename)


def get_sliced_images(img_input, img_output):
    x_train = slice_image(img_input, window_size)
    y_train = slice_image(img_output, window_size)
    # TODO change format of input/output
    return shuffle(x_train, y_train)


def slice_image(img, window):
    windows = []
    rows, columns, _ = np.shape(img)
    for r in range(0, rows, window):
        for c in range(0, columns, window):
            windows.append(img[r:r + window, c:c + window])
    return windows


def load_img(name):
    img = cv2.imread(name)
    return img


# def prepare_data(img_input, img_output):  # TODO unnecessary
#     slices_input = slice_image(img_input, window_size)
#     slices_output = slice_image(img_output, window_size)
#     x_train = np.zeros((len(slices_input), 20, 20, 3))
#     y_train = np.zeros((len(slices_output), 2))
#
#     for i, (slice_input, slice_output) in enumerate(zip(slices_input, slices_output)):
#         if np.sum(slice_input > THRESHOLD) >= 1:
#             y_train[i] = IS_ROAD
#             x_train[i] = slice_input
#         else:
#             y_train[i] = IS_NOT_ROAD
#             x_train[i] = slice_input
#
#     return shuffle(x_train, y_train)


# def get_surroundings(img, x, y):  # TODO unnecessary
#     return img[x - 9: x + 11, y - 9: y + 11]


# def get_random_base_data_from_images(img_input, img_output):  # TODO unnecessary
#     x_roads = []
#     x_no_roads = []
#     for i in range(1000):
#         x = random.randint(10, 589)
#         y = random.randint(10, 589)
#         img_input = get_surroundings(img_input, x, y)
#         img_output = get_surroundings(img_output, x, y)
#         if np.sum(img_output > THRESHOLD) >= 1:
#             x_roads.append(img_input)
#         else:
#             x_no_roads.append(img_input)
#     min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)
#
#     x_roads_np = np.array(x_roads[0:min_len])
#     x_no_roads_np = np.array(x_no_roads[0:min_len])
#     x_train = np.concatenate((x_roads_np, x_no_roads_np))
#
#     y_train = np.zeros((2 * min_len, 2))
#     y_train[0: min_len] = IS_ROAD
#     y_train[min_len: 2 * min_len] = IS_NOT_ROAD
#
#     return shuffle(x_train, y_train)


def train_model():
    filename_inputs = np.array(next(os.walk(train_loc_input))[2])
    filename_outputs = np.array(next(os.walk(train_loc_output))[2])
    filename_inputs, filename_outputs = shuffle(filename_inputs, filename_outputs)
    model = get_model()

    counter = 0
    image_number = 0
    for iteration in range(1):
        for filename_input, filename_output in zip(filename_inputs, filename_outputs):
            train_input = load_img(train_loc_input + filename_input)
            train_output = load_img(train_loc_output + filename_output)
            print(str(image_number) + "/" + str(len(filename_outputs)) + ": " + str(iteration))
            # x, y = prepare_data(train_input, train_output)
            x, y = get_sliced_images(train_input, train_output)
            model.fit(x, y, epochs=1)
            image_number += 1
            counter += 1
            if counter == 50:
                counter = 0
                save_model(model)
        image_number = 0
    save_model(model)


def test_model():
    filename_inputs = next(os.walk(test_loc_input))[2]
    filename_outputs = next(os.walk(test_loc_output))[2]
    model = get_model()

    for filename_input, filename_output in zip(filename_inputs, filename_outputs):
        test_input = load_img(test_loc_input + filename_input)
        test_output = load_img(test_loc_output + filename_output)
        # x, y = get_random_base_data_from_images(test_input, test_output)
        x, y = get_sliced_images(test_input, test_output)
        loss, accuracy = model.evaluate(x, y)
        print('Loss:', loss)
        print('Accuracy:', accuracy)


def main():
    # train
    train_model()
    # test
    test_model()


if __name__ == '__main__':
    main()
