import os

import cv2
import keras.backend as K
import numpy as np
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from sklearn.utils import shuffle

model_filename = "model.h5"

train_loc_output = "map/"
train_loc_input = "sat/"
test_loc_output = "test/map/"
test_loc_input = "test/sat/"
window_size = 128


# https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/39973
def dice_coeff(y, out):
    y_f = K.flatten(y)
    out_f = K.flatten(out)
    intersection = K.sum(y_f * out_f)
    score = (2. * intersection + 1.) / (K.sum(y_f) + K.sum(out_f) + 1.)
    return score


def dice_loss(y, out):
    loss = 1 - dice_coeff(y, out)
    return loss


def bce_dice_loss(y, out):
    loss = binary_crossentropy(y, out) + dice_loss(y, out)
    return loss


def get_model():
    try:
        model = load_model(model_filename,
                           custom_objects={'bce_dice_loss': bce_dice_loss,
                                           'dice_coeff': dice_coeff})
        print("Loaded saved model")
        return model
    except Exception as e:
        print(e)
        inputs = Input(shape=(128, 128, 3))

        down1 = Conv2D(64, (3, 3), padding='same')(inputs)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(64, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(128, (3, 3), padding='same')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

        down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3 = Conv2D(256, (3, 3), padding='same')(down3)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

        down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4 = Conv2D(512, (3, 3), padding='same')(down4)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

        center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(1024, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)

        up2 = UpSampling2D((2, 2))(center)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)

        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)

        classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

        model = Model(inputs=inputs, outputs=classify)

        model.compile(optimizer=RMSprop(lr=0.01, decay=1e-6), loss=bce_dice_loss, metrics=[dice_coeff])
        model.summary()
    return model


def save_model(net):
    net.save(model_filename)
    print("Model saved")


def slice_for_test(img, window):
    rows, columns, _ = np.shape(img)
    windows = np.empty((int(rows / window * columns / window), 2))
    for r in range(0, rows, window):
        for c in range(0, columns, window):
            output = ([1, 0] if all(img[r + window // 2, c + window // 2] == 255) else [0, 1])
            windows[int(r / window * columns / window + c / window)] = output
    return windows


def get_sliced_images(img_input, img_output):
    # img_input = cv2.resize(img_input, (128, 128))
    # img_output = cv2.resize(img_output, (128, 128))
    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    img_input = img_input / 255.
    img_output = img_output / 255.
    img_output = np.expand_dims(img_output, axis=2)
    x_train = slice_image(img_input, window_size)
    y_train = slice_image(img_output, window_size)
    return shuffle(x_train, y_train)


def slice_image(img, window):
    rows, columns, depth = np.shape(img)
    windows = np.empty((int(rows // window * columns // window), window, window, depth))

    for r in range(0, rows, window):
        for c in range(0, columns, window):
            windows[int(r / window * columns / window + c / window)] = img[r:r + window, c:c + window]
    return windows


def load_img(name):
    img = cv2.imread(name)
    return img


def train_model():
    filename_inputs = np.array(next(os.walk(train_loc_input))[2])
    # filename_outputs = np.array(next(os.walk(train_loc_output))[2])
    filename_inputs = shuffle(filename_inputs)
    model = get_model()

    image_number = 0
    for iteration in range(1000):
        for filename_input in filename_inputs:
            filename_output = filename_input[:-5] + ".tif"
            train_input = load_img(train_loc_input + filename_input)
            train_output = load_img(train_loc_output + filename_output)
            print(str(image_number) + "/" + str(len(filename_inputs)) + ": " + str(iteration))
            # x, y = prepare_data(train_input, train_output)
            train_input = cv2.resize(train_input, (1408, 1408))
            train_output = cv2.resize(train_output, (1408, 1408))
            for r in range(0, 1408, 128):
                t_input = train_input[r:r + 128, :]
                t_output = train_output[r:r + 128, :]
                x, y = get_sliced_images(np.copy(t_input), np.copy(t_output))
                model.fit(x, y, epochs=1)
            image_number += 1
            save_model(model)
        image_number = 0
    save_model(model)


def test_model(visualise=False):
    filename_inputs = next(os.walk(test_loc_input))[2]
    # filename_outputs = next(os.walk(test_loc_output))[2]
    model = get_model()

    for filename_input in filename_inputs:
        filename_output = filename_input[:-5] + ".tif"
        test_input = load_img(test_loc_input + filename_input)
        test_output = load_img(test_loc_output + filename_output)
        test_input = cv2.resize(test_input, (1408, 1408))
        test_output = cv2.resize(test_output, (1408, 1408))
        for r in range(0, 1408, 128):
            for c in range(0, 1408, 128):
                t_input = test_input[r:r + 128, c:c + 128]
                t_output = test_output[r:r + 128, c:c + 128]
                x, y = get_sliced_images(np.copy(t_input), np.copy(t_output))
                out = model.predict(x)
                loss, accuracy = model.evaluate(x, y)
                print('Loss:', loss)
                print('Accuracy:', accuracy)
                if visualise:
                    for a in x:
                        cv2.imshow('Input', a)
                        cv2.waitKey(1)
                    for a in y:
                        cv2.imshow('Output', a)
                        cv2.waitKey(1)
                    for a in out:
                        cv2.imshow('Prediction', a)
                        cv2.waitKey(3000)


def main():
    # train_model()
    test_model(visualise=True)


if __name__ == '__main__':
    main()
