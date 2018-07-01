import cv2
import keras.backend as K
import numpy as np
from keras.losses import binary_crossentropy
from keras.models import load_model

model_filename = "model.h5"

window_size = 128


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def get_model():
    model = load_model(model_filename,
                       custom_objects={'bce_dice_loss': bce_dice_loss,
                                       'dice_coeff': dice_coeff})
    return model


def get_sliced_images(img_input):
    img_input = img_input / 255.
    x_train = slice_image(img_input, window_size)
    return x_train


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


def get_image_from_net(test_input):
    model = get_model()
    test_input = cv2.resize(test_input, (1408, 1408))
    output_image = np.zeros([1408, 1408, 1])
    for r in range(0, 1408, 128):
        for c in range(0, 1408, 128):
            t_input = test_input[r:r + 128, c:c + 128]
            x = get_sliced_images(np.copy(t_input))
            out = model.predict(x)
            output_image[r:r + 128, c:c + 128] = out[0][:, :]
    return output_image


def roads(img):
    img = get_image_from_net(img)
    img = cv2.resize(img, (1500, 1500))
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    _, img = cv2.threshold(img, 0.2, 1, cv2.THRESH_BINARY)
    # cv2.imshow('Img', cv2.resize(img, (800, 800)))
    # cv2.waitKey(1000000)
    return img


def main():
    img = load_img('train/sat/10078660_15.tiff')
    output = roads(img)
    print(np.shape(img))


if __name__ == '__main__':
    main()
