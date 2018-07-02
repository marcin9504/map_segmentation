import cv2
import keras.backend as K
import numpy as np
from keras.losses import binary_crossentropy
from keras.models import load_model, model_from_json

model_filename = "model.h5"

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


def get_model2():
    json_file = open('json_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,
                            custom_objects={'bce_dice_loss': bce_dice_loss,
                                            'dice_coeff': dice_coeff})
    model.load_weights('weights-model.hdf5')
    return model


def save_model2(model):
    f = open("json_model.json", 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close()
    model.save_weights('weights-model.hdf5')


def get_image_from_net(test_input):
    model = get_model()
    # save_model2(model)
    # model = get_model2()

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
    _, img = cv2.threshold(img, 0.05, 1, cv2.THRESH_BINARY)
    # cv2.imshow('Img2', cv2.resize(img, (800, 800)))
    # cv2.waitKey(1000000)
    return img


def main():
    img = load_img('sat/10528735_15.tiff')
    output = roads(img)
    print(np.shape(output))


if __name__ == '__main__':
    main()
