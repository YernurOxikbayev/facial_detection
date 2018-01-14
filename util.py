import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from matplotlib import pyplot as plt
DATASET_PATH = '/Users/lanhnguyen/workspace/facial_point_prediction/train/'


def load(filename, image_height, image_width, test=False):
    train_images, train_labels, raw_images = read_input(filename)
    dataframe = pd.DataFrame(np.asarray(train_labels))
    dataframe['Image'] = pd.Series(train_images, index=dataframe.index)

    feature_cols = dataframe.columns[:-1]  # all but image column
    # transform image space-separated pixel values to normalized pixel vector
    dataframe['Image'] = dataframe['Image'].apply(lambda img: img / 255.0)
    dataframe = dataframe.dropna()  # drop entries w/NaN entries

    # get all image vectors and reshape to a #num_images x image_size x image_size x channels tensor
    X = np.vstack(dataframe['Image'])
    X = X.reshape(-1, image_height, image_width, 1)

    if not test:
        # get label features and scale pixel coordinates by image range
        y = dataframe[feature_cols].values
        # permute (image, label) pairs for training
    else:
        y = None
    return X, y, raw_images


def read_input(file_list):
    labels = []
    images = []
    raw_images = []
    with open(file_list, 'r') as ins:
        for row in ins:
            row_data = row.split()
            img = Image.open(row_data[0]).convert('L')
            box = (int(row_data[1]), int(row_data[3]), int(row_data[2]), int(row_data[4]))
            img = img.crop(box)
            img = img.resize((39, 39), Image.ANTIALIAS)
            for k in range(64):
                raw_images.append(img)
                img_arr = np.asarray(img)
                max = np.amax(img_arr)
                min = np.amin(img_arr)
                img_arr = img_arr/255 * (max-min)
                images.append(img_arr)

                label = row_data[5:(row_data.__len__())]
                label = normalize_label(label, box)
                labels.append(list(map(float, label)))

    return images, labels, raw_images


def normalize_label(label, box):
    i = 0
    while i < label.__len__():
        label[i] = float(label[i])
        if i % 2 == 0:
            label[i] = (label[i] - box[0]) * 39 / (box[2] - box[0])
        else:
            label[i] = (label[i] - box[1]) * 39 / (box[3] - box[1])
        i += 1
    return label


def plot_sample(x, y, truth=None):
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.scatter(y[0::2] * 96, y[1::2] * 96)
    if truth is not None:
        plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
    plt.savefig("data/img.png")

def plot_learning_curve(loss_train_record, loss_valid_record):
    plt.figure()
    plt.plot(loss_train_record, label='train')
    plt.plot(loss_valid_record, c='r', label='validation')
    plt.ylabel("RMSE")
    plt.legend(loc='upper left', frameon=False)
    plt.savefig("data/learning_curve.png")
