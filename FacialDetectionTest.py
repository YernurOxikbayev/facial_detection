import tensorflow as tf
import logging
from util import *
import tkinter
from tkinter import filedialog
import subprocess

NUM_EPOCHS = 100         # training epoch count
EARLY_STOP_PATIENCE = 1000   # epoch count before we stop if no improvement
VALIDATION_SIZE = 3466       # number of samples in validation dataset
BATCH_SIZE = 64            # number of samples in a training mini-batch
EVAL_BATCH_SIZE = 64        # number of samples in intermediate batch for performance check per epoch
NUM_CHANNELS = 1            # color channels in images
NUM_LABELS = 10             # number of label features per image
SEED = None                 # random seed for weight initialization (None = random)
DEBUG_MODE = True           # print debug lines
DROPOUT_RATE = 0.50         # chance of neuron dropout in training
L2_REG_PARAM = 1e-7         # lambda for L2 regularization of fully connected layer
LEARNING_BASE_RATE = 1e-3   # rates for exponential decay of learning rate
LEARNING_DECAY_RATE = 0.95  # ""
ADAM_REG_PARAM = 1e-4       # adam optimizer regularization param
IMAGE_SIZE = 39             # face images 39x39 pixels
LOG_DIR = "./log"
FACE_DETECTOR_DATA = './code_face/data'
FACE_DETECTOR_ENGINE = './code_face/FacePartDetect.exe'
OUTPUT_BOX = 'bbox.txt'
INPUT_IMAGE = "input_image"
logging.basicConfig(filename='testing.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

if __name__ == '__main__':
    root = tkinter.Tk()
    image_file = filedialog.askopenfilename(parent=root, title='Choose a file')
    file = open(INPUT_IMAGE, "w")
    file.writelines("1")
    file.writelines(image_file)
    file.close()
    subprocess.call([FACE_DETECTOR_ENGINE, FACE_DETECTOR_DATA, INPUT_IMAGE, OUTPUT_BOX])
    validation_dataset, validation_labels, raw_validation_images = load(OUTPUT_BOX, IMAGE_SIZE, IMAGE_SIZE)

    logging.info("Creating validation data.")
    validation_dataset = validation_dataset[:VALIDATION_SIZE, ...]
    validation_labels = validation_labels[:VALIDATION_SIZE]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./trained/model.ckpt-1999000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./trained/'))
        graph = tf.get_default_graph()
        train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        # input node to training CNN
        train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        train_data_node = graph.get_tensor_by_name("Placeholder:0")
        train_labels_node = graph.get_tensor_by_name("Placeholder_1:0")
        train_prediction = graph.get_tensor_by_name("fc2/fc:0")
        base_index = 0

        test_batch = validation_dataset[base_index:base_index + BATCH_SIZE]
        validate_output = sess.run(train_prediction, feed_dict={train_data_node: test_batch})

        img = raw_validation_images[base_index]
        draw = ImageDraw.Draw(img)
        label_result = validate_output[0]
        draw.point((int(label_result[0]), int(label_result[1])), fill=255)
        draw.point((int(label_result[2]), int(label_result[3])), fill=255)
        draw.point((int(label_result[4]), int(label_result[5])), fill=255)
        draw.point((int(label_result[6]), int(label_result[7])), fill=255)
        draw.point((int(label_result[8]), int(label_result[9])), fill=255)

        img.save(str(base_index) + ".png")
        img.show()


