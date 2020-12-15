import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def process_numpy(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(512).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def process_directory(path, batch_size=32, image_size=(256, 256)):
    dataset = keras.preprocessing.image_dataset_from_directory(directory=path, batch_size=batch_size, image_size=image_size, label_mode=None).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def scale(image):
    return (image - 127.5) / 127.5


def rescale(image):
    return (image * 127.5 + 127.5).astype(np.uint8)


def _process(image):
    img = tf.cast(image, tf.float32)
    img = scale(img)
    img = tf.image.random_flip_left_right(img)
    return img
