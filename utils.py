import tensorflow as tf
import numpy as np


def process_numpy(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(tf.data.experimental.AUTOTUNE).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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
