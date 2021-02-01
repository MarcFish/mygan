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
    return img


def apply_augment(image:tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_saturation(image, 0.0, 2.0)
    image = tf.image.random_contrast(image, 0.0, 1.0)
    # image = tf.image.random_flip_left_right(image)
    return image


def get_perceptual_func(model="vgg16"):
    if model == "vgg16":
        m = keras.applications.VGG16(include_top=False, pooling=None)
    elif model == "DenseNet201":
        m = keras.applications.DenseNet201(include_top=False, pooling=None)
    elif model == "EfficientNetB7":
        m = keras.applications.EfficientNetB7(include_top=False, pooling=None)
    elif model == "ResNet50":
        m = keras.applications.ResNet50(include_top=False, pooling=None)
    else:
        raise Exception("model not found")

    m.trainable = False

    def perceptual(pred, target):
        m_pred = m(pred)
        m_target = m(target)
        return tf.math.sqrt(tf.reduce_sum((m_pred-m_target)**2, axis=[1, 2, 3])) / tf.cast(tf.math.reduce_prod(m_pred.shape[1:]), tf.float32)

    return perceptual


class EMA:
    def __init__(self, model: keras.Model, tau=0.9):
        self.model = keras.models.clone_model(model)
        self.model.build(model.input_shape)
        self.tau = tau

    def register(self, model: keras.Model):
        for w, wt in zip(self.model.weights, model.weights):
            w.assign(wt)

    def update(self, model: keras.Model):
        for w, wt in zip(self.model.weights, model.weights):
            w.assign(self.tau * w + (1-self.tau) * wt)
