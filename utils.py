import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from pathlib import Path
from PIL import Image
import tensorflow_addons as tfa


layer_dict = {2: 32, 4: 16, 8: 8, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}


def process_numpy(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(512).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def process_directory(path, batch_size=32, img_shape=(512, 512, 3)):
    def path_to_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img)
        img.set_shape(img_shape)
        return img

    image_paths = [str(x) for x in Path(path).iterdir()]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(path_to_image, tf.data.experimental.AUTOTUNE).shuffle(512).padded_batch(batch_size=batch_size, drop_remainder=True).map(_process, tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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


class ShowCallback(keras.callbacks.Callback):
    def __init__(self, show=False, save_steps=200):
        super(ShowCallback, self).__init__()
        self.imgs = []
        self.show = show
        self.save_steps = save_steps
        if self.show:
            plt.figure()

    def on_train_begin(self, logs=None):
        self.noise = tf.random.normal(shape=(16, self.model.latent_dim))

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_steps == 0:
            img = self.model.gen(self.noise, training=False).numpy()  # 16, 64, 64, 3
            img = rescale(img)
            w = img.shape[1]
            img_all = np.ndarray(shape=(4 * w, 4 * w, 3), dtype=np.uint8)
            for i in range(16):
                row = (i // 4) * w
                col = (i % 4) * w
                img_all[row:row+w, col:col+w, :] = img[i]

            self.imgs.append(img_all)
            Image.fromarray(img_all).save(f"{batch}.png")
            if self.show:
                plt.imshow(img_all)
                plt.pause(0.1)

    def make_gif(self, name="test.gif", delta=0.01):
        imageio.mimsave(name, self.imgs, "GIF", duration=delta)
        return


class EMACallback(keras.callbacks.Callback):
    def __init__(self, tau=0.9, update_step=100):
        super(EMACallback, self).__init__()
        self.tau = tau
        self.update_step = update_step
        self.step = 0

    def on_train_begin(self, logs=None):
        self.ema = keras.models.clone_model(self.model.gen)
        self.ema.build(self.model.gen.input_shape)
        for w, wt in zip(self.model.gen.weights, self.ema.weights):
            wt.assign(w)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.update_step == 0:
            for w, wt in zip(self.model.gen.weights, self.ema.weights):
                wt.assign(self.tau * wt + (1-self.tau) * w)


class SaveCallback(keras.callbacks.Callback):
    def __init__(self, save_step=5):
        super(SaveCallback, self).__init__()
        self.save_step = save_step

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './ckpts', max_to_keep=5)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_step == 0:
            self.manager.save(epoch)


def convt(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return tfa.layers.SpectralNormalization(
        keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=use_bias))


def conv(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return tfa.layers.SpectralNormalization(
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=use_bias))
