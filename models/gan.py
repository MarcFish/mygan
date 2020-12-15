import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .model import Model


class GAN(Model):
    def _create_model(self):
        self.gen = keras.Sequential([
            keras.layers.Dense(512, activation=tf.nn.leaky_relu),
            keras.layers.Dense(512, activation=tf.nn.leaky_relu),
            keras.layers.Dense(512, activation=tf.nn.leaky_relu),
            keras.layers.Dense(512, activation=tf.nn.leaky_relu),
            keras.layers.Dense(np.prod(self.img_shape), activation="tanh"),
            keras.layers.Reshape(self.img_shape),
        ])
        self.dis = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dense(1),
        ])

    def _gen_loss(self, real, fake):
        return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)

    def _dis_loss(self, real, fake):
        real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
        fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss

    def generate(self, n=16, noise=None):
        if noise is None:
            noise = np.random.normal(size=[n, self.latent_dim]).astype(np.float32)
        image = self.gen(noise, training=False).numpy()
        return image.squeeze()
