import tensorflow as tf
import tensorflow.keras as keras

from .dcgan import DCGAN


class RaGAN(DCGAN):
    def _gen_loss(self, real, fake):
        return (keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), real - tf.reduce_mean(fake)) +
                keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), fake - tf.reduce_mean(real))) / 2.0

    def _dis_loss(self, real, fake):
        return (keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real - tf.reduce_mean(fake)) +
                keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake - tf.reduce_mean(real))) / 2.0
