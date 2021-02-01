import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import math

from .dcgan import DCGAN


class SpectralGAN(DCGAN):
    def _create_model(self):
        super(SpectralGAN, self)._create_model()
        img = keras.layers.Input(shape=self.img_shape)
        o = img
        for i, f in reversed(self.layer_dict.items()):
            if i < self.img_shape[0]:
                o = tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
        o = keras.layers.GlobalAveragePooling2D()(o)
        o = keras.layers.Dense(1)(o)
