import tensorflow.keras as keras
import tensorflow_addons as tfa
import math
from .gan import GAN


class DCGAN(GAN):
    def __init__(self, **kwargs):
        self.layer_dict = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        super(DCGAN, self).__init__(**kwargs)

    def _create_model(self):
        noise = keras.layers.Input(shape=(self.latent_dim,))
        for i, f in self.layer_dict.items():
            if i == 4:
                o = keras.layers.Dense(4 * 4 * f * self.filter_num)(noise)
                o = keras.layers.Reshape((4, 4, f * self.filter_num))(o)
            else:
                o = keras.layers.Conv2DTranspose(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                                 kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
            if i == self.img_shape[0] // 2:
                break

        o = keras.layers.Conv2DTranspose(filters=self.img_shape[-1], kernel_size=5, strides=2, padding='SAME',
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                                         activation="tanh")(o)
        self.gen = keras.Model(inputs=noise, outputs=o)

        img = keras.layers.Input(shape=self.img_shape)
        o = img
        for i, f in reversed(self.layer_dict.items()):
            if i < self.img_shape[0]:
                o = keras.layers.Conv2D(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
        o = keras.layers.GlobalAveragePooling2D()(o)
        o = keras.layers.Dense(1)(o)

        self.dis = keras.Model(inputs=img, outputs=o)
