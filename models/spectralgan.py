import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import math

from .dcgan import DCGAN


class SpectralGAN(DCGAN):
    def _create_model(self):
        super(SpectralGAN, self)._create_model()
        layer_size = list(range(int(math.log2(self.img_shape[0]) - 1)))
        for i in layer_size:
            f = self.filter_num * (2 ** (len(layer_size) - i))
            self.dis.add(tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=f, kernel_size=5, strides=2, padding="SAME")))
            self.dis.add(keras.layers.LeakyReLU(0.2))
        self.dis.add(keras.layers.GlobalAveragePooling2D())
        self.dis.add(keras.layers.Dense(1))
