import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from .dcgan import DCGAN


class SpectralGAN(DCGAN):
    def _create_model(self):
        super(SpectralGAN, self)._create_model()
        self.dis = keras.Sequential([
            tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding="SAME")),
            keras.layers.LeakyReLU(0.2),
            tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME")),
            keras.layers.LeakyReLU(0.2),
            tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME")),
            keras.layers.LeakyReLU(0.2),
            tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME")),
            keras.layers.LeakyReLU(0.2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1)
        ])