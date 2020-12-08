import tensorflow.keras as keras
import tensorflow_addons as tfa
from .gan import GAN


class DCGAN(GAN):
    def _create_model(self):
        self.gen = keras.Sequential([
            keras.layers.Dense(4*4*1024),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Reshape((4, 4, 1024)),
            keras.layers.Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding="SAME", activation="tanh"),
        ])
        self.dis = keras.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1)
        ])
