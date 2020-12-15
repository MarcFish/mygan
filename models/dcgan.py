import tensorflow.keras as keras
import tensorflow_addons as tfa
import math
from .gan import GAN


class DCGAN(GAN):
    def _create_model(self, norm="batch"):
        self.gen = keras.Sequential()
        self.dis = keras.Sequential()
        if norm == "batch":
            norm_layer = keras.layers.BatchNormalization
        else:
            norm_layer = tfa.layers.InstanceNormalization
        layer_size = list(range(int(math.log2(self.img_shape[0]) - 1)))
        for i in layer_size:
            f = self.filter_num * (2 ** (len(layer_size) - i))
            if i == 0:
                self.gen.add(keras.layers.Dense(4*4*f))
                self.gen.add(keras.layers.LeakyReLU(0.2))
                self.gen.add(keras.layers.Reshape((4, 4, f)))
            else:
                self.gen.add(keras.layers.Conv2DTranspose(filters=f, kernel_size=5, strides=2, padding="SAME", use_bias=False))
                self.gen.add(norm_layer())
                self.gen.add(keras.layers.LeakyReLU(0.2))
        self.gen.add(keras.layers.Conv2D(filters=self.img_shape[-1], kernel_size=5, strides=1, activation="tanh", padding='SAME', use_bias=False))
        for i in layer_size:
            f = self.filter_num * (2 ** (len(layer_size) - i))
            self.dis.add(keras.layers.Conv2D(filters=f, kernel_size=5, strides=2, padding="SAME"))
            self.dis.add(keras.layers.BatchNormalization())
            self.dis.add(keras.layers.LeakyReLU(0.2))
        self.dis.add(keras.layers.GlobalAveragePooling2D())
        self.dis.add(keras.layers.Dense(1))
