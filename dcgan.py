import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils import layer_dict, EMACallback, process_numpy, ShowCallback
from layers import AugmentLayer
from gan import GAN


class DCGAN(GAN):
    def build(self, input_shape):
        noise = keras.layers.Input(shape=(self.latent_dim,))
        for i, f in layer_dict.items():
            if i == 4:
                o = keras.layers.Dense(4 * 4 * f * self.filter_num)(noise)
                o = keras.layers.Reshape((4, 4, f * self.filter_num))(o)
            else:
                o = keras.layers.Conv2DTranspose(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
            if i == input_shape[0] // 2:
                break

        o = keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=5, strides=2, padding='SAME',
                                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                         activation="tanh")(o)
        self.gen = keras.Model(inputs=noise, outputs=o)

        img = keras.layers.Input(shape=input_shape)
        o = AugmentLayer()(img)
        for i, f in reversed(layer_dict.items()):
            if i < input_shape[0]:
                o = keras.layers.Conv2D(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
        o = keras.layers.GlobalAveragePooling2D()(o)
        o = keras.layers.Dense(1)(o)

        self.dis = keras.Model(inputs=img, outputs=o)


if __name__ == "__main__":
    k = 64
    anime_path = f"./cat{k}.npy"
    anime_imgs = np.load(anime_path)
    anime_dataset = process_numpy(anime_imgs, batch_size=32)

    ema = EMACallback()
    show = ShowCallback()
    model = DCGAN()
    model.build(anime_imgs.shape[1:])
    model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(anime_dataset, epochs=10, callbacks=[ema, show])
    show.make_gif("dcgan.gif")
