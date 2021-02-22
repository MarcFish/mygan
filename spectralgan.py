import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils import layer_dict, EMACallback, process_numpy, ShowCallback
from layers import AugmentLayer
from dcgan import DCGAN


class SpectralGAN(DCGAN):
    def build(self, input_shape):
        super(SpectralGAN, self).build(input_shape)
        img = keras.layers.Input(shape=input_shape)
        o = img
        for i, f in reversed(layer_dict.items()):
            if i < self.img_shape[0]:
                o = tfa.layers.SpectralNormalization(keras.layers.Conv2D(filters=f * self.filter_num, kernel_size=5, strides=2, padding="SAME",
                                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)
        o = keras.layers.GlobalAveragePooling2D()(o)
        o = keras.layers.Dense(1)(o)


if __name__ == "__main__":
    k = 64
    anime_path = f"./cat{k}.npy"
    anime_imgs = np.load(anime_path)
    anime_dataset = process_numpy(anime_imgs, batch_size=32)

    ema = EMACallback()
    show = ShowCallback()
    model = SpectralGAN()
    model.build(anime_imgs.shape[1:])
    model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(anime_dataset, epochs=10, callbacks=[ema, show])
    show.make_gif("spectralgan.gif")
