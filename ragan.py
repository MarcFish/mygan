import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils import layer_dict, EMACallback, process_numpy, ShowCallback
from layers import AugmentLayer
from dcgan import DCGAN


class RaGAN(DCGAN):
    def _gen_loss(self, real, fake):
        return (keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), real - tf.reduce_mean(fake)) +
                keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), fake - tf.reduce_mean(real))) / 2.0

    def _dis_loss(self, real, fake):
        return (keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real - tf.reduce_mean(fake)) +
                keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake - tf.reduce_mean(real))) / 2.0


if __name__ == "__main__":
    k = 64
    anime_path = f"./cat{k}.npy"
    anime_imgs = np.load(anime_path)
    anime_dataset = process_numpy(anime_imgs, batch_size=32)

    ema = EMACallback()
    show = ShowCallback()
    model = RaGAN()
    model.build(anime_imgs.shape[1:])
    model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(anime_dataset, epochs=10, callbacks=[ema, show])
    show.make_gif("ragan.gif")
