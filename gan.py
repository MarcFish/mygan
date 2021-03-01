import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from utils import EMACallback, process_numpy, ShowCallback
from layers import AugmentLayer


class GAN(keras.Model):
    def __init__(self, latent_dim=100, filter_num=16, ema_steps=100, perform_gp=False):
        self.latent_dim = latent_dim
        self.filter_num = filter_num
        self.ema_steps = ema_steps
        self.perform_gp = perform_gp
        super(GAN, self).__init__()

    def build(self, input_shape):
        self.gen = keras.Sequential([
            keras.layers.InputLayer([self.latent_dim]),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(np.prod(input_shape), activation="tanh"),
            keras.layers.Reshape(input_shape),
        ])
        self.dis = keras.Sequential([
            AugmentLayer(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dense(1),
        ])

    def _gen_loss(self, real, fake):
        return self.compiled_loss(tf.ones_like(fake), fake)

    def _dis_loss(self, real, fake):
        real_loss = self.compiled_loss(tf.ones_like(real), real)
        fake_loss = self.compiled_loss(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss / 2.0

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, images):
        noise = tf.random.normal((images.shape[0], self.latent_dim))
        with tf.GradientTape(persistent=True) as tape:
            gen_img = self.gen(noise, training=True)
            fake = self.dis(gen_img, training=True)
            real = self.dis(images, training=True)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)

            if self.perform_gp:
                alpha = tf.random.uniform((images.shape[0], 1, 1, 1))
                average_samples = (alpha * images) + (1-alpha) * gen_img
                gradients = tf.gradients(self.dis(average_samples), average_samples)[0]
                gradients_sqr = tf.math.square(gradients)
                gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
                gradients_l2_norm = tf.math.sqrt(gradients_sqr_sum)
                gradients_penalty = tf.math.square(1 - gradients_l2_norm) * 10.
                dis_loss += tf.reduce_mean(gradients_penalty)

        gen_gradients = tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.dis.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.d_optimizer.apply_gradients(zip(dis_gradients, self.dis.trainable_variables))
        return {"gen_loss": gen_loss, "dis_loss": dis_loss}

    def call(self, inputs, training=None, mask=None):
        return self.gen(inputs)

    def summary(self, **kwargs):
        self.gen.summary(**kwargs)
        self.dis.summary(**kwargs)


if __name__ == "__main__":
    anime_path = "E:/project/data/process/picture/anime64.npy"
    anime_dataset = process_numpy(np.load(anime_path), batch_size=64)

    ema = EMACallback()
    show = ShowCallback()
    model = GAN()
    model.build((64, 64, 3))
    model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(anime_dataset, epochs=10, callbacks=[ema, show])
