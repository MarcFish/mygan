import tensorflow as tf
import tensorflow.keras as keras
import math
import numpy as np

from .gan import GAN
from ..layers import AdaIN


class StyleGAN(GAN):
    def _create_model(self):
        self.constant = tf.constant(tf.random.normal(shape=(1,4,4,512)))
        self.batch_constant = tf.repeat(self.constant,repeats=self.batch_size,axis=0)
        self.map = keras.Sequential([
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
        ])
        inp_noise = keras.layers.Input(shape=(*self.img_shape[0:2],1))
        style_noise = keras.layers.Input(shape=(self.latent_dim,))
        style_inp = self.map(style_noise)
        constant_input = keras.layers.Input(shape=(4,4,512))
        x = constant_input
        layer_size = list(range(int(math.log2(self.img_shape[0])-2)))
        for i in layer_size:
            x = AdaIN(128*(len(layer_size)-i))([x,style_inp,inp_noise])
            x = AdaIN(128*(len(layer_size)-i), strides=2)([x,style_inp, inp_noise])
        x = keras.layers.Conv2D(filters=self.img_shape[-1], kernel_size=1, strides=1, activation="tanh", padding='SAME')(x)
        self.gen = keras.Model(inputs=[constant_input, inp_noise, style_noise], outputs=x)
        dis_list = [keras.layers.Conv2D(filters=1024, kernel_size=1, strides=1,padding="SAME")]
        for i in layer_size:
            dis_list.append(keras.layers.Conv2D(filters=128*(len(layer_size)-i), kernel_size=5, strides=1,padding="SAME"))
            dis_list.append(keras.layers.LeakyReLU(0.2))
            dis_list.append(keras.layers.BatchNormalization())
            dis_list.append(keras.layers.Conv2D(filters=128*(len(layer_size)-i), kernel_size=5, strides=2,padding="SAME"))
            dis_list.append(keras.layers.LeakyReLU(0.2))
            dis_list.append(keras.layers.BatchNormalization())
        dis_list.append(keras.layers.GlobalAveragePooling2D())
        dis_list.append(keras.layers.Dense(1))

        self.dis = keras.Sequential(dis_list)

    @tf.function
    def _train_step(self, images):
        style_noise = tf.random.uniform((self.batch_size, self.latent_dim))
        inp_noise = tf.random.normal((self.batch_size, *self.img_shape[0:2], 1))
        with tf.GradientTape(persistent=True) as tape:
            gen_img = self.gen([self.batch_constant, inp_noise, style_noise], training=True)
            fake = self.dis(gen_img, training=True)
            real = self.dis(images, training=True)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)

            if self.perform_gp:
                alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
                average_samples = (alpha * images) + (1-alpha) * gen_img
                gradients = tf.gradients(self.dis(average_samples), average_samples)[0]
                gradients_sqr = tf.math.square(gradients)
                gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
                gradients_l2_norm = tf.math.sqrt(gradients_sqr_sum)
                gradients_penalty = tf.math.square(1 - gradients_l2_norm) * 10.
                dis_loss += tf.reduce_mean(gradients_penalty)

        gen_gradients = tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.dis.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.dis_opt.apply_gradients(zip(dis_gradients, self.dis.trainable_variables))
        return gen_loss, dis_loss

    def generate(self, n=16):
        style_noise = np.random.uniform(size=[n, self.latent_dim]).astype(np.float32)
        inp_noise = np.random.normal(size=[n, *self.img_shape[0:2], 1]).astype(np.float32)
        batch_constant = tf.repeat(self.constant,repeats=n,axis=0)
        image = self.gen([batch_constant, inp_noise, style_noise]).numpy()
        return image.squeeze()
