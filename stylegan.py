import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random
from ragan import RaGAN
from layers import AugmentLayer, AdaInstanceNormalization
from utils import layer_dict, cal_gp, get_perceptual_func

perceptual = get_perceptual_func("vgg16")


def convt(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding=padding,
                                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02))


def conv(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=use_bias)


def act_layer(i):
    return keras.layers.LeakyReLU(0.1)(i)


def norm_layer(i):
    return tfa.layers.InstanceNormalization()(i)


class StyleGAN(RaGAN):
    def build(self, input_shape):
        self.M = keras.Sequential([keras.layers.InputLayer(input_shape=(self.latent_dim,))])
        for _ in range(8):
            self.M.add(keras.layers.Dense(self.latent_dim, kernel_initializer="he_normal"))
            self.M.add(keras.layers.LeakyReLU(0.2))
        style_list = []
        inp = keras.layers.Input(shape=(self.latent_dim,))
        noise = keras.layers.Input(shape=input_shape)
        self.n = 0
        for i, f in layer_dict.items():
            if i == min(layer_dict.keys()):
                style = keras.layers.Input(shape=(self.latent_dim,))
                style_list.append(style)
                self.n += 1
                o = keras.layers.Dense(self.latent_dim)(inp)
                o = o[:, tf.newaxis, tf.newaxis, :]
                o = convt(f * self.filter_num, kernel_size=min(layer_dict.keys()),
                          strides=1, padding="VALID")(o)
                o = AdaInstanceNormalization(f * self.filter_num)([o, style])
                noise_crop = keras.layers.Cropping2D((input_shape[0]-i)//2)(noise)
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=3, strides=1)(o)
                o = AdaInstanceNormalization(f * self.filter_num)([o, style])
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
            else:
                style = keras.layers.Input(shape=(self.latent_dim,))
                style_list.append(style)
                self.n += 1
                o = convt(f * self.filter_num, kernel_size=3, strides=2)(o)
                o = AdaInstanceNormalization(f * self.filter_num)([o, style])
                noise_crop = keras.layers.Cropping2D((input_shape[0]-i)//2)(noise)
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=3, strides=1)(o)
                o = AdaInstanceNormalization(f * self.filter_num)([o, style])
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
            if i == input_shape[0]:
                self.m = self.n
                break
        o = conv(input_shape[-1], kernel_size=1, strides=1)(o)
        o = keras.layers.Activation("tanh")(o)
        self._gen = keras.Model(inputs=[inp, noise] + style_list, outputs=o)

        style_list_n = []
        o_list = []
        inp = keras.layers.Input(shape=(self.latent_dim,))
        noise = keras.layers.Input(shape=input_shape)
        for _ in style_list:
            style = keras.layers.Input(shape=(self.latent_dim,))
            style_list_n.append(style)
            o_list.append(self.M(style))
        o = self._gen([inp, noise] + o_list)
        self.gen = keras.Model(inputs=[inp, noise] + style_list_n, outputs=o)

        img = keras.layers.Input(shape=input_shape)
        o = AugmentLayer()(img)
        for i, f in reversed(layer_dict.items()):
            if i < input_shape[0]:
                o = conv(f * self.filter_num, kernel_size=3, strides=2)(o)
                o = norm_layer(o)
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=3, strides=1)(o)
                o = norm_layer(o)
                o = act_layer(o)
        o = keras.layers.Flatten()(o)
        self.dis = keras.Model(inputs=img, outputs=o)
        self.perform_gp = True
        self.perform_pl = True
        self.pl_mean = 0
        self.pl_length = 0.

    def train_step(self, images):
        style_list = [tf.random.normal((tf.shape(images)[0], self.latent_dim))] * self.m
        style_list += [tf.random.normal((tf.shape(images)[0], self.latent_dim))] * (self.n - self.m)
        noise = tf.random.normal(tf.shape(images))
        inp = tf.ones(shape=(tf.shape(images)[0], self.latent_dim))
        with tf.GradientTape(persistent=True) as tape:
            w_space = []
            for style in style_list:
                w_space.append(self.M(style))
            gen_img = self._gen([inp, noise] + w_space)
            fake = self.dis(gen_img)
            real = self.dis(images)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)
            if self.perform_gp:
                dis_loss += cal_gp(self.dis, images, gen_img, 10.)

            if self.perform_pl:
                w_space2 = []
                for style in w_space:
                    std = 0.1 / (tf.math.reduce_std(style, axis=0, keepdims=True) + 1e-8)
                    w_space2.append(style + tf.random.normal(tf.shape(style)) / (std + 1e-8))
                pl_img = self._gen([inp, noise] + w_space2)
                self.pl_length = perceptual(pl_img, gen_img)
                if self.pl_mean > 0:
                    gen_loss += tf.reduce_mean(tf.math.square(self.pl_length - self.pl_mean))

        gen_gradients = tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.dis.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.d_optimizer.apply_gradients(zip(dis_gradients, self.dis.trainable_variables))
        return {"gen_loss": gen_loss, "dis_loss": dis_loss}
