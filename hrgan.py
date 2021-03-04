import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random

from pergan import PerGAN
from dcgan import DCGAN
from layers import NoiseLayer, GLULayer, AugmentLayer, Conv2DMod
from utils import apply_augment, get_perceptual_func, convt, conv, layer_dict, cal_gp

perceptual = get_perceptual_func("vgg16")


def norm_layer(i):
    return tfa.layers.InstanceNormalization()(i)


def act_layer(i):
    return keras.layers.LeakyReLU(0.2)(i)


class HrGAN(PerGAN):
    def build(self, input_shape):
        assert input_shape[0] >= 32
        self.M = keras.Sequential([keras.layers.InputLayer(input_shape=(self.latent_dim,))])
        for _ in range(5):
            self.M.add(keras.layers.Dense(self.latent_dim, kernel_initializer="he_normal"))
            self.M.add(keras.layers.LeakyReLU(0.2))

        def up(i, style, noise, l, l_):
            o = keras.layers.UpSampling2D(size=l_ // l, interpolation="bilinear")(i)
            o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)([o, style])
            noise_crop = keras.layers.Cropping2D((input_shape[0] - l_)//2)(noise)
            o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            o = keras.layers.Add()([o, o_n])
            o = act_layer(o)
            # o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)([o, style])
            # o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            # o = keras.layers.Add()([o, o_n])
            # o = act_layer(o)
            return o

        def down(i, style, noise, l, l_):
            o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=l // l_)([i, style])
            noise_crop = keras.layers.Cropping2D((input_shape[0] - l_)//2)(noise)
            o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            o = keras.layers.Add()([o, o_n])
            o = act_layer(o)
            # o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)([o, style])
            # o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            # o = keras.layers.Add()([o, o_n])
            # o = act_layer(o)
            return o

        def c(i, style, noise, l, l_):
            o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)([i, style])
            noise_crop = keras.layers.Cropping2D((input_shape[0] - l_)//2)(noise)
            o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            o = keras.layers.Add()([o, o_n])
            o = act_layer(o)
            # o = Conv2DMod(layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)([o, style])
            # o_n = conv(layer_dict[l_] * self.filter_num, kernel_size=1, strides=1)(noise_crop)
            # o = keras.layers.Add()([o, o_n])
            # o = act_layer(o)
            return o

        style_list = []
        inp = keras.layers.Input(shape=(self.latent_dim,))
        noise = keras.layers.Input(shape=input_shape)
        outs = []

        o_dict = {}
        l_list = [min(layer_dict.keys())]
        for l, s in layer_dict.items():
            if l == min(layer_dict.keys()):
                style = keras.layers.Input(shape=(self.latent_dim,))
                style_list.append(style)
                o = keras.layers.Dense(self.latent_dim)(inp)
                o = o[:, tf.newaxis, tf.newaxis, :]
                o = up(o, style, noise, 1, l)
                o_dict[min(layer_dict.keys())] = o
            if l <= input_shape[0]:
                l_list.append(l)
                t_dict = dict()

                for l in l_list[:-1]:
                    for l_ in l_list:
                        t_dict.setdefault(l_, [])
                        style = keras.layers.Input(shape=(self.latent_dim,))
                        style_list.append(style)
                        if l == l_:
                            o = c(o_dict[l], style, noise, l, l_)
                            t_dict[l_].append(o)
                        elif l > l_:
                            o = down(o_dict[l], style, noise, l, l_)
                            t_dict[l_].append(o)
                        else:
                            o = up(o_dict[l], style, noise, l, l_)
                            t_dict[l_].append(o)

                for l, os in t_dict.items():
                    if len(os) == 1:
                        o = os[0]
                    else:
                        o = keras.layers.Concatenate(axis=-1)(os)

                    o = conv(filters=layer_dict[l] * self.filter_num, kernel_size=1, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)

                    o_dict[l] = o

        for l, o in o_dict.items():
            style = keras.layers.Input(shape=(self.latent_dim,))
            style_list.append(style)
            rgb = Conv2DMod(input_shape[-1], kernel_size=1, strides=1, demod=False)([o, style])
            rgb = keras.layers.UpSampling2D(size=input_shape[1]//l, interpolation="bilinear")(rgb)
            outs.append(rgb)
        o = keras.layers.Add()(outs)
        self._gen = keras.Model(inputs=[inp, noise] + style_list, outputs=o)

        style_list_n = []
        o_list = []
        inp = keras.layers.Input(shape=(self.latent_dim,))
        noise = keras.layers.Input(shape=input_shape)
        self.n = len(style_list)
        self.m = self.n
        for _ in style_list:
            style = keras.layers.Input(shape=(self.latent_dim,))
            style_list_n.append(style)
            o_list.append(self.M(style))
        o = self._gen([inp, noise] + o_list)
        self.gen = keras.Model(inputs=[inp, noise] + style_list_n, outputs=o)

        def up(i, l, l_):
            o = convt(filters=layer_dict[l_] * self.filter_num, kernel_size=3, strides=l_ // l)(i)
            o = norm_layer(o)
            o = act_layer(o)
            return o

        def down(i, l, l_):
            o = conv(filters=layer_dict[l_] * self.filter_num, kernel_size=3, strides=l // l_)(i)
            o = norm_layer(o)
            o = act_layer(o)
            return o

        def c(i, l, l_):
            o = conv(filters=layer_dict[l_] * self.filter_num, kernel_size=3, strides=1)(i)
            o = norm_layer(o)
            o = act_layer(o)
            return o

        images = keras.layers.Input(shape=input_shape)
        o = AugmentLayer()(images)

        o_dict = dict()
        l_list = []
        for l, s in reversed(layer_dict.items()):
            if l == input_shape[0]:
                o = conv(filters=layer_dict[l] * self.filter_num, kernel_size=3, strides=1)(o)
                o = norm_layer(o)
                o = act_layer(o)
                o_dict[l] = o
                l_list.append(l)
            if l < input_shape[0]:
                l_list.append(l)
                t_dict = dict()
                for l in l_list[:-1]:
                    for l_ in l_list:
                        t_dict.setdefault(l_, [])
                        if l == l_:
                            o = c(o_dict[l], l, l_)
                            # o = o_dict[l]
                            t_dict[l_].append(o)
                        elif l > l_:
                            o = down(o_dict[l], l, l_)
                            t_dict[l_].append(o)
                        else:
                            o = up(o_dict[l], l, l_)
                            t_dict[l_].append(o)

                for l, os in t_dict.items():
                    if len(os) == 1:
                        o = os[0]
                    else:
                        o = keras.layers.Concatenate(axis=-1)(os)

                    o = conv(filters=64, kernel_size=1, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)
                    o = conv(filters=64, kernel_size=3, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)
                    o = conv(filters=64, kernel_size=1, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)

                    o_dict[l] = o

        os = []
        for l, o in o_dict.items():
            if l >= 32:
                o = conv(filters=input_shape[-1], kernel_size=1, strides=1)(o)
                o = norm_layer(o)
                o = act_layer(o)
                os.append(o)
        o = keras.layers.Flatten()(o_dict[min(layer_dict.keys())])
        self.dis = keras.Model(inputs=images, outputs=[o]+os)

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
            fake, *fs = self.dis(gen_img)
            real, *rs = self.dis(images)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)
            for f, r in zip(fs, rs):
                p = perceptual(f, r)
                gen_loss += p / len(fs)
                dis_loss -= p / len(fs)
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
