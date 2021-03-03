import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random

from stylegan import StyleGAN
from layers import AugmentLayer, AdaInstanceNormalization, Conv2DMod
from utils import layer_dict, cal_gp, get_perceptual_func

perceptual = get_perceptual_func("ResNet50")


def up(i, size):
    return keras.layers.UpSampling2D(size=size, interpolation="bilinear")(i)
    # return keras.layers.Conv2DTranspose(filters=i.shape[-1], kernel_size=3, strides=size, padding="SAME", kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(i)


def conv(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=use_bias)


def act_layer(i):
    return keras.layers.LeakyReLU(0.1)(i)


def norm_layer(i):
    return keras.layers.BatchNormalization()(i)


class StyleGAN2(StyleGAN):
    def build(self, input_shape):
        self.M = keras.Sequential([keras.layers.InputLayer(input_shape=(self.latent_dim,))])
        for _ in range(8):
            self.M.add(keras.layers.Dense(self.latent_dim, kernel_initializer="he_normal", activation="swish"))

        style_list = []
        inp = keras.layers.Input(shape=(self.latent_dim,))
        noise = keras.layers.Input(shape=input_shape)
        outs = []
        self.n = 0
        for i, f in layer_dict.items():
            if i == min(layer_dict.keys()):
                style = keras.layers.Input(shape=(self.latent_dim,))
                style_list.append(style)
                self.n += 1
                o = inp[:, tf.newaxis, tf.newaxis, :]
                o = up(o, size=i)
                o = Conv2DMod(f * self.filter_num, kernel_size=3, strides=1)([o, style])
                noise_crop = keras.layers.Cropping2D((input_shape[0]-i)//2)(noise)
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                o = Conv2DMod(f * self.filter_num, kernel_size=3, strides=1)([o, style])
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                rgb = Conv2DMod(input_shape[-1], kernel_size=1, strides=1, demod=False)([o, style])
                outs.append(up(rgb, size=input_shape[0]//i))
            else:
                style = keras.layers.Input(shape=(self.latent_dim,))
                style_list.append(style)
                self.n += 1
                o = up(o, size=2)
                o = Conv2DMod(f * self.filter_num, kernel_size=3, strides=1)([o, style])
                noise_crop = keras.layers.Cropping2D((input_shape[0]-i)//2)(noise)
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                o = Conv2DMod(f * self.filter_num, kernel_size=3, strides=1)([o, style])
                o_n = conv(f * self.filter_num, kernel_size=1, strides=1)(noise_crop)
                o = keras.layers.Add()([o, o_n])
                o = act_layer(o)
                rgb = Conv2DMod(input_shape[-1], kernel_size=1, strides=1, demod=False)([o, style])
                outs.append(up(rgb, size=input_shape[0]//i))
            if i == input_shape[0]:
                self.m = self.n
                break
        o = keras.layers.Add()(outs)
        # o = keras.layers.Activation("tanh")(o)
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
                res = conv(f * self.filter_num, kernel_size=3, strides=2)(o)
                o = conv(f * self.filter_num, kernel_size=3, strides=2)(o)
                o = norm_layer(o)
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=3, strides=1)(o)
                o = norm_layer(o)
                o = keras.layers.Add()([res, o])
                o = act_layer(o)
        o = keras.layers.Flatten()(o)
        self.dis = keras.Model(inputs=img, outputs=o)
        self.perform_gp = True
        self.perform_pl = True
        self.pl_mean = 0
        self.pl_length = 0.
