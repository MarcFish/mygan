import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random

from pergan import PerGAN
from layers import NoiseLayer, GLULayer, AugmentLayer
from utils import apply_augment, get_perceptual_func, convt, conv, layer_dict


def norm_layer(i):
    return tfa.layers.FilterResponseNormalization()(i)


def act_layer(i):
    return tfa.layers.TLU()(i)


class HrGAN(PerGAN):
    def build(self, input_shape):
        assert input_shape[0] >= 32

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

        noise = keras.layers.Input(shape=(self.latent_dim,))
        o = noise[:, tf.newaxis, tf.newaxis, :]
        o = convt(layer_dict[min(layer_dict.keys())] * self.filter_num, kernel_size=min(layer_dict.keys()), strides=1, padding="VALID")(o)
        o = norm_layer(o)
        o = act_layer(o)

        o_dict = {min(layer_dict.keys()): o}
        l_list = [min(layer_dict.keys())]
        for l, s in layer_dict.items():
            if l == min(layer_dict.keys()):
                continue
            if l <= input_shape[0]:
                l_list.append(l)
                t_dict = dict()

                for l in l_list[:-1]:
                    for l_ in l_list:
                        t_dict.setdefault(l_, [])
                        if l == l_:
                            # o = c(o_dict[l], l, l_)
                            o = o_dict[l]
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

                    o = conv(filters=layer_dict[l] * self.filter_num, kernel_size=1, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)

                    o_dict[l] = o

        o = conv(filters=input_shape[-1], kernel_size=1, strides=1)(o_dict[input_shape[0]])
        o = keras.layers.Activation("tanh")(o)

        self.gen = keras.Model(inputs=noise, outputs=o)

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
                            # o = c(o_dict[l], l, l_)
                            o = o_dict[l]
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

                    o = conv(filters=layer_dict[l] * self.filter_num, kernel_size=1, strides=1)(o)
                    o = norm_layer(o)
                    o = act_layer(o)

                    o_dict[l] = o

        # os = []
        # for l, o in o_dict.items():
        #     if l >= 32:
        #         o = conv(filters=input_shape[-1], kernel_size=1, strides=1)(o)
        #         o = norm_layer(o)
        #         o = act_layer(o)
        #         os.append(o)
        o = keras.layers.Flatten()(o_dict[min(layer_dict.keys())])
        self.dis = keras.Model(inputs=images, outputs=o)
