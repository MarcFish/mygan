import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm

from .gan import GAN
from ..layers import NoiseLayer, GLULayer


def convt(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return tfa.layers.SpectralNormalization(
        keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02), use_bias=use_bias))


def conv(filters, kernel_size, strides, use_bias=False, padding="SAME"):
    return tfa.layers.SpectralNormalization(
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02), use_bias=use_bias))


def UpBlock(out_channels, inputs):
    o = convt(out_channels*2, kernel_size=3, strides=2)(inputs)
    o = keras.layers.BatchNormalization()(o)
    o = GLULayer(out_channels)(o)
    return o


def UpBlockComp(out_channels, inputs):
    o = convt(out_channels*2, kernel_size=3, strides=2)(inputs)
    o = NoiseLayer()(o)
    o = keras.layers.BatchNormalization()(o)
    o = GLULayer(out_channels)(o)
    o = convt(out_channels*2, kernel_size=3, strides=1)(o)
    o = NoiseLayer()(o)
    o = keras.layers.BatchNormalization()(o)
    o = GLULayer(out_channels)(o)
    return o


def SEBlock(out_channels, inputs):
    small = inputs[0]
    big = inputs[1]
    o = tfa.layers.AdaptiveAveragePooling2D(4)(small)
    o = conv(out_channels, kernel_size=4, strides=1, padding="VALID")(o)
    o = keras.layers.Activation("swish")(o)
    o = conv(out_channels, kernel_size=1, strides=1, padding="VALID")(o)
    o = keras.layers.Activation("sigmoid")(o)
    return o * big


def DownBlock(out_channels, inputs):
    o = conv(out_channels, kernel_size=4, strides=2)(inputs)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU(0.2)(o)
    return o


def DownBlockComp(out_channels, inputs):
    o1 = conv(out_channels, kernel_size=4, strides=2)(inputs)
    o1 = keras.layers.BatchNormalization()(o1)
    o1 = keras.layers.LeakyReLU(0.2)(o1)
    o1 = conv(out_channels, kernel_size=3, strides=1)(o1)
    o1 = keras.layers.BatchNormalization()(o1)
    o1 = keras.layers.LeakyReLU(0.2)(o1)

    o2 = keras.layers.AveragePooling2D()(inputs)
    o2 = conv(out_channels, kernel_size=1, strides=1, padding="VALID")(o2)
    o2 = keras.layers.BatchNormalization()(o2)
    o2 = keras.layers.LeakyReLU(0.2)(o2)
    return (o1 + o2) / 2.0


class FastGAN(GAN):
    def _create_model(self):
        assert self.filter_num >= 8
        assert self.img_shape[0] >= 128
        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*self.filter_num)

        style_noise = keras.layers.Input(shape=(self.latent_dim,))
        noise = style_noise[:,tf.newaxis, tf.newaxis, :]
        o = convt(nfc[4]*2, kernel_size=4, strides=1, padding="VALID")(noise)
        o = keras.layers.BatchNormalization()(o)
        feat_4 = GLULayer(nfc[4])(o)
        feat_8 = UpBlockComp(nfc[8], feat_4)
        feat_16 = UpBlock(nfc[16], feat_8)
        feat_32 = UpBlockComp(nfc[32], feat_16)
        feat_64 = SEBlock(nfc[64], [feat_4, UpBlock(nfc[64], feat_32)])
        feat_128 = SEBlock(nfc[128], [feat_8, UpBlockComp(nfc[128], feat_64)])

        o128 = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_128)))
        if self.img_shape[0] == 128:
            o = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_128)))
        if self.img_shape[0] >= 256:
            feat_256 = SEBlock(nfc[256], [feat_16, UpBlock(nfc[256], feat_128)])
        if self.img_shape[0] == 256:
            o = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_256)))
        if self.img_shape[0] >= 512:
            feat_512 = SEBlock(nfc[512], [feat_32, UpBlockComp(nfc[512], feat_256)])
        if self.img_shape[0] == 512:
            o = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_512)))
        if self.img_shape[0] == 1024:
            feat_1024 = UpBlock(nfc[1024], feat_512)
            o = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_1024)))

        self._gen = keras.Model(inputs=style_noise, outputs=[o, o128])
        self.gen = keras.Model(inputs=style_noise, outputs=o)

        self.dis = self.gen