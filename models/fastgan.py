import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random

from .gan import GAN
from ..layers import NoiseLayer, GLULayer, AugmentLayer
from ..utils import apply_augment, get_perceptual_func


perceptual = get_perceptual_func()


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
        assert self.img_shape[0] >= 256
        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*self.filter_num)

        style_noise = keras.layers.Input(shape=(self.latent_dim,))
        noise = style_noise[:, tf.newaxis, tf.newaxis, :]
        o = convt(nfc[4]*2, kernel_size=4, strides=1, padding="VALID")(noise)
        o = keras.layers.BatchNormalization()(o)
        feat_4 = GLULayer(nfc[4])(o)
        feat_8 = UpBlockComp(nfc[8], feat_4)
        feat_16 = UpBlock(nfc[16], feat_8)
        feat_32 = UpBlockComp(nfc[32], feat_16)
        feat_64 = SEBlock(nfc[64], [feat_4, UpBlock(nfc[64], feat_32)])
        feat_128 = SEBlock(nfc[128], [feat_8, UpBlockComp(nfc[128], feat_64)])

        o128 = keras.layers.Activation("tanh")((conv(self.img_shape[-1], kernel_size=3, strides=1)(feat_128)))
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

        img = keras.layers.Input(shape=self.img_shape)
        img_128 = keras.layers.Input(shape=(128, 128, self.img_shape[-1]))
        img = AugmentLayer()(img)
        img_128 = AugmentLayer()(img_128)
        if self.img_shape[0] == 256:
            feat_2 = conv(nfc[256], 3, 1)(img)
            feat_2 = keras.layers.LeakyReLU(0.2)(feat_2)
        if self.img_shape[0] == 512:
            feat_2 = conv(nfc[512], 4, 2)(img)
            feat_2 = keras.layers.LeakyReLU(0.2)(feat_2)
        if self.img_shape[0] == 1024:
            feat_2 = conv(nfc[1024], 4, 2)(img)
            feat_2 = keras.layers.LeakyReLU(0.2)(feat_2)
            feat_2 = conv(nfc[1024], 4, 2)(feat_2)
            feat_2 = keras.layers.BatchNormalization()(feat_2)
            feat_2 = keras.layers.LeakyReLU(0.2)(feat_2)

        feat_4 = DownBlockComp(nfc[256], feat_2)
        feat_8 = DownBlockComp(nfc[128], feat_4)
        feat_16 = DownBlockComp(nfc[64], feat_8)
        feat_16 = SEBlock(nfc[64], [feat_2, feat_16])
        feat_32 = DownBlockComp(nfc[32], feat_16)
        feat_32 = SEBlock(nfc[32], [feat_4, feat_32])

        feat_last = DownBlockComp(nfc[16], feat_32)
        feat_last = SEBlock(nfc[16], [feat_8, feat_last])

        rf_0 = conv(nfc[8], 1, 1, padding="VALID")(feat_last)
        rf_0 = keras.layers.BatchNormalization()(rf_0)
        rf_0 = keras.layers.LeakyReLU(0.2)(rf_0)
        rf_0 = conv(1, 4, 1, padding="VALID")(rf_0)
        rf_0 = keras.layers.Flatten()(rf_0)

        feat_small = conv(nfc[256], 4, 2)(img_128)
        feat_small = keras.layers.LeakyReLU(0.2)(feat_small)
        feat_small = DownBlock(nfc[128], feat_small)
        feat_small = DownBlock(nfc[64], feat_small)
        feat_small = DownBlock(nfc[32], feat_small)

        rf_1 = conv(1, 4, 1, padding="VALID")(feat_small)
        rf_1 = keras.layers.Flatten()(rf_1)

        dis_out = keras.layers.Concatenate(axis=-1)([rf_0, rf_1])
        self.fake_dis = keras.Model(inputs=[img,img_128], outputs=dis_out)

        def decoder(inputs):
            o = tfa.layers.AdaptiveAveragePooling2D(8)(inputs)
            o = UpBlock(nfc[16], o)
            o = UpBlock(nfc[32], o)
            o = UpBlock(nfc[64], o)
            o = UpBlock(nfc[128], o)
            o = conv(self.img_shape[-1], 3, 1)(o)
            o = keras.layers.Activation("tanh")(o)
            return o

        rec_img_big = decoder(feat_last)
        rec_img_small = decoder(feat_small)
        # TODO: part
        self.real_dis = keras.Model(inputs=[img, img_128], outputs=[dis_out, rec_img_big, rec_img_small])
        self.dis = [self.real_dis, self.fake_dis]

    @tf.function
    def _train_step(self, images):
        style_noise = tf.random.uniform((self.batch_size, self.latent_dim))
        part = random.randint(0, 3)
        real_img = apply_augment(images)
        with tf.GradientTape(persistent=True) as tape:
            gen_img, gen_img128 = self._gen(style_noise, training=True)
            fake = self.fake_dis([gen_img, gen_img128])
            real, rec_img_big, rec_img_small = self.real_dis([real_img, tf.image.resize(real_img, (128, 128))])
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)
            dis_loss += perceptual(rec_img_big, tf.image.resize(real_img, (rec_img_big.shape[1],rec_img_big.shape[1])))
            dis_loss += perceptual(rec_img_small, tf.image.resize(real_img, (rec_img_small.shape[1], rec_img_small.shape[1])))

        gen_gradients = tape.gradient(gen_loss, self._gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.real_dis.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.dis_opt.apply_gradients(zip(dis_gradients, self.real_dis.trainable_variables))
        return gen_loss, dis_loss