import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from tqdm import tqdm
import random

from ragan import RaGAN
from layers import NoiseLayer, GLULayer, AugmentLayer
from utils import apply_augment, get_perceptual_func, convt, conv, layer_dict


def norm_layer(i):
    return tfa.layers.InstanceNormalization()(i)


def act_layer(i):
    return keras.layers.Activation("swish")(i)


perceptual = get_perceptual_func("ResNet50")


class PerGAN(RaGAN):
    def build(self, input_shape):
        noise = keras.layers.Input(shape=(self.latent_dim,))
        for i, f in layer_dict.items():
            if i == 4:
                o = noise[:, tf.newaxis, tf.newaxis, :]
                o = convt(f * self.filter_num, kernel_size=min(layer_dict.keys()),
                          strides=1, padding="VALID")(o)
                o = norm_layer(o)
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=1, strides=1)(o)
                o = keras.layers.GaussianNoise(0.02)(o)
                o = norm_layer(o)
                o = act_layer(o)
            else:
                o = convt(f * self.filter_num, kernel_size=3, strides=2)(o)
                o = norm_layer(o)
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=1, strides=1)(o)
                o = keras.layers.GaussianNoise(0.02)(o)
                o = norm_layer(o)
                o = act_layer(o)
            if i == input_shape[0]:
                break

        o = conv(input_shape[-1], kernel_size=1, strides=1)(o)
        o = keras.layers.Activation("tanh")(o)
        self.gen = keras.Model(inputs=noise, outputs=o)

        img = keras.layers.Input(shape=input_shape)
        o = AugmentLayer()(img)
        os = [o]
        for i, f in reversed(layer_dict.items()):
            if i < input_shape[0]:
                o = conv(f*self.filter_num, kernel_size=3, strides=2)(o)
                o = norm_layer(o)
                o = act_layer(o)
                o = conv(f * self.filter_num, kernel_size=1, strides=1)(o)
                o = keras.layers.GaussianNoise(0.02)(o)
                o = norm_layer(o)
                o = act_layer(o)
                if i >= 32:
                    o_ = conv(input_shape[-1], kernel_size=1, strides=1)(o)
                    o_ = norm_layer(o_)
                    o_ = keras.layers.Activation("tanh")(o_)
                    os.append(o_)
        o = keras.layers.Flatten()(o)

        self.dis = keras.Model(inputs=img, outputs=[o]+os)

    def train_step(self, images):
        noise = tf.random.normal((tf.shape(images)[0], self.latent_dim))
        with tf.GradientTape(persistent=True) as tape:
            gen_img = self.gen(noise, training=True)
            fake, *fs = self.dis(gen_img, training=True)
            real, *rs = self.dis(images, training=True)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)
            for f, r in zip(fs, rs):
                p = perceptual(f, r)
                gen_loss += p
                dis_loss -= p
        gen_gradients = tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.dis.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.d_optimizer.apply_gradients(zip(dis_gradients, self.dis.trainable_variables))
        return {"gen_loss": gen_loss, "dis_loss": dis_loss}
