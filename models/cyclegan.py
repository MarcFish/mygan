import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
from ..utils import *
from PIL import Image

from .gan import GAN


class CycleGAN(GAN):
    def __init__(self, img_shape, lr=1e-4, l2=5e-5, batch_size=32, epochs=10, lambda_=10.):
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.lambda_ = lambda_
        self._create_model()
        self.gen_opt_s = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5)
        self.dis_opt_s = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5)
        self.gen_opt_t = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5)
        self.dis_opt_t = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5)

    @tf.function
    def _train_step(self, real_s, real_t):
        with tf.GradientTape(persistent=True) as tape:
            fake_t = self.gen_s(real_s)
            cycle_s = self.gen_t(fake_t)

            fake_s = self.gen_t(real_t)
            cycle_t = self.gen_s(fake_s)

            same_s = self.gen_s(real_s)
            same_t = self.gen_t(real_t)

            dis_real_s = self.dis_s(real_s)
            dis_real_t = self.dis_t(real_t)

            dis_fake_s = self.dis_s(fake_s)
            dis_fake_t = self.dis_t(fake_t)

            gen_s_loss = self._gen_loss(dis_real_s, dis_fake_s)
            gen_t_loss = self._gen_loss(dis_real_t, dis_real_t)

            total_cycle_loss = self._cycle_loss(real_s, cycle_s) + self._cycle_loss(real_t, cycle_t)

            total_gen_s_loss = gen_s_loss + total_cycle_loss + self._indentity_loss(real_s, same_s)
            total_gen_t_loss = gen_t_loss + total_cycle_loss + self._indentity_loss(real_t, same_t)

            dis_s_loss = self._dis_loss(dis_real_s, dis_fake_s)
            dis_t_loss = self._dis_loss(dis_real_t, dis_fake_t)

        gen_s_gradients = tape.gradient(total_gen_s_loss, self.gen_s.trainable_variables)
        gen_t_gradients = tape.gradient(total_gen_t_loss, self.gen_t.trainable_variables)
        dis_s_gradients = tape.gradient(dis_s_loss, self.dis_s.trainable_variables)
        dis_t_gradients = tape.gradient(dis_t_loss, self.dis_t.trainable_variables)

        self.gen_opt_s.apply_gradients(zip(gen_s_gradients, self.gen_s.trainable_variables))
        self.gen_opt_t.apply_gradients(zip(gen_t_gradients, self.gen_t.trainable_variables))
        self.dis_opt_s.apply_gradients(zip(dis_s_gradients, self.dis_s.trainable_variables))
        self.dis_opt_t.apply_gradients(zip(dis_t_gradients, self.dis_t.trainable_variables))

        return total_gen_s_loss, total_gen_t_loss, dis_s_loss, dis_t_loss

    def train(self, s_dataset, t_dataset):
        self.s_sample = list(s_dataset.take(1).as_numpy_iterator())[0]
        self.t_sample = list(t_dataset.take(1).as_numpy_iterator())[0]
        for epoch in range(self.epochs):
            start = time.time()
            with tqdm(total=len(s_dataset),
                      bar_format="gen s loss:{postfix[0][gen_s]}; dis s loss:{postfix[0][dis_s]}; "
                                 "gen t loss:{postfix[0][gen_t]}; dis t loss:{postfix[0][dis_t]}; "
                                 "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]",
                      postfix=[dict(gen_s=0.0, dis_s=0.0, gen_t=0.0, dis_t=0.0)]) as t:
              for s, t in zip(s_dataset, t_dataset):
                gen_s_loss, gen_t_loss, dis_s_loss, dis_t_loss = self._train_step(s, t)
                t.postfix[0]["gen_s"] = gen_s_loss
                t.postfix[0]["gen_t"] = gen_t_loss
                t.postfix[0]["dis_s"] = dis_s_loss
                t.postfix[0]["dis_t"] = dis_t_loss
                t.update()
            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))
            self.generate_samples(epoch=epoch, show=False)

    def _cycle_loss(self, real, cycle):
        return self.lambda_ * tf.reduce_mean(tf.abs(real - cycle))

    def _indentity_loss(self, real, same):
        return self.lambda_ * 0.5 * tf.reduce_mean(tf.abs(real - same))

    def _create_model(self):
        self.gen_s = self._get_unet()
        self.gen_t = self._get_unet()
        self.dis_s = keras.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=1, kernel_size=5, strides=2, padding="SAME"),
        ])
        self.dis_t = keras.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=1, kernel_size=5, strides=2, padding="SAME"),
        ])

    def _get_unet(self):
        input_s = keras.layers.Input(shape=self.img_shape)  # 64
        down_stack = [
            (keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding="SAME"),  # 32
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="SAME"),  # 16
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME"),  # 8
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="SAME"),  # 4
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="SAME"),  # 2
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
        ]

        up_stack = [
            (keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding="SAME"),  # 4
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding="SAME"),  # 8
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding="SAME"),  # 16
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
            (keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding="SAME"),  # 32
            tfa.layers.InstanceNormalization(),
            keras.layers.LeakyReLU(0.2)),
        ]

        last = keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding="SAME", activation="tanh")  # 64

        x = input_s
        skips = []
        for i, down in enumerate(down_stack):
            for d in down:
                x = d(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            for u in up:
                x = u(x)
            x = keras.layers.Concatenate()([x, skip])
        x = last(x)
        return keras.Model(inputs=input_s, outputs=x)

    def s2t(self, s):
        t = self.gen_s(s)
        return t.numpy()

    def t2s(self, t):
        s = self.gen_t(t)
        return s.numpy()

    def generate(self):
        pass

    def generate_samples(self, epoch=0, save=True, show=False, path=None):
        s = self.s_sample[:4]
        t = self.s_sample[:4]
        t_ = self.s2t(s)
        s_ = self.t2s(t)
        s = rescale(s)
        t = rescale(t)
        s_ = rescale(s_)
        t_ = rescale(t_)
        img = np.concatenate([s, t, s_, t_])
        img_all = np.ndarray(shape=(4*self.img_shape[0],4*self.img_shape[0], 3), dtype=np.uint8)
        w = self.img_shape[0]
        for i in range(16):
            row = (i // 4) * w
            col = (i % 4) * w
            img_all[row:row+w, col:col+w, :] = img[i]
        img = Image.fromarray(img_all)
        if save:
            if path is None:
                img.save(f'image_at_epoch_{epoch}.png')
            else:
                img.save(path)
        if show:
            img.show()
