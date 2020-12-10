import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import abc
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
from ..utils import rescale


class Model(abc.ABC):
    def __init__(self, latent_dim, img_shape, lr=1e-4, l2=5e-5, batch_size=32, epochs=10, perform_gp=False):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self._create_model()
        self.perform_gp = perform_gp
        self.gen_opt = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5, beta_2=0.9)
        self.dis_opt = tfa.optimizers.AdamW(lr=self.lr, weight_decay=self.l2, beta_1=0.5, beta_2=0.9)

    @abc.abstractmethod
    def _create_model(self):
        return NotImplementedError

    @abc.abstractmethod
    def _gen_loss(self, real, fake):
        return NotImplementedError

    @abc.abstractmethod
    def _dis_loss(self, real, fake):
        return NotImplementedError

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal((self.batch_size, self.latent_dim))
        if self.perform_gp:
            alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        with tf.GradientTape(persistent=True) as tape:
            gen_img = self.gen(noise, training=True)
            fake = self.dis(gen_img, training=True)
            real = self.dis(images, training=True)
            gen_loss = self._gen_loss(real, fake)
            dis_loss = self._dis_loss(real, fake)

            if self.perform_gp:
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

    def train(self, dataset):
        for epoch in range(self.epochs):
            start = time.time()
            with tqdm(total=len(dataset),
                      bar_format="gen loss:{postfix[0][gen]}; dis loss:{postfix[0][dis]}; {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]",
                      postfix=[dict(gen=0.0,dis=0.0)]) as t:
              for image_batch in dataset:
                gen_loss, dis_loss = self._train_step(image_batch)
                t.postfix[0]["gen"] = gen_loss
                t.postfix[0]["dis"] = dis_loss
                t.update()
            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))
            self.generate_samples(epoch=epoch, show=False)
            tf.saved_model.save(self.gen, f"gen_model_{epoch}")

    @abc.abstractmethod
    def generate(self, noise=None):
        return NotImplementedError

    def generate_samples(self, epoch=0, save=True, show=False, path=None):
        img = self.generate()  # 16, 64, 64, 3
        img = rescale(img)
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
