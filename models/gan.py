import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from tqdm import tqdm
import time


from ..utils import EMA, rescale


class GAN:
    def __init__(self, latent_dim, img_shape, lr=1e-4, batch_size=32, epochs=10, ema_steps=100, perform_gp=False, filter_num=32):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.filter_num = filter_num
        self.perform_gp = perform_gp
        self.ema_steps = ema_steps
        self.seed = np.random.uniform(low=0.0, high=1.0, size=(16, self.latent_dim)).astype(np.float32)

        self._create_model()
        self.gen_opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)
        self.dis_opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)

        self.ema = EMA(self.gen)
        self.gen_ema = self.ema.model

        self.ckpt_dict = {"gen_opt": self.gen_opt,
                          "dis_opt": self.dis_opt,
                          "gen": self.gen,
                          "ema": self.gen_ema}

        self._register_ckpt()
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), **self.ckpt_dict)
        self.manager = tf.train.CheckpointManager(self.ckpt, './ckpts', max_to_keep=5)

    def _create_model(self):
        self.gen = keras.Sequential([
            keras.layers.InputLayer([self.latent_dim]),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(np.prod(self.img_shape), activation="tanh"),
            keras.layers.Reshape(self.img_shape),
        ])
        self.dis = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            tfa.layers.Maxout(128),
            keras.layers.Dense(1),
        ])

    def _gen_loss(self, real, fake):
        return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)

    def _dis_loss(self, real, fake):
        real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
        fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def _train_step(self, images):
        noise = tf.random.uniform((self.batch_size, self.latent_dim))
        with tf.GradientTape(persistent=True) as tape:
            gen_img = self.gen(noise, training=True)
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

    def train(self, dataset):
        step = 0
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
                    if step % self.ema_steps == 0:
                        if step == 0:
                            self.ema.register(self.gen)
                        self.ema.update(self.gen)
                    step += 1
            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))
            self.generate_samples(epoch=epoch, show=False)
            # self.save_model(f"gen_model_{epoch}")
            self.save_ckpt()

    def generate(self, n=16, noise=None):
        if noise is None:
            noise = np.random.uniform(low=-1.0, high=1.0, size=(n, self.latent_dim)).astype(np.float32)
        image = self.gen_ema(noise, training=False).numpy()
        return image.squeeze()

    def generate_samples(self, epoch=0, save=True, show=False, path=None):
        img = self.generate(noise=self.seed)  # 16, 64, 64, 3
        img = rescale(img)
        img_all = np.ndarray(shape=(4*self.img_shape[0], 4*self.img_shape[0], 3), dtype=np.uint8)
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

    def _register_ckpt(self):
        pass

    def save_ckpt(self):
        self.manager.save()

    def restore_ckpt(self):
        self.ckpt.restore(self.manager.latest_checkpoint)

    def save_model(self, path):
        tf.saved_model.save(self.gen_ema, path)

    def read_model(self, path):
        self.gen_ema = tf.saved_model.load(path)
