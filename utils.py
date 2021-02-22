import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import imageio

layer_dict = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}


def process_numpy(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(512).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def process_directory(path, batch_size=32, image_size=(256, 256)):
    dataset = keras.preprocessing.image_dataset_from_directory(directory=path, batch_size=batch_size, image_size=image_size, label_mode=None).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def scale(image):
    return (image - 127.5) / 127.5


def rescale(image):
    return (image * 127.5 + 127.5).astype(np.uint8)


def _process(image):
    img = tf.cast(image, tf.float32)
    img = scale(img)
    return img


def apply_augment(image:tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_saturation(image, 0.0, 2.0)
    image = tf.image.random_contrast(image, 0.0, 1.0)
    # image = tf.image.random_flip_left_right(image)
    return image


def get_perceptual_func(model="vgg16"):
    if model == "vgg16":
        m = keras.applications.VGG16(include_top=False, pooling=None)
    elif model == "DenseNet201":
        m = keras.applications.DenseNet201(include_top=False, pooling=None)
    elif model == "EfficientNetB7":
        m = keras.applications.EfficientNetB7(include_top=False, pooling=None)
    elif model == "ResNet50":
        m = keras.applications.ResNet50(include_top=False, pooling=None)
    else:
        raise Exception("model not found")

    m.trainable = False

    def perceptual(pred, target):
        m_pred = m(pred)
        m_target = m(target)
        return tf.math.sqrt(tf.reduce_sum((m_pred-m_target)**2, axis=[1, 2, 3])) / tf.cast(tf.math.reduce_prod(m_pred.shape[1:]), tf.float32)

    return perceptual


class ShowCallback(keras.callbacks.Callback):
    def __init__(self, show=False):
        super(ShowCallback, self).__init__()
        self.imgs = []
        self.show = show
        if self.show:
            plt.figure()

    def on_train_begin(self, logs=None):
        self.noise = tf.random.normal(shape=(16, self.model.latent_dim))

    def on_train_batch_begin(self, batch, logs=None):
        img = self.model.gen(self.noise).numpy()  # 16, 64, 64, 3
        img = rescale(img)
        w = img.shape[1]
        img_all = np.ndarray(shape=(4 * w, 4 * w, 3), dtype=np.uint8)
        for i in range(16):
            row = (i // 4) * w
            col = (i % 4) * w
            img_all[row:row+w, col:col+w, :] = img[i]

        self.imgs.append(img_all)
        if self.show:
            plt.imshow(img_all)
            plt.pause(0.1)

    def make_gif(self, name="test.gif", delta=0.01):
        imageio.mimsave(name, self.imgs, "GIF", duration=delta)
        return


class EMACallback(keras.callbacks.Callback):
    def __init__(self, tau=0.9, update_step=100):
        super(EMACallback, self).__init__()
        self.tau = tau
        self.update_step = update_step
        self.step = 0

    def on_train_begin(self, logs=None):
        self.ema = keras.models.clone_model(self.model.gen)
        self.ema.build(self.model.gen.input_shape)
        for w, wt in zip(self.model.gen.weights, self.ema.weights):
            wt.assign(w)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.update_step == 0:
            for w, wt in zip(self.model.gen.weights, self.ema.weights):
                wt.assign(self.tau * wt + (1-self.tau) * w)


class SaveCallback(keras.callbacks.Callback):
    def __init__(self, save_step=5):
        super(SaveCallback, self).__init__()
        self.save_step = save_step

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './ckpts', max_to_keep=5)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_step == 0:
            self.manager.save(epoch)