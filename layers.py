import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa


class NoiseLayer(keras.layers.Layer):
    def __init__(self):
        super(NoiseLayer, self).__init__()

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(1,), initializer=keras.initializers.Zeros())

    def call(self, inputs, **kwargs):
        if kwargs["training"]:
            noise = tf.random.normal(shape=inputs.shape)
            inputs = inputs + self.weight * noise
        return inputs


class GLULayer(keras.layers.Layer):
    def __init__(self, c):
        super(GLULayer, self).__init__()
        self.c = c

    def call(self, inputs, **kwargs):
        c = self.c
        w = inputs[:, :, :, :c]
        v = keras.activations.get("sigmoid")(inputs[:, :, :, c:])
        return v * w


class AugmentLayer(keras.layers.Layer):
    def __init__(self):
        super(AugmentLayer, self).__init__()

    def call(self, inputs, **kwargs):
        bright = tf.random.uniform(shape=tf.shape(inputs), minval=-0.5, maxval=0.5)
        satura = tf.random.uniform(shape=tf.shape(inputs), maxval=2.0)
        contrast = tf.random.uniform(shape=tf.shape(inputs))
        img = inputs + bright
        img_mean = tf.reduce_mean(img, axis=-1, keepdims=True)
        img = (img - img_mean) * satura + img_mean
        img_mean = tf.reduce_mean(img, axis=[1, 2, 3], keepdims=True)
        img = (img - img_mean) * contrast + img_mean
        return img
