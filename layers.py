import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa


class AdaIN(keras.layers.Layer):
    def __init__(self, filters=128, padding="SAME", strides=1, kernel_size=5):
        super(AdaIN, self).__init__()
        self.filters = filters
        self.padding = padding
        self.strides = strides
        self.activation = keras.layers.LeakyReLU(0.2)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        def crop_to_fit(x):
            height = x[1].shape[1]
            width = x[1].shape[2]
            return x[0][:, :height, :width, :]

        self.gamma_dense = keras.layers.Dense(self.filters)
        self.beta_dense = keras.layers.Dense(self.filters)
        self.delta_dense = keras.layers.Dense(self.filters, kernel_initializer="zeros")
        self.crop = keras.layers.Lambda(crop_to_fit)
        self.conv = keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")

    def call(self, inputs, training=True):
        inp, style, inoise = inputs
        gamma = self.gamma_dense(style)
        beta = self.beta_dense(style)
        out = self.conv(inp)
        delta = self.crop([inoise, out])
        out = out + delta
        mean, std = tf.nn.moments(out, axes=[1, 2], keepdims=True)
        y = (out - mean) / std
        g = tf.reshape(gamma, [-1, 1, 1, self.filters]) + 1.0
        b = tf.reshape(beta, [-1, 1, 1, self.filters])
        out = y * g + b
        out = self.activation(out)
        return out


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
