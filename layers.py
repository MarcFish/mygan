import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa


class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation="leaky_relu", kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(DenseLayer, self).__init__()
        self.layers = keras.Sequential()
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.units = units
        self.activation = tf.nn.leaky_relu

    def build(self, input_shape):
        for unit in self.units:
            self.layers.add(keras.layers.Dense(units=unit, activation=self.activation,
                                               kernel_initializer=self.kernel_initializer,
                                               bias_initializer=self.bias_initializer))
            self.layers.add(keras.layers.BatchNormalization())

    def call(self, inputs, training=True):
        o = self.layers(inputs, training=training)
        return o


class Conv2DT(keras.layers.Layer):
    def __init__(self, filters=128, kernel_size=3, strides=2, padding="SAME", activation="elu", use_norm=True):
        super(Conv2DT, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.use_norm = use_norm

    def build(self, input_shape):
        self.convt = keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size,
                                                  strides=self.strides, padding=self.padding)
        if self.use_norm:
            self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        o = self.convt(inputs)
        if self.use_norm:
            o = self.norm(o)
        o = self.activation(o)
        return o


class ResConv2DT(keras.layers.Layer):
    def __init__(self, filters=128, kernel_size=3, strides=1, padding="SAME", activation="elu", use_norm=True):
        super(ResConv2DT, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.use_norm = use_norm

    def build(self, input_shape):
        self.res = keras.Sequential()
        self.res.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=1, strides=self.strides, padding=self.padding,
                                                  activation='elu'))
        self.res.add(
            keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding=self.padding, activation='elu'))
        self.res.add(keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=1, strides=1, padding=self.padding))
        if self.strides != 1 or self.filters != input_shape[-1]:
            self.use_h = True
            self.h = keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=1, strides=self.strides,
                                                  padding=self.padding)
        else:
            self.use_h = False
        if self.use_norm:
            self.norm = tfa.layers.InstanceNormalization()

    def call(self, inputs, training=True):
        o = self.res(inputs)
        if self.use_h:
            inputs = self.h(inputs)
        o = inputs + o
        if self.use_norm:
            o = self.norm(o)
        o = self.activation(o)
        return o


class Conv2D(keras.layers.Layer):
    def __init__(self, filters=128, kernel_size=3, strides=2, padding="SAME", activation="elu", use_norm=True):
        super(Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.use_norm = use_norm

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                        padding=self.padding)
        if self.use_norm:
            self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        o = self.conv(inputs)
        if self.use_norm:
            o = self.norm(o)
        o = self.activation(o)
        return o


class ResConv2D(keras.layers.Layer):
    def __init__(self, filters=128, kernel_size=3, strides=1, padding="SAME", activation="elu", use_norm=True):
        super(ResConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.use_norm = use_norm

    def build(self, input_shape):
        self.res = keras.Sequential()
        self.res.add(keras.layers.Conv2D(filters=64, kernel_size=1, strides=self.strides, padding=self.padding,
                                         activation=self.activation))
        self.res.add(
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding=self.padding, activation=self.activation))
        self.res.add(keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=1, padding=self.padding))
        if self.strides != 1 or self.filters != input_shape[-1]:
            self.use_h = True
            self.h = keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=self.strides, padding=self.padding)
        else:
            self.use_h = False
        if self.use_norm:
            self.norm = tfa.layers.InstanceNormalization()

    def call(self, inputs, training=True):
        o = self.res(inputs)
        if self.use_h:
            inputs = self.h(inputs)
        o = inputs + o
        if self.use_norm:
            o = self.norm(o)
        o = self.activation(o)
        return o


class AdaIN(keras.layers.Layer):
    def __init__(self, filters=128, padding="SAME", strides=1, activation="elu"):
        super(AdaIN, self).__init__()
        self.filters = filters
        self.padding = padding
        self.strides = strides
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        def crop_to_fit(x):
            height = x[1].shape[1]
            width = x[1].shape[2]
            return x[0][:, :height, :width, :]

        self.gamma_dense = keras.layers.Dense(self.filters)
        self.beta_dense = keras.layers.Dense(self.filters)
        self.delta_dense = keras.layers.Dense(self.filters, kernel_initializer="zeros")
        self.crop = keras.layers.Lambda(crop_to_fit)
        self.conv = Conv2DT(filters=self.filters, strides=self.strides, activation=None, use_norm=False)

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