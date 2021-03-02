import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa


class NoiseLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(1,), initializer=keras.initializers.Zeros())

    def call(self, inputs, **kwargs):
        if kwargs["training"]:
            noise = tf.random.normal(shape=inputs.shape)
            inputs = inputs + self.weight * noise
        return inputs


class GLULayer(keras.layers.Layer):
    def __init__(self, c, **kwargs):
        super(GLULayer, self).__init__(**kwargs)
        self.c = c

    def call(self, inputs, **kwargs):
        c = self.c
        w = inputs[:, :, :, :c]
        v = keras.activations.get("sigmoid")(inputs[:, :, :, c:])
        return v * w

    def get_config(self):
        config = super(GLULayer, self).get_config()
        config.update({"c": self.c})
        return config


class AugmentLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AugmentLayer, self).__init__(**kwargs)

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


class AdaInstanceNormalization(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        noise_shape = input_shape[0]
        style_shape = input_shape[1]
        self.b = keras.layers.Dense(self.units, kernel_initializer="he_normal", bias_initializer="ones")
        self.g = keras.layers.Dense(self.units, kernel_initializer="he_normal", bias_initializer="zeros")
        self.b.build(style_shape)
        self.g.build(style_shape)
        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        noise = inputs[0]
        style = inputs[1]
        beta = self.b(style)
        gamma = self.g(style)
        beta = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)
        gamma = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        mean, variance = tf.nn.moments(noise, axes=[1, 2], keepdims=True)
        normed = (noise - mean) / (tf.math.sqrt(variance) + 1e-6)
        return normed * gamma + beta

    def get_config(self):
        config = {
            "units": self.units
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


custom_objects = {name: globals()[name] for name in globals() if isinstance(globals()[name], type) and issubclass(globals()[name], keras.layers.Layer)}
