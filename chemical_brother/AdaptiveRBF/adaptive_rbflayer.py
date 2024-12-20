import tensorflow as tf


class AdaptiveRBFLayer(tf.keras.layers.Layer):
    def __init__(self, num_centers, initial_gamma=1.0, activation=None):
        super().__init__()
        self.centers = None
        self.gamma = None
        self.num_centers = num_centers
        self.initial_gamma = initial_gamma
        if isinstance(activation, str):
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = activation

    def build(self, input_shape):
        self.centers = self.add_weight(
            shape=(self.num_centers, input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5),
            trainable=True,
        )
        self.gamma = self.add_weight(
            shape=(self.num_centers,),
            initializer=tf.keras.initializers.Constant(self.initial_gamma),
            trainable=True,
        )

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        res = tf.exp(-self.gamma * l2)
        if self.activation:
            res = self.activation(res)
        return res
