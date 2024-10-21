import tensorflow as tf


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, num_centers, gamma=0.1):
        super().__init__()
        self.centers = None
        self.num_centers = num_centers
        self.gamma = gamma

    def build(self, input_shape):
        self.centers = self.add_weight(
            shape=(self.num_centers, input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.gamma * l2)
