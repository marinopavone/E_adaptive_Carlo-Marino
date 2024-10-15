from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
        self.centers = None

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=(int(input_shape[1]), self.units),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        diff = K.expand_dims(inputs) - self.centers
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
