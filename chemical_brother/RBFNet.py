from tensorflow.keras.models import Model
from tensorflow.keras import layers, activations

from chemical_brother.RBFLayer import RBFLayer


class RBFNet(Model):
    def __init__(self, output_dim=10):
        super().__init__()
        self.rbf_layer = RBFLayer(256, 0.2)
        self.dense = layers.Dense(units=output_dim, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        x = self.rbf_layer(inputs, *args, **kwargs)
        x = activations.relu(x)
        return self.dense(x)
