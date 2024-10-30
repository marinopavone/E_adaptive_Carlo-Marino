from tensorflow.keras.models import Model
from tensorflow.keras import layers

from chemical_brother.RBF.rbflayer import RBFLayer


class RBFNet(Model):
    def __init__(self, output_dim=10, gamma=0.05):
        super().__init__()
        self.rbf_layer = RBFLayer(num_centers=output_dim, gamma=gamma)
        self.dense = layers.Dense(units=output_dim, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        x = self.rbf_layer(inputs, *args, **kwargs)
        # x = activations.relu(x)
        return self.dense(x)
