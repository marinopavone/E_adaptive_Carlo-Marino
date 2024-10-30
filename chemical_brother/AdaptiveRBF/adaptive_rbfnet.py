from tensorflow.keras.models import Model
from tensorflow.keras import layers

from chemical_brother.AdaptiveRBF.adaptive_rbflayer import AdaptiveRBFLayer


class AdaptiveRBFNet(Model):
    def __init__(self, output_dim=10, initial_gamma=0.1):
        super().__init__()
        self.adaptive_rbf_layer = AdaptiveRBFLayer(
            num_centers=output_dim, initial_gamma=initial_gamma
        )
        # self.dense = layers.Dense(units=output_dim, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        x = self.adaptive_rbf_layer(inputs, *args, **kwargs)
        return x
