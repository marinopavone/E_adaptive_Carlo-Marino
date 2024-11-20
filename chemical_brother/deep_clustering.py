from base64 import decode

from keras import Model
from keras.src import layers
import tensorflow as tf

from chemical_brother.AdaptiveRBF.adaptive_rbflayer import AdaptiveRBFLayer


class DeepClustering(Model):
    def __init__(self, reconstrunction_dim=1000, n_classes=1000, centroids_per_class=1):
        super(DeepClustering, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(reconstrunction_dim, activation="relu"),
            ]
        )

        self.rbf_layer = AdaptiveRBFLayer(
            num_centers=n_classes * centroids_per_class, activation="softmax"
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        clustering = self.rbf_layer(encoded)
        return clustering, decoded
