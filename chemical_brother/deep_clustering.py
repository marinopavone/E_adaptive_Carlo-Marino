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
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(n_classes, activation="linear"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(reconstrunction_dim, activation="sigmoid"),
            ]
        )

        self.rbf_layer = AdaptiveRBFLayer(
            num_centers=n_classes * centroids_per_class,
            initial_gamma=1.0,
            activation=None,
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        clustering = self.rbf_layer(encoded)
        return clustering, decoded
