from base64 import decode

from keras import Model
from keras.src import layers
import tensorflow as tf

from chemical_brother.AdaptiveRBF.adaptive_rbflayer import AdaptiveRBFLayer


class sil_AUTOENC(Model):
    def __init__(self, reconstrunction_dim, n_classes, layers_dim, centroids_per_class):
        super(sil_AUTOENC, self).__init__()

        self.ld = layers_dim
        Z_dim = self.ld[2]
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(self.ld[0], activation="relu"),
                layers.Dense(self.ld[1], activation="relu"),
                layers.Dense(Z_dim, activation="relu",name='output_Z'),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(self.ld[3], activation="relu"),
                layers.Dense(self.ld[4], activation="relu"),
                layers.Dense(reconstrunction_dim, activation="relu",name='output_Xrec'),
            ]
        )
        # self.rbf_layer = AdaptiveRBFLayer(name='output_RBF',
        #     num_centers=n_classes * centroids_per_class, activation="softmax",
        # )
        # self.Z = tf.keras.Sequential([layers.Dense(Z_dim, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        Z = encoded
        #classification=self.rbf_layer(encoded)
        decoded = self.decoder(encoded)
        X_rec = decoded
        concat_layer = tf.concat([Z ,X_rec], axis=-1)
        out_dict = { 'output_Z'   : Z,
                     'output_Xrec': decoded ,
                     # 'output_RBF' :classification
                     }
        return out_dict
