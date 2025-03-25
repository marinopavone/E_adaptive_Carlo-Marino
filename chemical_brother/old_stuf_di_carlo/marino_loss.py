import tensorflow as tf
from sklearn.metrics import silhouette_samples, silhouette_score

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, X_dim, Z_dim, factor=1.0, name="custom_loss"):
        """
        Initialize the custom loss.

        Args:
            factor: A scaling factor for the loss (default 1.0).
            name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.factor = factor
        self.X_dim = X_dim
        self.Z_dim = Z_dim


    def call(self, G_true, net_out ):

        Z= net_out[:, 0 : self.Z_dim]
        X_rec = net_out[:, self.Z_dim+1 : self.Z_dim+1 + self.X_dim ] # mse_loss = tf.reduce_mean(tf.square(X_true - X_pred))

        X_true = G_true[:, 0 : self.X_dim]
        y_true = G_true[:, -1]


        similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)

        # Normalize predictions (if necessary)
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=1)

        # Compute losses (e.g., based on target constraints like margin)
        loss = tf.reduce_mean(tf.maximum(0.0, 1 - y_true * similarity_matrix))

        loss_MSE = tf.reduce_mean(tf.square(X_true - X_rec))
        # sil_loss = silhouette_score(Z, y_true)

        return self.factor * loss + (1-sil_loss)

# def marino_silhouette_loss(self, y_true, net_out ):
#     Z_pred = self.model.encode(X_true)
#
#     sil = sillhouette(Z_pred, y_true)
#     # Calculate the mean distance between points in the same cluster
#     mse_loss = tf.reduce_mean(tf.square(X_true - X_pred))

    # combined = 0.6 * mse_loss + 0.4 * Silhouette  # Adjust weights as needed
    # same_cluster_dist = tf.reduce_mean(
    #     tf.map_fn(
    #         lambda z: tf.reduce_mean(
    #             tf.norm(z - tf.gather(Z_pred, tf.where(tf.equal(y_true, z))), axis=1)
    #         ),
    #         X_true,
    #     )
    # )
    #
    # # Calculate the mean distance between points in different clusters
    # diff_cluster_dist = tf.reduce_mean(
    #     tf.map_fn(
    #         lambda z: tf.reduce_mean(
    #             tf.norm(
    #                 z - tf.gather(Z_pred, tf.where(tf.not_equal(X_true, z))), axis=1
    #             )
    #         ),
    #         X_true,
    #     )
    # )
#     #
#     # # Calculate the silhouette score
#     # silhouette = (same_cluster_dist - diff_cluster_dist) / tf.maximum(
#     #     same_cluster_dist, diff_cluster_dist
#     # )
#
#     # Silhouette = 1.0 - silhouette
#     return combined
#