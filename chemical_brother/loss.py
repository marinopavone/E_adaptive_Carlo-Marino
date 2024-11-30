import tensorflow as tf


class SilhouetteLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        Silhouette loss function.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Silhouette loss value.
        """
        # Calculate the mean distance between points in the same cluster
        same_cluster_dist = tf.reduce_mean(
            tf.map_fn(
                lambda x: tf.reduce_mean(
                    tf.norm(
                        x - tf.gather(y_pred, tf.where(tf.equal(y_true, x))), axis=1
                    )
                ),
                y_true,
            )
        )

        # Calculate the mean distance between points in different clusters
        diff_cluster_dist = tf.reduce_mean(
            tf.map_fn(
                lambda x: tf.reduce_mean(
                    tf.norm(
                        x - tf.gather(y_pred, tf.where(tf.not_equal(y_true, x))), axis=1
                    )
                ),
                y_true,
            )
        )

        # Calculate the silhouette score
        silhouette = (same_cluster_dist - diff_cluster_dist) / tf.maximum(
            same_cluster_dist, diff_cluster_dist
        )

        # Return the negative silhouette score as the loss
        return 1.0 - silhouette


class CorelLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        A simple example of COREL Loss function.
        y_true: Ground truth values (e.g., labels or targets).
        y_pred: Predicted embeddings or scores.
        """
        # Compute pairwise distances or similarities
        similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)

        # Normalize predictions (if necessary)
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=1)

        # Compute losses (e.g., based on target constraints like margin)
        loss = tf.reduce_mean(tf.maximum(0.0, 1 - y_true * similarity_matrix))

        return loss
