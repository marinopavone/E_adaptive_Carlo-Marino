import tensorflow as tf


class SilhouetteLoss(tf.keras.losses.Loss):
    def call(self, y_true, X):
        """
        Silhouette loss function.

        Args:
            y_true: True labels (integer-encoded cluster labels).
            X: Predicted feature embeddings or data points.

        Returns:
            Silhouette loss value (1.0 - silhouette score).
        """
        # Ensure y_true is a tensor of shape (batch_size,)
        y_true = tf.cast(y_true, tf.int32)
        X = tf.cast(X, tf.float32)

        unique_labels = tf.unique(y_true).y  # Get unique cluster labels
        n_clusters = tf.shape(unique_labels)[0]

        # Compute intra-cluster distances (a[i] for each point)
        intra_cluster_dists = []
        for label in unique_labels:

            mask = tf.equal(y_true, label)  # Points belonging to the current cluster
            cluster_points = tf.boolean_mask(X, mask)  # Points in the current cluster
            pairwise_distances = tf.norm(
                tf.expand_dims(cluster_points, axis=1) - cluster_points, axis=-1
            )
            # Exclude self-distances (diagonal) when computing mean distance
            intra_dist = tf.reduce_mean(pairwise_distances - tf.linalg.diag(tf.linalg.diag_part(pairwise_distances)))
            intra_cluster_dists.append(intra_dist)
        a = tf.stack(intra_cluster_dists)

        # Compute inter-cluster distances (b[i] for each point)
        inter_cluster_dists = []
        for label in unique_labels:
            mask = tf.equal(y_true, label)
            current_cluster_points = tf.boolean_mask(X, mask)
            other_clusters_points = tf.boolean_mask(X, tf.logical_not(mask))
            if tf.shape(other_clusters_points)[0] > 0:
                dist = tf.reduce_mean(
                    tf.norm(
                        tf.expand_dims(current_cluster_points, axis=1) - other_clusters_points,
                        axis=-1
                    )
                )
            else:
                dist = 0.0  # No other clusters; handle edge case
            inter_cluster_dists.append(dist)
            fsddfs=0
        b = tf.stack(inter_cluster_dists)

        # Compute Silhouette score for each cluster
        a=tf.reduce_mean(a)
        b=tf.reduce_mean(b)
        silhouette_values = (b - a) / tf.maximum(a, b)
        silhouette_score = tf.reduce_mean(silhouette_values)

        # Loss is 1 - silhouette score
        return silhouette_score.numpy()


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
