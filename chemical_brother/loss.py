import tensorflow as tf


def silhouette_loss(y_true, y_pred):
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
                tf.norm(x - tf.gather(y_pred, tf.where(tf.equal(y_true, x))), axis=1)
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
    return -silhouette


def soft_silhouette_loss(y_true, y_pred, alpha=1.0):
    """
    Soft silhouette loss function.

    Args:
        y_pred: Predicted probabilities (soft assignments).
        alpha: Softness parameter (default=1.0).

    Returns:
        Soft silhouette loss value.
    """
    # Calculate the soft assignment of points to clusters
    soft_assignments = tf.nn.softmax(y_pred, axis=1)

    # Calculate the mean distance between points in the same cluster
    same_cluster_dist = tf.reduce_mean(
        tf.reduce_mean(
            tf.square(tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=2)),
            axis=[1, 2],
        )
        * tf.expand_dims(soft_assignments, axis=2),
        axis=1,
    )

    # Calculate the mean distance between points in different clusters
    diff_cluster_dist = tf.reduce_mean(
        tf.reduce_mean(
            tf.square(tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=2)),
            axis=[1, 2],
        )
        * (1 - tf.expand_dims(soft_assignments, axis=2)),
        axis=1,
    )

    # Calculate the soft silhouette score
    soft_silhouette = (same_cluster_dist - diff_cluster_dist) / tf.maximum(
        same_cluster_dist, diff_cluster_dist
    )
    soft_silhouette = tf.reduce_mean(soft_silhouette)

    # Apply the softness parameter
    soft_silhouette = tf.pow(soft_silhouette, alpha)

    # Return the negative soft silhouette score as the loss
    return -soft_silhouette
