import itertools

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from chemical_brother.deep_clustering import DeepClustering
import tensorflow as tf
from sklearn.metrics import silhouette_samples, silhouette_score

from chemical_brother.old_stuf_di_carlo.loss import SilhouetteLoss


y_true = tf.constant([0, 0, 1, 1, 1, 1], dtype=tf.int32)

# y_pred contains predicted embeddings for each data point
X = tf.constant([
    [4.0, 4.0],  # Cluster 1
    [4.1, 4.2],  # Cluster 0
    [8.0, 8.0],  # Cluster 1
    [8.1, 8.2],  # Cluster 1
    [4.0, 4.0],  # Cluster 1
    [4.1, 4.2],  # Cluster 2
], dtype=tf.float32)

loss_fn = SilhouetteLoss()
loss_value = loss_fn(y_true, X)

y_true=y_true.numpy()
X=X.numpy()


# sil_loss = silhouette_samples(X, y_true, metric='euclidean')
sil_loss = silhouette_score(X, y_true, metric='euclidean')


# Print the result
print(f"Silhouette Loss Value: {loss_value.numpy()}")
print(f"Silhouette Loss Value scikit-learn: {sil_loss}")