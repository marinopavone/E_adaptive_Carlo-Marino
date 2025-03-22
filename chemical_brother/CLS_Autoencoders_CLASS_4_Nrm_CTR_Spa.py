import pandas as pd
import tensorflow as tf
import numpy as np

def KL_divergence(rho, rho_hat):
    """ Compute KL divergence loss to enforce sparsity """
    rho_hat = tf.clip_by_value(rho_hat, 1e-10, 1.0)  # Avoid log(0)
    rho = tf.constant(rho, dtype=tf.float32)
    kl = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
    return tf.reduce_sum(kl)


import tensorflow as tf


class CLS_AutoEncoder(tf.keras.Model):
    def __init__(self, num_classes, hidden_units=10, bottleneck_units=3):
        super(CLS_AutoEncoder, self).__init__()
        tf.random.set_seed(42)

        self.flatten_layer = tf.keras.layers.Flatten()

        # Encoder
        self.dense_dec = tf.keras.layers.Dense(hidden_units, activation=None)  # No activation before BN
        self.bn_dec = tf.keras.layers.BatchNormalization()
        self.relu_dec = tf.keras.layers.ReLU()

        # Bottleneck layer (latent space)
        self.bottleneck = tf.keras.layers.Dense(bottleneck_units, activation=None)
        self.bn_bottleneck = tf.keras.layers.BatchNormalization()
        self.relu_bottleneck = tf.keras.layers.ReLU()

        # Decoder layers
        self.dense_enc = tf.keras.layers.Dense(hidden_units, activation=None)
        self.bn_enc = tf.keras.layers.BatchNormalization()
        self.relu_enc = tf.keras.layers.ReLU()

        self.dense_final = tf.keras.layers.Dense(hidden_units, activation=None)
        self.bn_final = tf.keras.layers.BatchNormalization()

        # Classification head
        self.classifier = tf.keras.layers.Dense(hidden_units, activation=None)
        self.bn_classifier = tf.keras.layers.BatchNormalization()
        self.relu_classifier = tf.keras.layers.ReLU()

        self.classifier_out = tf.keras.layers.Dense(num_classes, activation="softmax")  # Softmax for multi-class

    def call(self, inp, training=False):
        x_reshaped = self.flatten_layer(inp)

        # Encoder Path
        x = self.dense_dec(x_reshaped)
        x = self.bn_dec(x, training=training)
        x = self.relu_dec(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bn_bottleneck(x, training=training)
        x_hid = self.relu_bottleneck(x)  # Bottleneck output

        # Reconstruction Path
        x_rec = self.dense_enc(x_hid)
        x_rec = self.bn_enc(x_rec, training=training)
        x_rec = self.relu_enc(x_rec)

        x_rec = self.dense_final(x_rec)
        x_rec = self.bn_final(x_rec, training=training)  # Keep activation linear for reconstruction

        # Classification Path
        x_class = self.classifier(x_hid)
        x_class = self.bn_classifier(x_class, training=training)
        x_class = self.relu_classifier(x_class)

        x_class = self.classifier_out(x_class)  # Softmax activation for classification

        return x_rec, x_reshaped, x_hid, x_class

    def get_config(self):
        return {
            "num_classes": self.classifier_out.units,
            "hidden_units": self.dense_dec.units,
            "bottleneck_units": self.bottleneck.units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# Loss function combining reconstruction and classification loss
def loss(x, x_bar, y, x_class, model, lambda_rec=1.0, lambda_class=0.3):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
    classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, x_class))
    total_loss = lambda_rec * reconstruction_loss + lambda_class * classification_loss
    return total_loss


# Gradient computation
def grads(model, inputs, labels):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped, hidden, classification = model(inputs)
        loss_value = loss(inputs_reshaped, reconstruction, labels, classification, model)
    return loss_value, tape.gradient(loss_value,
                                     model.trainable_variables), inputs_reshaped, reconstruction, classification


# Training function
def CLS_autoencoder_training(x_train, y_train, num_epochs=200,
                             batch_size=128, learning_rate=0.001, momentum=0.9,
                             num_classes=5, hidden_units=10, bottleneck_units=3):
    model = CLS_AutoEncoder(num_classes, hidden_units, bottleneck_units)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=num_epochs,
    #                                                              decay_rate=0.8)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    global_step = tf.Variable(0)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for i in range(0, len(x_train), batch_size):
            x_inp = x_train[i: i + batch_size]
            y_inp = y_train[i: i + batch_size]
            loss_value, grad, inputs_reshaped, reconstruction, classification = grads(model, x_inp, y_inp)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        print("Step: {}, Loss: {}".format(global_step.numpy(), tf.reduce_sum(loss_value)))
        loss_value_final = tf.reduce_sum(loss_value)
    return model, loss_value_final


def test_CLS_Autoencoder(model, x_test, y_test):
    # Perform model predictions
    reconstructions, _, _, classifications = model.predict(x_test)

    # Compute Mean Squared Error (MSE) loss for reconstruction
    loss = np.mean(np.square(x_test - reconstructions), axis=1)  # Per sample loss
    predicted_classes = np.argmax(classifications, axis=1)  # Convert softmax scores to class indices
    labels = np.argmax(y_test, axis=1)  # Convert softmax scores to class indices

    # Unique classes
    classes = np.unique(labels)

    loss_dict = {}
    global_reconstruction_loss = []
    global_accuracy = []

    print("\nReconstruction & Classification Loss Comparison per Class:\n")
    for cls in classes:
        # Get indices for current class
        cls_indices = np.where(labels == cls)[0]

        # Compute average reconstruction loss for this class
        avg_loss = np.mean(loss[cls_indices])
        global_reconstruction_loss.append(avg_loss)

        # Compute classification accuracy for this class
        cls_accuracy = np.mean(predicted_classes[cls_indices] == labels[cls_indices])
        global_accuracy.append(cls_accuracy)

        loss_dict[cls] = {
            "reconstruction_loss": avg_loss,
            "classification_accuracy": global_accuracy
        }

        print(f"Class {cls}: Reconstruction Loss = {avg_loss:.6f}, Classification Accuracy = {cls_accuracy:.2%}")

    # Compute global reconstruction loss & classification accuracy
    global_reconstruction_loss = np.mean(global_reconstruction_loss)
    global_classification_accuracy = np.mean(global_accuracy)

    print(f"\nGlobal: Reconstruction Loss = {global_reconstruction_loss:.6f}, Classification Accuracy = {global_classification_accuracy:.2%}")

    return global_classification_accuracy



