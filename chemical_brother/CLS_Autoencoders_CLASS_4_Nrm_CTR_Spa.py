import pandas as pd
import tensorflow as tf
import numpy as np

def KL_divergence(rho, rho_hat):
    """ Compute KL divergence loss to enforce sparsity """
    rho_hat = tf.clip_by_value(rho_hat, 1e-10, 1.0)  # Avoid log(0)
    rho = tf.constant(rho, dtype=tf.float32)
    kl = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
    return tf.reduce_sum(kl)


class CLS_AutoEncoder(tf.keras.Model):
    def __init__(self, num_classes, hidden_units=10, bottleneck_units=3):
        super(CLS_AutoEncoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_dec = tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu)

        # Bottleneck layer (latent space)
        self.bottleneck = tf.keras.layers.Dense(bottleneck_units, activation=tf.nn.relu)

        # Decoder layers
        self.dense_enc = tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu)
        self.dense_final = tf.keras.layers.Dense(hidden_units)

        # Classification head
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inp, training=False):
        x_reshaped = self.flatten_layer(inp)
        x = self.dense_dec(x_reshaped)
        x = self.bottleneck(x)
        x_hid = x  # Bottleneck output

        # Reconstruction path
        x_rec = self.dense_enc(x)
        x_rec = self.dense_final(x_rec)

        # Classification path
        x_class = self.classifier(x_hid)

        return x_rec, x_reshaped, x_hid, x_class

    def get_config(self):
        return {"num_classes": self.classifier.units, "hidden_units": self.dense_dec.units,
                "bottleneck_units": self.bottleneck.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Loss function combining reconstruction and classification loss
def loss(x, x_bar, y, x_class, model, lambda_rec=1.0, lambda_class=1.0):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
    classification_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, x_class))
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
def CLS_autoencoder_training(x_train, y_train, num_classes, num_epochs=200, batch_size=128, learning_rate=0.001,
                         momentum=0.9):
    model = CLS_AutoEncoder(num_classes=num_classes)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
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