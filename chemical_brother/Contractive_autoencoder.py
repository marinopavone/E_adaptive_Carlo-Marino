from __future__ import annotations
import tensorflow as tf
import os
from enum import Enum
from random import randint

import pandas as pd
import numpy as np
from tensorflow import keras

def generate_contractive_autoenc(x_train,x_test,encoding_dim,input_shape) -> keras.Model:
    # Define the input layer
    input_img = keras.Input(shape=(input_shape,))

    # Define the encoded layer
    encoded = keras.layers.Dense(encoding_dim, activation='relu', name="encoder")(input_img)

    # Define the decoded layer
    decoded = keras.layers.Dense(input_shape, activation='sigmoid', name="decoder")(encoded)

    # Define the contractive autoencoder model
    autoencoder = keras.Model(input_img, decoded)

    # Define the regularization coefficient
    lam = 1e-4

    # Define the loss function with the regularization term
    def contractive_loss(autoencoder, lam):
        mse = keras.losses.MeanSquaredError()  # Creare l'istanza correttamente

        def loss(y_true, y_pred):
            mse_loss = mse(y_true, y_pred)  # Calcolare il MSE

            # Ottenere i pesi del layer 'dense'
            w = autoencoder.get_layer('dense').kernel  # W (matrice dei pesi)
            w_t = tf.transpose(w)  # Trasposta di W

            # Output del layer 'dense'
            h = autoencoder.get_layer('dense').output  # Attenzione: output del grafo

            # Derivata dell'attivazione (sigmoide derivata: h * (1 - h))
            dh = h * (1 - h)

            # Calcolo della perdita contrattiva
            contractive = lam * tf.reduce_sum(dh ** 2 * tf.reduce_sum(w_t ** 2, axis=1), axis=1)

            return mse_loss + tf.reduce_mean(contractive)  # Restituisce il valore corretto

        return loss

    # Compile the autoencoder model with the custom loss function
    autoencoder.compile(optimizer='adam', loss="mse")

    # Train the autoencoder model
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    autoencoder.save("autoencoder_model.keras")  # Saves in Keras format

    # Encode and decode some digits from the test set
    return autoencoder