import pandas as pd
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)


def KL_divergence(rho, rho_hat):
    """ Compute KL divergence loss to enforce sparsity """
    rho_hat = tf.clip_by_value(rho_hat, 1e-10, 1.0)  # Avoid log(0)
    rho = tf.constant(rho, dtype=tf.float32)
    kl = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
    return tf.reduce_sum(kl)


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        tf.random.set_seed(41)

        peppe=6
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_dec = tf.keras.layers.Dense(peppe, activation=tf.nn.relu)

        self.bottleneck = tf.keras.layers.Dense(3, activation=tf.nn.relu)

        self.dense_enc = tf.keras.layers.Dense(peppe, activation=tf.nn.relu)

        self.dense_final = tf.keras.layers.Dense(peppe)

    def call(self, inp):
        x_reshaped = self.flatten_layer(inp)
        # print(x_reshaped.shape)
        x = self.dense_dec(x_reshaped)
        x = self.bottleneck(x)
        x_hid = x
        x = self.dense_enc(x)
        x = self.dense_final(x)
        return x, x_reshaped, x_hid
    # ONLY FOR SAVINGGGGGGG  ✅ Make the model serializable scrivo anche get_config e from_config
    def get_config(self):
        return {}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#
# class AutoEncoder_PCA(tf.keras.Model):
#     def __init__(self):
#         super(AutoEncoder_PCA, self).__init__()
#
#         peppe=6
#         self.flatten_layer = tf.keras.layers.Flatten()
#         self.dense_dec = tf.keras.layers.Dense(peppe, activation=tf.nn.relu)
#
#         self.bottleneck = tf.keras.layers.Dense(3, activation=tf.nn.relu)
#
#         self.dense_enc = tf.keras.layers.Dense(peppe, activation=tf.nn.relu)
#
#         self.dense_final = tf.keras.layers.Dense(peppe)
#
#     def call(self, inp):
#         x_reshaped = self.flatten_layer(inp)
#         # print(x_reshaped.shape)
#         x = self.dense_dec(x_reshaped)
#         x = self.bottleneck(x)
#         x_hid = x
#         x = self.dense_enc(x)
#         x = self.dense_final(x)
#         return x, x_reshaped, x_hid
#     # ONLY FOR SAVINGGGGGGG  ✅ Make the model serializable scrivo anche get_config e from_config
#     def get_config(self):
#         return {}
#     @classmethod
#     def from_config(cls, config):
#         return cls()
def loss(x, x_bar, h, model, Lambda=100):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
    W = tf.Variable(model.bottleneck.weights[0])
    total_loss = reconstruction_loss
    return total_loss
def CTR_loss(x, x_bar, h, model, Lambda=100):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
    W = tf.Variable(model.bottleneck.weights[0])
    dh = h * (1 - h)  # N_batch x N_hidden
    W = tf.transpose(W)
    # contractive = Lambda * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
    batch_size = tf.cast(tf.shape(x)[0], tf.float32)
    contractive = Lambda * tf.reduce_mean(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1) / batch_size
    total_loss = reconstruction_loss + contractive
    return total_loss
def Spa_loss(x, x_bar, h, model, beta=0.1, rho=0.05):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))  # Reconstruction error
    rho_hat = tf.reduce_mean(h, axis=0)
    sparsity_loss = KL_divergence(rho, rho_hat)
    # Total loss
    total_loss = reconstruction_loss + beta * sparsity_loss
    return total_loss
def grads(model, inputs):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped, hidden = model(inputs)
        # reconstruction, inputs_reshaped, hidden = model.bottleneck(inputs)
        loss_value = loss(inputs_reshaped, reconstruction, hidden, model)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction

def CTR_grads(model, inputs, Lambda=100):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped, hidden = model(inputs)
        loss_value = CTR_loss(inputs_reshaped, reconstruction, hidden, model, Lambda)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction

def Spa_grads(model, inputs, beta=0.1, rho=0.05):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped, hidden = model(inputs)
        loss_value = Spa_loss(inputs_reshaped, reconstruction, hidden, model, beta, rho)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction


def autoencoder_training(x_train, num_epochs = 200, batch_size = 128,
                         learning_rate=0.001, momentum=0.9):
    model = AutoEncoder()
    # optimizer = tf.optimizers.Adam(learning_rate=learning_rate)#, momentum=momentum)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    global_step = tf.Variable(0)
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        for x in range(0, len(x_train), batch_size):
            x_inp = x_train[x: x + batch_size]
            loss_value, grad, inputs_reshaped, reconstruction = grads(model, x_inp)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        print("Step: {}, Loss: {}".format(global_step.numpy(), tf.reduce_sum(loss_value)))
        loss_value_final = tf.reduce_sum(loss_value)
    return model, loss_value_final

def CTR_autoencoder_training(x_train, num_epochs = 200, batch_size = 128,
                             learning_rate=0.001, momentum=0.9, Lambda=0.1):
    model = AutoEncoder()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate,weight_decay=0.01)#, momentum=momentum)
    global_step = tf.Variable(0)
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        for x in range(0, len(x_train), batch_size):
            x_inp = x_train[x: x + batch_size]
            loss_value, grad, inputs_reshaped, reconstruction = CTR_grads(model, x_inp,Lambda)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            global_step.assign_add(1)

        print("Step: {}, Loss: {}".format(global_step.numpy(), tf.reduce_sum(loss_value)))
        loss_value_final = tf.reduce_sum(loss_value)
    return model, loss_value_final

def Spa_autoencoder_training(x_train, num_epochs=200, batch_size=128,
                                learning_rate=0.001, beta=0.1, rho=0.05):
    model = AutoEncoder()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, weight_decay=0.01)
    global_step = tf.Variable(0)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for x in range(0, len(x_train), batch_size):
            x_inp = x_train[x: x + batch_size]
            loss_value, grad, inputs_reshaped, reconstruction = Spa_grads(model, x_inp, beta, rho)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            global_step.assign_add(1)

        print("Step: {}, Loss: {}".format(global_step.numpy(), tf.reduce_sum(loss_value)))

        loss_value_final = tf.reduce_sum(loss_value)
    return model, loss_value_final


def compare_autoencoders(Nrm_model, CTR_model, x_test, y_test):

    # Compute reconstructions
    nrm_reconstructions = Nrm_model.predict(x_test)[0]
    ctr_reconstructions = CTR_model.predict(x_test)[0]

    # Compute Mean Squared Error (MSE) loss
    nrm_loss = np.square(x_test - nrm_reconstructions)  # Adjust axes if needed
    ctr_loss = np.square(x_test - ctr_reconstructions)

    # Unique classes
    classes = np.unique(y_test)

    loss_dict = {}

    nrm_global_loss = list()
    ctr_global_loss = list()

    print("\nReconstruction Loss Comparison per Class:\n")
    for cls in classes:
        # Get indices for current class
        cls_indices = np.where(y_test == cls)[0]

        # Compute average loss for this class
        nrm_avg_loss = np.mean(nrm_loss[cls_indices])
        ctr_avg_loss = np.mean(ctr_loss[cls_indices])
        nrm_global_loss.append(nrm_avg_loss)
        ctr_global_loss.append(ctr_avg_loss)

        loss_dict[cls] = {"Nrm_model_loss": nrm_avg_loss, "CTR_model_loss": ctr_avg_loss}

        print(f"Class {cls}: Nrm_model Loss = {nrm_avg_loss:.6f}, CTR_model Loss = {ctr_avg_loss:.6f}")
    print(f"Global : Nrm_model Loss = {np.mean(nrm_global_loss):.6f}, CTR_model Loss = {np.mean(ctr_global_loss):.6f}")

    return loss_dict

import numpy as np

def compare_3_autoencoders(Nrm_model, CTR_model, Spa_model, x_test, y_test):
    # Compute reconstructions
    nrm_reconstructions = Nrm_model.predict(x_test)[0]
    ctr_reconstructions = CTR_model.predict(x_test)[0]
    spa_reconstructions = Spa_model.predict(x_test)[0]

    # Compute Mean Squared Error (MSE) loss
    nrm_loss = np.square(x_test - nrm_reconstructions)
    ctr_loss = np.square(x_test - ctr_reconstructions)
    spa_loss = np.square(x_test - spa_reconstructions)

    # Unique classes
    classes = np.unique(y_test)

    loss_dict = {}
    nrm_global_loss = list()
    ctr_global_loss = list()
    spa_global_loss = list()

    print("\nReconstruction Loss Comparison per Class:\n")
    for cls in classes:
        # Get indices for current class
        cls_indices = np.where(y_test == cls)[0]

        # Compute average loss for this class
        nrm_avg_loss = np.mean(nrm_loss[cls_indices])
        ctr_avg_loss = np.mean(ctr_loss[cls_indices])
        spa_avg_loss = np.mean(spa_loss[cls_indices])

        nrm_global_loss.append(nrm_avg_loss)
        ctr_global_loss.append(ctr_avg_loss)
        spa_global_loss.append(spa_avg_loss)

        loss_dict[cls] = {
            "Nrm_model_loss": nrm_avg_loss,
            "CTR_model_loss": ctr_avg_loss,
            "Spa_model_loss": spa_avg_loss
        }
        print(f"Class {cls}: Nrm_model Loss = {nrm_avg_loss:.6f}, CTR_model Loss = {ctr_avg_loss:.6f}, Spa_model Loss = {spa_avg_loss:.6f}")
    print("")
    print(f"Global : Nrm_model Loss = {np.mean(nrm_global_loss):.6f}, CTR_model Loss = {np.mean(ctr_global_loss):.6f}, Spa_model Loss = {np.mean(spa_global_loss):.6f}")
    return loss_dict









def add_gaussian_noise(array, mean=0, std=1):
    noise = np.random.normal(loc=mean, scale=std, size=array.shape)
    return array + noise


def degradate(array, mean=0, std=1):
    noise = np.random.normal(loc=mean, scale=std, size=array.shape)
    return array + noise