import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.saving import custom_object_scope
from chemical_brother.Contractive_autoencoderTF import *
from keras.utils import custom_object_scope
#%%    load data
pickle_filename = 'X&Y_train_X&Y_test_X&Y_anomaly.pkl'
# dill.load_module(python_session)
with open(pickle_filename, 'rb') as file:
    loaded_variables = pickle.load(file)
x_train         = loaded_variables['x_train'       ]
y_train         = loaded_variables['y_train'       ]
x_test          = loaded_variables['x_test'        ]
y_test          = loaded_variables['y_test'        ]
x_anomaly       = loaded_variables['x_anomaly']
y_anomaly       = loaded_variables['y_anomaly']

num_epochs = 1000
batch_size = 1600
learning_rate=0.0001
model_info = "ep_" + str(num_epochs) + "_bs_" + str(batch_size) + "_lr_" + str(learning_rate) + ""


# Define the hyperparameter grid
num_epochs_list = [1024*8]  # Different epoch values
batch_size_list = [4096*3] #[512,1024,2048,4096]
learning_rate_list = [ 0.05, 0.3]#[0.001, 0.01, 0.1]

best_model = None
best_loss = float("inf")
best_params = None

from itertools import product
# _______________________________________________________ Normal
for num_epochs, batch_size, learning_rate in product(num_epochs_list, batch_size_list, learning_rate_list):
    model, loss = autoencoder_training(x_train, num_epochs, batch_size, learning_rate)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, loss={loss}")

if loss < best_loss:
    best_loss = loss
    best_model = model
    best_params = (num_epochs, batch_size, learning_rate)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best Loss:", best_loss)
best_model.save("Tensorflow_models/CDC_autoencoders/Nrm_model_" + str(best_params) + ".keras")
# _______________________________________________________ Contractive
# Define the hyperparameter grid

Lambda = [0.001, 0.005, 0.01]

best_model = None
best_loss = float("inf")
best_params = None
for num_epochs, batch_size, learning_rate, Lambda in product(num_epochs_list, batch_size_list, learning_rate_list, Lambda):
    model, loss = CTR_autoencoder_training(x_train, num_epochs, batch_size, learning_rate, Lambda)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, loss={loss}")

if loss < best_loss:
    best_loss = loss
    best_model = model
    best_params = (num_epochs, batch_size, learning_rate)
print("# _______________________________________________________ Contractive autoencoder")

print("Best Parameters:", best_params)
print("Best Loss:", best_loss)
best_model.save("Tensorflow_models/CDC_autoencoders/CTR_model_" + str(best_params) + ".keras")

#
# Nrm_model = autoencoder_training(x_train, num_epochs ,batch_size,learning_rate)
# CTR_model = CTR_autoencoder_training(x_train, num_epochs ,batch_size,learning_rate,Lambda=1)
#
#
#
#
#
# Nrm_model.save("Tensorflow_models/CDC_autoencoders/Nrm_model" + str(model_info) + ".keras")
# CTR_model.save("Tensorflow_models/CDC_autoencoders/CTR_model" + str(model_info) + ".keras")
#
#
