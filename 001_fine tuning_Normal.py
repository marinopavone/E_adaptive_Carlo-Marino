import numpy as np
import pickle
from chemical_brother.Autoencoders_CLASS_4_Nrm_CTR_Spa import *

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


#%%   tuning
# Define the hyperparameter grid
num_epochs_list = [2000]  # Different epoch values
batch_size_list = [ 16800] #[512,1024,2048,4096]
learning_rate_list = [ 0.0001, 0.001, 0.01, 0.1]
learning_rate_list = [  0.1]
momentum = [ 0.0001, 0.001, 0.01, 0.1][::-1]
momentum = [0.001]
# Lambda = [0.001, 0.01, 1, 5 ]
# beta_list=[0.005, 0.05, 0.3 ]
# rho_list=[0.005, 0.05, 0.3 ]
# beta_list=[0.0001, 0.001, 0.01, 0.1, 1]
# rho_list=[0.0001, 0.001, 0.01, 0.1, 1]

best_model = None
best_loss = float("inf")
best_params = None

from itertools import product
# _______________________________________________________ Normal
for num_epochs, batch_size, learning_rate, momentum in product(num_epochs_list, batch_size_list, learning_rate_list, momentum):
    model, loss = autoencoder_training(x_train, num_epochs, batch_size, learning_rate, momentum)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, momentum={momentum}, loss={loss}")

    if loss < best_loss:
        best_loss = loss
        best_model = model
        best_params = (num_epochs, batch_size, learning_rate)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best Loss:", best_loss)
best_model.save("Tensorflow_models/CDC_autoencoders/Nrm_model_" + str(best_params) + ".keras")
