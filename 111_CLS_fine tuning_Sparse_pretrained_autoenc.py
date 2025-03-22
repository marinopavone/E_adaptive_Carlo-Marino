import numpy as np
import pickle
from chemical_brother.CLS_Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from chemical_brother.Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from chemical_brother.CLS_e_basta_CLASS_4_Nrm_CTR_Spa import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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

#%%   label encoding
y_train = tf.keras.utils.to_categorical(y_train)  # Adjust indexing
y_test = tf.keras.utils.to_categorical(y_test)  # Adjust indexing

from keras.models import load_model
# Load models with custom object registration

# Nrm_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Nrm_model_(2000, 16800, 0.1).keras",
#                        custom_objects={'AutoEncoder': AutoEncoder})
# CTR_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_CTR_model_(2000, 16800, 0.01, 0.01).keras",
#                        custom_objects={'AutoEncoder': AutoEncoder})
Spa_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Spa_model_(2000, 16800, 0.01, 0.3, 0.3).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})


x_train_enc = Spa_autoencoder.bottleneck(x_train)
x_test_enc = Spa_autoencoder.bottleneck(x_test)
#%%   tuning
# Define the hyperparameter grid
num_epochs_list = [2000]  # Different epoch values
batch_size_list = [ 1000]
learning_rate_list = [ 0.0001, 0.001, ]
# momentum = [ 0.0001, 0.001, 0.01, 0.1][::-1]
momentum = [ 0.05, 0.1][::-1]
# Lambda = [0.001, 0.01, 1, 5 ]
# beta_list=[0.005, 0.05, 0.3 ]
# rho_list=[0.005, 0.05, 0.3 ]
# beta_list=[0.0001, 0.001, 0.01, 0.1, 1]
# rho_list=[0.0001, 0.001, 0.01, 0.1, 1]

best_model = None
best_accuracy = 0
best_params = None
i=0
import os
os.system('say "start"')
from itertools import product
# _______________________________________________________ Normal
for num_epochs, batch_size, learning_rate, momentum in product(num_epochs_list, batch_size_list, learning_rate_list, momentum):
    model, _ = CLS_neural_network_training(x_train_enc, y_train, num_epochs=num_epochs,
                                            batch_size=batch_size, learning_rate=learning_rate,momentum=momentum,
                                            num_classes=y_train.shape[1])
    global_classification_accuracy = test_CLS_neural_network(model, x_test_enc, y_test)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, momentum={momentum}, loss={loss}")
    os.system('say "iterazione completata"')

    if global_classification_accuracy > best_accuracy:
        best_accuracy = global_classification_accuracy
        best_model = model
        best_params = (num_epochs, batch_size, learning_rate, momentum)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best ACCURACY:", best_accuracy)
best_model.save("Tensorflow_models/CDC_CLS_only/BEST_Spa_CLS_only" + str(best_params) + ".keras")

test_CLS_neural_network(best_model,x_test_enc,y_test)


os.system('say "fine dio marmellata"')
