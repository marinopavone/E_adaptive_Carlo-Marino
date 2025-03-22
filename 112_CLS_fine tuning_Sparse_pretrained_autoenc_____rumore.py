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
Nrm_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST_Nrm_model_(2000, 16800, 0.1).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})

x_train_enc = Nrm_autoencoder.bottleneck(x_train)
x_test_enc = Nrm_autoencoder.bottleneck(x_test)
#%%   tuning
# Define the hyperparameter grid
num_epochs_list = [2000]  # Different epoch values
batch_size_list = [ 512]
learning_rate_list = [ 0.0001, 0.001, ]
momentum = [ 0.0001, 0.001, 0.01, 0.1][::-1]
momentum = [ 0.05, 0.1][::-1]
# Lambda = [0.001, 0.01, 1, 5 ]
# beta_list=[0.005, 0.05, 0.3 ]
# rho_list=[0.005, 0.05, 0.3 ]
# beta_list=[0.0001, 0.001, 0.01, 0.1, 1]
# rho_list=[0.0001, 0.001, 0.01, 0.1, 1]

best_model = None
best_loss = float("inf")
best_params = None

model, loss = CLS_neural_network_training(x_train_enc, y_train, num_epochs=2000,
                                        batch_size=1000, learning_rate=0.001, momentum=0.9,
                                        num_classes=y_train.shape[1], hidden_units=20)
std_noise_test = [0, 0.0005, 0.002, 0.03, 0.1, 0.4, 0.8, 10]


print("Comparing data _____________________________")
for std in std_noise_test:
    print("Standard deviation is: _____________________ " + str(std))
    x_test_deg = add_gaussian_noise(x_test, mean=0, std=std)
    x_test_enc = Nrm_autoencoder.bottleneck(x_test_deg)

    a = test_CLS_neural_network(model, x_test_enc, y_test)
