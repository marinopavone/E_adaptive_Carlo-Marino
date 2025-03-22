import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.saving import custom_object_scope
from chemical_brother.Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from keras.utils import custom_object_scope
#%%    load data
pickle_filename = 'X&Y_train_X&Y_test_X&Y_anomaly.pkl'
# pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-19 19:56:35.pkl'
# pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-19 19:58:16.pkl'
# pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-19 19:59:15.pkl'
# dill.load_module(python_session)
with open(pickle_filename, 'rb') as file:
    loaded_variables = pickle.load(file)
x_train         = loaded_variables['x_train'       ]
y_train         = loaded_variables['y_train'       ]
x_test          = loaded_variables['x_test'        ]
y_test          = loaded_variables['y_test'        ]
# x_anomaly       = loaded_variables['x_anomaly']
# y_anomaly       = loaded_variables['y_anomaly']

#%%   se voglio fa il training mi setto gli iperparametri

from keras.models import load_model


# Load models with custom object registration
Nrm_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Nrm_model_(2000, 16800, 0.1).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})
CTR_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_CTR_model_(2000, 16800, 0.01, 0.01).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})
Spa_autoencoder = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Spa_model_(2000, 16800, 0.01, 0.3, 0.3).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})


# Load models with custom object registration
Nrm_classifier = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Nrm_model_(2000, 16800, 0.1).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})
CTR_classifier = load_model("Tensorflow_models/CDC_autoencoders/BEST5_CTR_model_(2000, 16800, 0.01, 0.01).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})
Spa_classifier = load_model("Tensorflow_models/CDC_autoencoders/BEST5_Spa_model_(2000, 16800, 0.01, 0.3, 0.3).keras",
                       custom_objects={'AutoEncoder': AutoEncoder})

std_noise_test = [0, 0.0005, 0.002, 0.03, 0.1, 0.4, 0.8, 10]


print("Comparing PCA data _____________________________")
for std in std_noise_test:
    print("Standard deviation is: _____________________ " + str(std))
    x_test_deg_pca = add_gaussian_noise(x_test, mean=0, std=std)
    compare_3_cla(Nrm_model_pca, CTR_model_pca, Spa_model_pca, x_test_deg_pca, y_test)


