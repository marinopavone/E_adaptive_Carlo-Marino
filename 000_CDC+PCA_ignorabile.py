import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.saving import custom_object_scope
from chemical_brother.Autoencoders_CLASS_4_Nrm_CTR_Spa import *
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
# x_anomaly       = loaded_variables['x_anomaly']
# y_anomaly       = loaded_variables['y_anomaly']

#%%   extracting PCA transformation
from sklearn.decomposition import KernelPCA, PCA
# Set number of principal components
n_components = 6

# Apply Kernel PCA with RBF kernel
# pca = KernelPCA(n_components=n_components, kernel='linear')  # gamma controls kernel width
pca = PCA(n_components=n_components)  # gamma controls kernel width

# Fit and transform the training data
x_train = pca.fit_transform(x_train)
x_test  = pca.transform(x_test)
# x_anomaly = pca.transform(x_anomaly)

#%%   extracting PCA transformation

num_epochs = 250
batch_size = 12288
learning_rate=0.3
momentum = 0.07
model_info = ("ep_" + str(num_epochs) + "_bs_" + str(batch_size) +
              "_lr_" + str(learning_rate) +"_mo_" + str(momentum) + "")

retrain =True #False #
# train the model
if retrain:
    Nrm_model,Nrm_loss = autoencoder_training(x_train, num_epochs ,batch_size,learning_rate,momentum)
    CTR_model,CTR_loss = CTR_autoencoder_training(x_train, num_epochs ,batch_size,learning_rate,momentum,Lambda=1)
    Nrm_model.save("Tensorflow_models/CDC_autoencoders/Nrm_model" + str(model_info) + ".keras")
    CTR_model.save("Tensorflow_models/CDC_autoencoders/CTR_model" + str(model_info) + ".keras")
else:
    from keras.models import load_model

    # Load models with custom object registration
    Nrm_model = load_model("Tensorflow_models/CDC_autoencoders/Nrm_model" + str(model_info) + ".keras",
                           custom_objects={'AutoEncoder': AutoEncoder})

    CTR_model = load_model("Tensorflow_models/CDC_autoencoders/CTR_model" + str(model_info) + ".keras",
                           custom_objects={'AutoEncoder': AutoEncoder})


std_noise_test = [0, 0.0005, 0.002, 0.03, 0.1, 0.4, 0.8, 10]
for std in std_noise_test:
    print("Standard deviation is: _____________________ " + str(std))
    x_test_deg = add_gaussian_noise(x_test, mean=0, std=std)
    compare_autoencoders(Nrm_model, CTR_model, x_test_deg, y_test)


nrm_reconstructions = Nrm_model.predict(x_test[1:3])[0]
ctr_reconstructions = CTR_model.predict(x_test[1:3])[0]

