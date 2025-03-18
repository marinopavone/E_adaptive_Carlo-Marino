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



# ______________________________________________ contractvive

# ______________________________________________ contractive
# ______________________________________________ normal

# ______________________________________________ normal
 # _____________________________________________ Sparse
epochs = 2000
batch_size = 16800
lr = 0.005
beta = 0.3
rho = 0.3
best_params = (epochs, batch_size, lr, beta, rho)
retrain = False #True #
if retrain:
    best_model, loss = Spa_autoencoder_training(x_train, epochs, batch_size, lr, beta, rho)
    # best_model.save("Tensorflow_models/CDC_autoencoders/BEST_Spa_model_" + str(best_params) + ".keras")
else:
    from keras.models import load_model
    Spa_model = load_model("Tensorflow_models/CDC_autoencoders/BEST_PCA_Spa_model_None.keras",
                           custom_objects={'AutoEncoder': AutoEncoder})
    # Nrm_model = load_model("Tensorflow_models/CDC_autoencoders/BEST_Spa_model_" + str(best_params) + ".keras",
    #                        custom_objects={'AutoEncoder': AutoEncoder})
# ______________________________________________ Sparse

