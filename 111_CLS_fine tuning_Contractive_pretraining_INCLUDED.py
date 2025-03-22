import numpy as np
import pickle
from chemical_brother.CLS_Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from chemical_brother.Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from chemical_brother.CLS_e_basta_CLASS_4_Nrm_CTR_Spa import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#%%    load data
pickle_filename = 'X&Y_train_X&Y_test_X&Y_anomaly.pkl'
pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-20 09:19:23.pkl'
pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-20 09:21:21.pkl'
pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-22 11:40:22.pkl'
# pickle_filename = 'Datasets Pikle/X&Y_ds_created_2025-03-20 09:20:57.pkl'# dill.load_module(python_session)
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

#%%   tuning
# Define the hyperparameter grid
num_epochs_list = [4000]  # Different epoch values
batch_size_list = [16800]
learning_rate_list = [  0.1]
momentum = [ 0.0001, 0.001, 0.01, 0.1][::-1]
momentum = [ 0.05][::-1]
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

    os.system('say "autoencoder"')
    autoencoder, rec_loss = autoencoder_training(x_train, num_epochs, batch_size, learning_rate, momentum)

    x_train_enc = autoencoder.bottleneck(x_train)
    x_test_enc = autoencoder.bottleneck(x_test)
    os.system('say "classificazione"')
    model, _ = CLS_neural_network_training(x_train_enc, y_train, num_epochs=num_epochs,
                                            batch_size=batch_size, learning_rate=learning_rate,momentum=momentum,
                                            num_classes=y_train.shape[1])
    global_classification_accuracy = test_CLS_neural_network(model, x_test_enc, y_test)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, momentum={momentum}, loss={loss}")
    os.system('say "iterazione completata"')

    if global_classification_accuracy > best_accuracy:
        best_accuracy = global_classification_accuracy
        best_model = model
        best_autoencoder = autoencoder
        best_params = (num_epochs, batch_size, learning_rate, momentum)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best ACCURACY:", best_accuracy)
best_model.save("Tensorflow_models/CDC_AUTandCLS_tuning_INCLUDED/BEST_Nrm_CLS_only" + str(best_params) + ".keras")
best_autoencoder.save("Tensorflow_models/CDC_AUTandCLS_tuning_INCLUDED/BEST_Nrm_CLS_only" + str(best_params) + ".keras")

# test_CLS_neural_network(best_model,x_test_enc,y_test)

import os
os.system('say "fine,  mannaggia a  dio"')
