import numpy as np
import pickle
from chemical_brother.CLS_Autoencoders_CLASS_4_Nrm_CTR_Spa import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from itertools import product

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
#%%   tuning
# Define the hyperparameter grid
num_epochs_list = [3000]  # Different epoch values
batch_size_list = [ 8400]
learning_rate_list = [   0.2,0.3]
momentum = [ 0.0001, 0.001, 0.01, 0.1][::-1]
momentum = [  0.0001, 0.005][::-1]
# Lambda = [0.001, 0.01, 1, 5 ]
# beta_list=[0.005, 0.05, 0.3 ]
# rho_list=[0.005, 0.05, 0.3 ]
# beta_list=[0.0001, 0.001, 0.01, 0.1, 1]
# rho_list=[0.0001, 0.001, 0.01, 0.1, 1]
N_iter = np.sum([1 for num_epochs, batch_size, learning_rate, momentum in product(num_epochs_list, batch_size_list, learning_rate_list, momentum)])
print(N_iter, "iterazioni da fare")
best_model = None
best_accuracy = 0
best_params = None

# model, loss = CLS_autoencoder_training(x_train, y_train, num_epochs=2000,
#                                         batch_size=1000, learning_rate=0.001, momentum=0.9,
#                                         num_classes=y_train.shape[1], hidden_units=10, bottleneck_units=5)
# test_CLS_Autoencoder(model, x_test, y_test)
# model.save("Tensorflow_models/CDC_autoencoders/first_CLS.keras")

# from keras.models import load_model
# model = load_model("Tensorflow_models/CDC_CLS_Autoencoders/first_CLS.keras",
#                        custom_objects={'AutoEncoder': CLS_AutoEncoder})
#
# test_CLS_Autoencoder(model,x_test,y_test)
import os
message= 'say "start "'
os.system(message)
i=0
# _______________________________________________________ Normal
for num_epochs, batch_size, learning_rate, momentum in product(num_epochs_list, batch_size_list, learning_rate_list, momentum):
    model, _ = CLS_autoencoder_training(x_train, y_train, num_epochs=num_epochs,
                                            batch_size=batch_size, learning_rate=learning_rate,momentum=momentum,
                                            num_classes=y_train.shape[1], hidden_units=10, bottleneck_units=5)
    global_classification_accuracy = test_CLS_Autoencoder(model, x_test, y_test)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, momentum={momentum}, loss={loss}")
    test_CLS_Autoencoder(model, x_test, y_test)
    if global_classification_accuracy > best_accuracy:
        best_accuracy = global_classification_accuracy
        best_model = model
        best_params = (num_epochs, batch_size, learning_rate, momentum)
    i+=1
    message = 'say "completate'+str(i)+  ' itarazioni su'+str(N_iter)+  ' "'
    os.system(message)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best accuracy:", best_accuracy)
best_model.save("Tensorflow_models/CDC_CLS_Autoencoders/BEST_Nrm_model_" + str(best_params) + ".keras")

test_CLS_Autoencoder(best_model,x_test,y_test)
message = 'say "fine , dio scemo "'
os.system(message)