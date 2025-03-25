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
num_epochs_list = [2000]  # Different epoch values
batch_size_list = [ 500]
learning_rate_list = [0.01, 0.002]
# momentum = [ 0.0001, 0.01, 0.1][::-1]
momentum = [  0 ][::-1]
Lambda = [0.001, 0.01]
# Lambda = [0.01]
# beta_list=[0.005, 0.05, 0.3 ]
# rho_list=[0.005, 0.05, 0.3 ]
# beta_list=[0.0001, 0.001, 0.01, 0.1, 1]
# rho_list=[0.0001, 0.001, 0.01, 0.1, 1]
N_iter = np.sum([1 for num_epochs, batch_size, learning_rate, momentum in product(num_epochs_list, batch_size_list, learning_rate_list, momentum)])
print(N_iter, "iterazioni da fare")
best_model = None
best_accuracy = 0
best_params = None

import os
message= 'say "start "'
os.system(message)
i=0
# _______________________________________________________ Normal
for num_epochs, batch_size, learning_rate, momentum, lambda_CTR in product(num_epochs_list, batch_size_list, learning_rate_list, momentum, Lambda):
    model, _ = CLS_CTR_autoencoder_training(x_train, y_train, num_epochs=num_epochs,
                                            batch_size=batch_size, learning_rate=learning_rate,momentum=momentum,
                                            num_classes=y_train.shape[1], hidden_units=10, bottleneck_units=5,
                                            lambda_rec = 0.1, lambda_cla = 0.3, lambda_CTR = lambda_CTR)
    global_classification_accuracy = test_CLS_Autoencoder(model, x_test, y_test)
    print(f"Training with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, momentum={momentum}, loss={loss}, lambda_CTR={lambda_CTR},")
    test_CLS_Autoencoder(model, x_test, y_test)
    if global_classification_accuracy > best_accuracy:
        best_accuracy = global_classification_accuracy
        best_model = model
        best_params = (num_epochs, batch_size, learning_rate, lambda_CTR)
    i+=1
    message = 'say "completate'+str(i)+  ' itarazioni su'+str(N_iter)+  ' "'
    os.system(message)
print("# _______________________________________________________ Normal autoencoder")
print("Best Parameters:", best_params)
print("Best accuracy:", best_accuracy)
best_model.save("Tensorflow_models/CDC_CLS_Autoencoders/BEST_CTR_model_" + str(best_params) + ".keras")

test_CLS_Autoencoder(best_model,x_test,y_test)
message = 'say "fine , dio scemo "'
os.system(message)






def CTR_loss(x, x_bar, y, x_class, model, h,lambda_rec=1.0, lambda_class=0.3 , lambda_CTR=0.01):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
    W = tf.Variable(model.bottleneck.weights[0])
    dh = h * (1 - h)  # N_batch x N_hidden
    W = tf.transpose(W)
    # contractive = Lambda * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
    batch_size = tf.cast(tf.shape(x)[0], tf.float32)
    classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, x_class))
    contractive = tf.reduce_mean(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1) / batch_size
    total_loss = lambda_rec * reconstruction_loss + lambda_class * classification_loss + lambda_CTR * contractive
    return total_loss

def CTR_grads(model, inputs, lables, lambda_rec=0.01, lambda_cla=0.03, lambda_CTR=0.01):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped, hidden, classification  = model(inputs)
        loss_value = CTR_loss(inputs_reshaped, reconstruction,lables, classification, model,hidden, lambda_rec, lambda_cla, lambda_CTR)
    return loss_value, tape.gradient(loss_value,
                                     model.trainable_variables), inputs_reshaped, reconstruction, classification
#