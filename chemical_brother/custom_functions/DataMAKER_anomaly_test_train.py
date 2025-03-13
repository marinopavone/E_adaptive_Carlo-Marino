from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from chemical_brother.data_maker import DataMaker, ChemicalClass
from chemical_brother.Contractive_autoencoder import generate_contractive_autoenc
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from tensorflow import keras
import dill
import pickle
#%%   Dataset known substances

known_sub = DataMaker("/Users/marinopavone/PycharmProjects/E_adaptive_Carlo-Marino/dataset")
known_sub.set_contamination_classes(
    [
        ChemicalClass.SODIUM_HYDROXIDE,
        ChemicalClass.SODIUM_CHLORIDE,
        ChemicalClass.SODIUM_HYPOCHLORITE,
        ChemicalClass.POTASSIUM_NITRATE,
        ChemicalClass.CALCIUM_NITRATE,

    ]
)
#%%   Dataset anomaly substances

anomaly_sub = DataMaker("/Users/marinopavone/PycharmProjects/E_adaptive_Carlo-Marino/dataset")
anomaly_sub.set_contamination_classes(
    [
        ChemicalClass.POTABLE_WATER,
        ChemicalClass.NELSEN,
        ChemicalClass.PHOSPHORIC_ACID,
        ChemicalClass.FORMIC_ACID,
        ChemicalClass.HYDROCHLORIC_ACID,
        ChemicalClass.HYDROGEN_PEROXIDE,
        ChemicalClass.ACETONE,
        ChemicalClass.ACETIC_ACID,
        ChemicalClass.ETHANOL,
        ChemicalClass.AMMONIA,
    ]
)

test_experiment_selection=[1,3,5]
stady_state_start,stady_state_end = 500,800
train, test = known_sub.split_train_test_by_experiment(test_experiment_selection, stady_state_start,stady_state_end)
x_train = train.drop(columns=["CLASS"]).to_numpy()
y_train = train[["CLASS"]].to_numpy()
x_test = test.drop(columns=["CLASS"]).to_numpy()
y_test = test[["CLASS"]].to_numpy()

anomaly_data = anomaly_sub.make_full_dataset()
anomaly_labels = anomaly_data[["CLASS"]].to_numpy()
anomaly_data = anomaly_data.drop(columns=["CLASS"]).to_numpy()

x_anomaly = anomaly_data
y_anomaly = anomaly_labels

#%%   encoding and normalization

scaler = MinMaxScaler()
scaler.fit(x_train)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
feature_encoder = LabelEncoder()
feature_encoder.fit(train.columns.to_numpy())

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder_anomaly = LabelEncoder()
label_encoder_anomaly.fit(y_anomaly)

x_anomaly = scaler.transform(x_test)
y_anomaly = label_encoder_anomaly.transform(anomaly_labels)

#%%   save TRAIN TEST SPLIT DB
print("saving TRAIN TEST ")
pickle_filename = 'X&Y_train_X&Y_test_X&Y_anomaly.pkl'
data_to_save = {'x_train': x_train,
                'y_train': y_train,
                'x_test' : x_test,
                'y_test' : y_test,
                'x_anomaly' : x_anomaly,
                'y_anomaly' : y_anomaly
                }
# data_to_save = {'all_dataset': all_dataset, 'substance_set': substance_set, 'train_folds': train_folds, 'test_folds': test_folds}
with open(pickle_filename, 'wb') as pickle_file:
    dill.dump(data_to_save, pickle_file)

