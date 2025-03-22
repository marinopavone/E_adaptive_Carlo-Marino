from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from chemical_brother.data_maker import DataMaker, ChemicalClass
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


        ChemicalClass.CALCIUM_NITRATE,

        ChemicalClass.HYDROCHLORIC_ACID,
        ChemicalClass.HYDROGEN_PEROXIDE,
        ChemicalClass.ETHANOL,
        ChemicalClass.AMMONIA,
    ]
)
#%%   Dataset anomaly substances

anomaly_sub = DataMaker("/Users/marinopavone/PycharmProjects/E_adaptive_Carlo-Marino/dataset")
anomaly_sub.set_contamination_classes(
    [
        ChemicalClass.NELSEN,
        ChemicalClass.ACETIC_ACID,

        ChemicalClass.FORMIC_ACID,

        ChemicalClass.POTASSIUM_NITRATE,


        ChemicalClass.POTABLE_WATER,
        ChemicalClass.SODIUM_HYDROXIDE,
        ChemicalClass.SODIUM_CHLORIDE,
        ChemicalClass.ACETONE,
        ChemicalClass.SODIUM_HYPOCHLORITE,
        ChemicalClass.PHOSPHORIC_ACID,
    ]
)

test_experiment_selection=[4,8,9]  # seleziono quale dei 10 fold usare come test (gli altri andranno come training)
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
from time import gmtime, strftime
time_trak= strftime("%Y-%m-%d %H:%M:%S", gmtime())



print("saving TRAIN TEST ")
path ='Datasets Pikle/X&Y_ds_created_'
pickle_filename = path + time_trak + '.pkl'
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

Dataset_info_file_name = path + time_trak + '.txt'
with open(Dataset_info_file_name, "w") as f:
    s = ["First line of text.\n", "Second line of text.\n", "Third line of text.\n"]
    f.write(f"Substances \n")
    for i, cls in enumerate(label_encoder.classes_):
        f.write(f" {i}: = {cls} \n")
    f.write(f"Anomalies \n")
    for i, cls in enumerate(label_encoder_anomaly.classes_):
        f.write(f" {i}: = {cls} \n")

    f.write(f"\n")

    f.write(f"fold for TEST SET: {test_experiment_selection} \n")
    f.write(f"from sample {stady_state_start} to sample {stady_state_end} \n")

    f.write(f"\n")
    f.write(f"e mo so cazzi tuoi \n")
