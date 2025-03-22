from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from chemical_brother.data_maker import DataMaker, ChemicalClass
# from chemical_brother.deep_clustering import DeepClustering
from chemical_brother.marino_sil_AUTOENC import sil_AUTOENC
from chemical_brother.marino_loss import CustomLoss
import tensorflow as tf
import numpy as np
#%%   Dataset building

data_maker = DataMaker("../dataset/")
data_maker.set_contamination_classes(
    [
        ChemicalClass.SODIUM_HYDROXIDE,
        ChemicalClass.SODIUM_CHLORIDE,
        ChemicalClass.SODIUM_HYPOCHLORITE,
        ChemicalClass.POTASSIUM_NITRATE,
        ChemicalClass.CALCIUM_NITRATE,
        # ChemicalClass.POTABLE_WATER,
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
data = data_maker.make_full_dataset()
labels = data[["CLASS"]].to_numpy()
data = data.drop(columns=["CLASS"]).to_numpy()
#%%   encoding and normalization

scaler = MinMaxScaler()

label_encoder = LabelEncoder()
label_encoder.fit(labels)

scaler.fit(data)
# train, test = data_maker.make_train_test_experiments(3, 500)
test_experiment_selection=[1,3,5]
train, test = data_maker.split_train_test_by_experiment(test_experiment_selection, 500)
x_train = train.drop(columns=["CLASS"]).to_numpy()
y_train = train[["CLASS"]].to_numpy()
x_test = test.drop(columns=["CLASS"]).to_numpy()
y_test = test[["CLASS"]].to_numpy()

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
#%%   inizialize training

n_classes = len(label_encoder.classes_)
X_dim = x_train.shape[1]
centroids_per_class=3

encoded_dim = 7
Layers_dim = [30, 15, encoded_dim, 15, 30]

sil_autoenc_model = sil_AUTOENC(
    reconstrunction_dim=X_dim,
    n_classes=n_classes,
    layers_dim=Layers_dim,
    centroids_per_class = centroids_per_class
)


sil_autoenc_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    # loss=["sparse_categorical_crossentropy", "mse"],
    metrics=["mae"],
    loss={'output_Z' : CustomLoss(X_dim = X_dim, Z_dim =encoded_dim),
        'output_Xrec': 'mse',
        # 'output_RBF' : 'sparse_categorical_crossentropy'
          },
    loss_weights={'output_Z': 0.7,
                  'output_Xrec': 0.3,
                  # 'output_RBF': 0.1
                  }
)

sil_autoenc_model.summary()
#%%   start training
y_tr=y_train.reshape([len(y_train),1])
y_ts=y_test.reshape([len(y_test),1])

new_y = np.concatenate((x_train, y_tr), axis=1)
new_y_test = np.concatenate((x_test, y_ts), axis=1)

sil_autoenc_model.fit(
    # x_train,
    # new_y,
    x_train,
    [x_train, y_train],
    epochs=100,
    batch_size=500,
    # validation_data=(x_test, [y_test, x_test]),
    validation_data=(x_test,new_y_test),
)


