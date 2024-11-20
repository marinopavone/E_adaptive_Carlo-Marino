from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from chemical_brother.data_maker import DataMaker, ChemicalClass
from chemical_brother.deep_clustering import DeepClustering
import tensorflow as tf


def main():
    data_maker = DataMaker("dataset/")
    data_maker.set_contamination_classes(
        [
            ChemicalClass.SODIUM_HYDROXIDE,
            ChemicalClass.SODIUM_CHLORIDE,
            ChemicalClass.SODIUM_HYPOCHLORITE,
            ChemicalClass.POTASSIUM_NITRATE,
            ChemicalClass.CALCIUM_NITRATE,
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
    data = data_maker.make_full_dataset()
    labels = data[["CLASS"]].to_numpy()
    data = data.drop(columns=["CLASS"]).to_numpy()

    scaler = MinMaxScaler()

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    scaler.fit(data)
    train, test = data_maker.make_train_test_experiments(2, 300)
    x_train = train.drop(columns=["CLASS"]).to_numpy()
    y_train = train[["CLASS"]].to_numpy()
    x_test = test.drop(columns=["CLASS"]).to_numpy()
    y_test = test[["CLASS"]].to_numpy()

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    reconstruction_dim = len(train.drop(columns=["CLASS"]).columns)
    n_classes = len(label_encoder.classes_)

    deep_clustering_model = DeepClustering(
        reconstrunction_dim=reconstruction_dim,
        n_classes=n_classes,
        centroids_per_class=3,
    )

    deep_clustering_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=["sparse_categorical_crossentropy", "mse"],
        metrics=["accuracy", "mae"],
    )

    deep_clustering_model.summary()

    deep_clustering_model.fit(
        x_train,
        [y_train, x_train],
        epochs=100,
        batch_size=64,
        validation_data=(x_test, [y_test, x_test]),
    )


if __name__ == "__main__":
    main()
