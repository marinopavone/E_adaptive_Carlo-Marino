import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from chemical_brother.data_maker import DataMaker, ChemicalClass
from chemical_brother.deep_clustering import DeepClustering
import tensorflow as tf

from chemical_brother.loss import SilhouetteLoss


def main():
    data_maker = DataMaker("dataset/")
    data_maker.set_contamination_classes(
        [
            ChemicalClass.SODIUM_HYDROXIDE,
            ChemicalClass.SODIUM_CHLORIDE,
            ChemicalClass.POTASSIUM_NITRATE,
            ChemicalClass.CALCIUM_NITRATE,
        ]
    )
    data = data_maker.make_full_dataset()
    labels = data[["CLASS"]].to_numpy()
    names = data.drop(columns=["CLASS"]).columns.to_numpy()
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
        centroids_per_class=1,
    )

    deep_clustering_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=[SilhouetteLoss(), "mae"],
        loss_weights=[0.2, 0.8],
        metrics=["accuracy", "mse"],
    )

    deep_clustering_model.summary()
    deep_clustering_model.fit(
        x_train,
        [y_train, x_train],
        epochs=10,
        batch_size=64,
        validation_data=(x_test, [y_test, x_test]),
    )

    zeta = deep_clustering_model.encoder.predict(x_test)

    plt.scatter(zeta[:, 0], zeta[:, 1])
    plt.show()

    centers = deep_clustering_model.rbf_layer.centers.numpy()
    gamma = deep_clustering_model.rbf_layer.gamma.numpy()

    clust, rec = deep_clustering_model.predict(x_test)

    plt.imshow(centers, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Features")
    plt.ylabel("Classes")
    plt.show()

    combinations = list(itertools.combinations(range(centers.shape[1]), 2))
    for combination in combinations:
        plt.figure(figsize=(10, 8))
        for class_index in range(label_encoder.classes_.shape[0]):
            class_data = x_test[y_test == class_index]
            zeta_class = deep_clustering_model.encoder.predict(class_data)
            plt.scatter(
                class_data[:, combination[0]],
                class_data[:, combination[1]],
                label=f"Class {class_index + 1}",
                c=[f"C{class_index}"],
            )
            plt.scatter(
                zeta_class[:, combination[0]],
                zeta_class[:, combination[1]],
                label=f"Class {class_index + 1} Zeta",
                c=[f"C{class_index}"],
                marker="v",
            )
        for class_index, center in enumerate(centers):
            plt.scatter(
                center[combination[0]],
                center[combination[1]],
                label=f"Class {class_index + 1} Center",
                c=[f"C{class_index}"],
                marker="x",
            )
            circle = plt.Circle(
                (center[combination[0]], center[combination[1]]),
                1 / np.sqrt(2 * gamma[class_index]),
                color=f"C{class_index}",
                fill=True,
                alpha=0.1,
                linewidth=1,
            )
            plt.gca().add_artist(circle)
        plt.title(
            f"Clustering Centers (Feature {combination[0] + 1} vs Feature {combination[1] + 1})"
        )
        plt.xlabel(f"{names[combination[0]]}")
        plt.ylabel(f"{names[combination[1]]}")
        plt.legend()
        plt.savefig(
            f"figures/class_centers_{names[combination[0]]}_{names[combination[1]]}.png",
            dpi=300,
        )
        plt.show()


if __name__ == "__main__":
    main()
