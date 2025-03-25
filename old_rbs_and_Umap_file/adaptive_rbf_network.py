import itertools
import os

import numpy as np
import tensorflow as tf
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

from chemical_brother.AdaptiveRBF.adaptive_rbfnet import AdaptiveRBFNet
from chemical_brother.data_maker import DataMaker, ChemicalClass
from chemical_brother.old_stuf_di_carlo.loss import SilhouetteLoss, CorelLoss


def main():
    # chemical_sensors = pd.read_csv("chemical_brother/chemical_sensors.csv")
    for file in os.scandir("../figures/"):
        if file.is_file():
            os.unlink(file.path)

    data_maker = DataMaker("../dataset/")
    data_maker.set_contamination_classes(
        [
            ChemicalClass.SODIUM_HYDROXIDE,
            ChemicalClass.SODIUM_CHLORIDE,
            ChemicalClass.SODIUM_HYPOCHLORITE,
        ]
    )
    chemical_sensors = data_maker.make_steady_state_dataset(700)

    labels = chemical_sensors[["CLASS"]].to_numpy()
    data = chemical_sensors.drop(columns=["CLASS"]).to_numpy()
    names = chemical_sensors.drop(columns=["CLASS"]).columns.to_numpy()

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    data_scaled = scaler.fit_transform(data)

    encoded_labels = label_encoder.fit_transform(labels)

    n_neighbors = 100
    umap_reducer = umap.UMAP(n_components=4, n_neighbors=n_neighbors, n_jobs=-1)
    data_scaled = umap_reducer.fit_transform(data_scaled)

    data_scaled_train, data_scaled_test, labels_train, labels_test = train_test_split(
        data_scaled, encoded_labels, test_size=0.2, random_state=32
    )

    model = AdaptiveRBFNet(
        output_dim=label_encoder.classes_.shape[0], initial_gamma=1.0
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(data_scaled_train, labels_train, epochs=1000, batch_size=256)
    # predictions = model.predict(data_scaled_test)
    # print(predictions)

    model.evaluate(data_scaled_test, labels_test)

    centers = model.adaptive_rbf_layer.centers.numpy()
    gamma = model.adaptive_rbf_layer.gamma.numpy()

    plt.imshow(centers, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Features")
    plt.ylabel("Classes")
    plt.show()

    combinations = list(itertools.combinations(range(centers.shape[1]), 2))
    for combination in combinations:
        plt.figure(figsize=(10, 8))
        for class_index in range(label_encoder.classes_.shape[0]):
            class_data = data_scaled_train[labels_train == class_index]
            plt.scatter(
                class_data[:, combination[0]],
                class_data[:, combination[1]],
                label=f"Class {class_index+1}",
                c=[f"C{class_index}"],
            )
        for class_index, center in enumerate(centers):
            plt.scatter(
                center[combination[0]],
                center[combination[1]],
                label=f"Class {class_index+1} Center",
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
            f"Clustering Centers (Feature {combination[0]+1} vs Feature {combination[1]+1})"
        )
        plt.xlabel(f"{names[combination[0]]}")
        plt.ylabel(f"{names[combination[1]]}")
        plt.legend()
        plt.savefig(
            f"figures/class_centers_{names[combination[0]]}_{names[combination[1]]}.png",
            dpi=300,
        )
        plt.show()


# print(centers)


if __name__ == "__main__":
    main()
