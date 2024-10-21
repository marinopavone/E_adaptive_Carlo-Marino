import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from chemical_brother.adaptive_rbfnet import AdaptiveRBFNet


def main():
    chemical_sensors = pd.read_csv("chemical_brother/chemical_sensors.csv")
    names = [
        "OFFCHIP_GOLD_78kHz_IN-PHASE",
        "OFFCHIP_PLATINUM_200Hz_IN-PHASE",
        "OFFCHIP_PLATINUM_200Hz_QUADRATURE",
        "OFFCHIP_GOLD_200Hz_IN-PHASE",
        "OFFCHIP_GOLD_200Hz_QUADRATURE",
        "OFFCHIP_SILVER_200Hz_IN-PHASE",
        "OFFCHIP_SILVER_200Hz_QUADRATURE",
        "OFFCHIP_NICKEL_200Hz_IN-PHASE",
        "OFFCHIP_NICKEL_200Hz_QUADRATURE",
    ]
    data = chemical_sensors[names].to_numpy()

    labels = chemical_sensors[["CLASS"]].to_numpy()

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    data_scaled = scaler.fit_transform(data)

    encoded_labels = label_encoder.fit_transform(labels)

    data_scaled_train, data_scaled_test, labels_train, labels_test = train_test_split(
        data_scaled, encoded_labels, test_size=0.2, random_state=32
    )

    model = AdaptiveRBFNet(output_dim=7, initial_gamma=0.5)
    model.compile(
        optimizer=tf.keras.optimizers.Lion(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(data_scaled_train, labels_train, epochs=100, batch_size=128)
    # predictions = model.predict(data_scaled_test)
    # print(predictions)

    model.evaluate(data_scaled_test, labels_test)

    centers = model.adaptive_rbf_layer.centers.numpy()
    gamma = model.adaptive_rbf_layer.gamma.numpy()

    import matplotlib.pyplot as plt

    plt.imshow(centers, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Features")
    plt.ylabel("Classes")
    plt.show()

    combinations = list(itertools.combinations(range(centers.shape[1]), 2))
    for combination in combinations:
        plt.figure(figsize=(10, 8))
        for class_index in range(7):
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
        # plt.show()
        plt.savefig(
            f"class_centers_{names[combination[0]]}_{names[combination[1]]}.png",
            dpi=300,
        )


# print(centers)


if __name__ == "__main__":
    main()
