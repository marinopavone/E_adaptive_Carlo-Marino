import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from chemical_brother.RBF.rbfnet import RBFNet


def main():
    chemical_sensors = pd.read_csv("chemical_brother/chemical_sensors.csv")
    data = chemical_sensors[
        [
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
    ].to_numpy()

    labels = chemical_sensors[["CLASS"]].to_numpy()

    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    data_scaled = scaler.fit_transform(data)

    encoded_labels = label_encoder.fit_transform(labels)

    data_scaled_train, data_scaled_test, labels_train, labels_test = train_test_split(
        data_scaled, encoded_labels, test_size=0.2
    )

    model = RBFNet(output_dim=7, gamma=0.05)
    model.compile(
        optimizer=tf.keras.optimizers.Lion(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(data_scaled_train, labels_train, epochs=1000, batch_size=128)
    predictions = model.predict(data_scaled_test)
    # print(predictions)

    model.evaluate(data_scaled_test, labels_test)

    centers = model.rbf_layer.centers.numpy()
    import matplotlib.pyplot as plt

    plt.imshow(centers, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Features")
    plt.ylabel("Classes")
    plt.show()
    # print(centers)


if __name__ == "__main__":
    main()
