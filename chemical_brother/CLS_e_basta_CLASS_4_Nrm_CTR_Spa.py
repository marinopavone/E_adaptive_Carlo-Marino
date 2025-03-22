import pandas as pd
import tensorflow as tf
import numpy as np

class CLS_neural_network(tf.keras.Model):
    def __init__(self, num_classes, hidden_units=10, ):
        super(CLS_neural_network, self).__init__()
        tf.random.set_seed(42)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.hidden_l1 = tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(hidden_units, activation=tf.nn.atan)
        #
        self.classifier_out = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inp):
        x_reshaped = self.flatten_layer(inp)
        x_hidden = self.hidden_l1(x_reshaped)
        x_hidden = self.hidden_l2(x_hidden)
        x_class = self.classifier_out(x_hidden)

        return x_class, x_hidden

    def get_config(self):
        return {}
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Loss function combining reconstruction and classification loss
def loss( x_class,labels):
    classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, x_class))
    # total_loss = lambda_rec * reconstruction_loss + lambda_class * classification_loss
    return classification_loss


# Gradient computation
def grads(model, inputs, labels):
    with tf.GradientTape() as tape:
        x_class, x_hidden = model(inputs)
        loss_value = loss(x_class,labels)
    return loss_value, tape.gradient(loss_value,model.trainable_variables)


# Training function
def CLS_neural_network_training(x_train, y_train, num_epochs=200,
                             batch_size=128, learning_rate=0.001, momentum=0.9,
                             num_classes=5, hidden_units=10):
    model = CLS_neural_network(num_classes, hidden_units)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    global_step = tf.Variable(0)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for i in range(0, len(x_train), batch_size):
            x_inp = x_train[i: i + batch_size]
            y_inp = y_train[i: i + batch_size]
            loss_value, grad = grads(model, x_inp, y_inp)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        print("Step: {}, Loss: {}".format(global_step.numpy(), tf.reduce_sum(loss_value)))
        loss_value_final = tf.reduce_sum(loss_value)
    return model, loss_value_final


def test_CLS_neural_network(model, x_test, y_test):
    # Perform model predictions
    classifications,_ = model.predict(x_test)

    # Compute Mean Squared Error (MSE) loss for reconstruction
    predicted_classes = np.argmax(classifications, axis=1)  # Convert softmax scores to class indices
    labels = np.argmax(y_test, axis=1)  # Convert softmax scores to class indices

    # Unique classes
    classes = np.unique(labels)

    loss_dict = {}
    global_reconstruction_loss = []
    global_accuracy = []

    print("\nReconstruction & Classification Loss Comparison per Class:\n")
    for cls in classes:
        # Get indices for current class
        cls_indices = np.where(labels == cls)[0]

        # Compute classification accuracy for this class
        cls_accuracy = np.mean(predicted_classes[cls_indices] == labels[cls_indices])
        global_accuracy.append(cls_accuracy)

        loss_dict[cls] = {
            "classification_accuracy": global_accuracy
        }

        print(f"Class {cls}: Classification Accuracy = {cls_accuracy:.2%}")

    # Compute global reconstruction loss & classification accuracy
    global_classification_accuracy = np.mean(global_accuracy)

    print(f"\nGlobal Classification Accuracy = {global_classification_accuracy:.2%}")

    return global_classification_accuracy



