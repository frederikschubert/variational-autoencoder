from tensorflow import keras
import numpy as np


class Encoder(keras.Model):
    def __init__(self, z_dimension):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            padding="same",
            name="conv1",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            padding="same",
            name="conv2",
        )
        self.conv3 = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            padding="same",
            name="conv3",
        )
        self.hidden = keras.layers.Flatten()
        self.z_mean = keras.layers.Dense(z_dimension)
        self.z_log_variance = keras.layers.Dense(z_dimension)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.hidden(x)
        return self.z_mean(x), self.z_log_variance(x)
