from tensorflow import keras
import numpy as np


class Decoder(keras.Model):
    def __init__(self, output_shape, convolutional):
        super().__init__()
        self.convolutional = convolutional
        self.hidden1 = keras.layers.Dense(256)
        self.hidden2 = keras.layers.Dense(256)
        self.hidden3 = keras.layers.Dense(np.product(output_shape))
        self.reshaped = keras.layers.Reshape(output_shape)
        self.deconv1 = keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, activation="relu", padding="same", name="deconv1"
        )
        self.deconv2 = keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, activation="relu", padding="same", name="deconv2"
        )
        self.deconv3 = keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, activation="relu", padding="same", name="deconv3"
        )
        self.deconv4 = keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=1,
            activation="sigmoid",
            padding="same",
            name="output",
        )

    def call(self, inputs):
        if self.convolutional:
            x = self.hidden3(inputs)
            x = self.reshaped(x)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            return self.deconv4(x)
        else:
            x = self.hidden1(inputs)
            x = self.hidden2(x)
            x = self.hidden3(x)
            return self.reshaped(x)
