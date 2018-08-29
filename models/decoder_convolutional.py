from tensorflow import keras
import numpy as np


class DecoderConvolutional(keras.Model):
    def __init__(self, output_shape):
        super().__init__()
        self.reshaped1 = keras.layers.Dense(np.product(output_shape), activation="relu")
        self.reshaped2 = keras.layers.Reshape(output_shape)
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
            filters=1, kernel_size=4, padding="same", name="output"
        )

    def call(self, inputs):
        x = self.reshaped1(inputs)
        x = self.reshaped2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.deconv4(x)
