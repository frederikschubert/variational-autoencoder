from tensorflow import keras
import numpy as np


class DecoderDiscrete(keras.Model):
    def __init__(self, output_shape):
        super().__init__()
        self.hidden1 = keras.layers.Dense(256, activation="relu")
        self.hidden2 = keras.layers.Dense(256, activation="relu")
        self.reshaped1 = keras.layers.Dense(np.product(output_shape))
        self.reshaped2 = keras.layers.Reshape(output_shape)

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.reshaped1(x)
        return self.reshaped2(x)
