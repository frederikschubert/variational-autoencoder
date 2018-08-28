from tensorflow import keras


class Encoder(keras.Model):
    def __init__(self, z_dimension, convolutional):
        super().__init__()
        self.convolutional = convolutional
        self.hidden1 = keras.layers.Dense(256, activation="relu")
        self.hidden2 = keras.layers.Dense(256, activation="relu")
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
        self.flatten = keras.layers.Flatten()
        self.z_mean = keras.layers.Dense(z_dimension)
        self.z_log_variance = keras.layers.Dense(z_dimension)

    def call(self, inputs):
        if self.convolutional:
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            return self.z_mean(x), self.z_log_variance(x)
        else:
            x = self.flatten(inputs)
            x = self.hidden1(x)
            x = self.hidden2(x)
            return self.z_mean(x), self.z_log_variance(x)
