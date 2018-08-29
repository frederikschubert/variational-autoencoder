from tensorflow import keras


class Encoder(keras.Model):
    def __init__(self, z_dimension):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(256, activation="relu")
        self.hidden2 = keras.layers.Dense(256, activation="relu")
        self.z_mean = keras.layers.Dense(z_dimension)
        self.z_log_variance = keras.layers.Dense(z_dimension)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.z_mean(x), self.z_log_variance(x)
