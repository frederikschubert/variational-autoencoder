from tensorflow.keras import models, layers
import numpy as np


def create_convolutional_decoder(data_shape):
    return models.Sequential(
        layers=[
            layers.Dense(np.product(data_shape), activation="relu"),
            layers.Reshape(data_shape),
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                activation="relu",
                padding="same",
                name="deconv1",
            ),
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                activation="relu",
                padding="same",
                name="deconv2",
            ),
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                activation="relu",
                padding="same",
                name="deconv3",
            ),
            layers.Conv2DTranspose(
                filters=1, kernel_size=4, padding="same", name="output"
            ),
        ],
        name="decoder",
    )


def create_decoder(data_shape):
    return models.Sequential(
        layers=[
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(np.product(data_shape)),
            layers.Reshape(data_shape),
        ],
        name="decoder",
    )
