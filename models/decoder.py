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


def create_decoder(data_shape, z_dimension, conditioning_vector=None):
    z_inputs = layers.Input(shape=[z_dimension])
    if conditioning_vector is not None:
        condition_inputs = layers.Input(shape=[1])
        concat = layers.Concatenate()([z_inputs, condition_inputs])
        hidden1 = layers.Dense(256, activation="relu")(concat)
    else:
        hidden1 = layers.Dense(256, activation="relu")(z_inputs)
    hidden2 = layers.Dense(256, activation="relu")(hidden1)
    outputs = layers.Dense(np.product(data_shape))(hidden2)
    outputs = layers.Reshape(data_shape)(outputs)
    if conditioning_vector is not None:
        return models.Model([z_inputs, condition_inputs], outputs, name="decoder")
    else:
        return models.Model(z_inputs, outputs, name="decoder")
