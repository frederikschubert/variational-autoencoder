from functools import partial

from tensorflow.keras import models, layers
import numpy as np

deconv = partial(
    layers.Conv2DTranspose, filters=64, kernel_size=4, activation="relu", padding="same"
)

dense = partial(layers.Dense, activation="relu")


def create_convolutional_decoder(data_shape):
    return models.Sequential(
        layers=[
            dense(np.product(data_shape)),
            layers.Reshape(data_shape),
            deconv(name="deconv1"),
            deconv(name="deconv2"),
            deconv(name="deconv3"),
            layers.Conv2DTranspose(
                filters=1, kernel_size=4, padding="same", name="output"
            ),
        ],
        name="decoder",
    )


def create_decoder(data_shape, z_dimension, conditional=False):
    inputs = layers.Input(shape=[z_dimension])
    if conditional:
        condition_inputs = layers.Input(shape=[10])
        inputs = [inputs, condition_inputs]
        concat = layers.Concatenate()(inputs)
        hidden1 = dense(256)(concat)
    else:
        hidden1 = dense(256)(inputs)
    hidden2 = dense(256)(hidden1)
    outputs = layers.Dense(np.product(data_shape))(hidden2)
    outputs = layers.Reshape(data_shape)(outputs)
    return models.Model(inputs, outputs, name="decoder")
