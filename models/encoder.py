from functools import partial
from tensorflow.keras import models, layers

conv = partial(
    layers.Conv2D,
    filters=64,
    kernel_size=4,
    strides=2,
    activation="relu",
    padding="same",
)

dense = partial(layers.Dense, activation="relu")


def create_convolutional_encoder(data_shape, z_dimension):
    inputs = layers.Input(shape=data_shape)
    conv1 = conv(name="conv1")(inputs)
    conv2 = conv(name="conv2")(conv1)
    conv3 = conv(name="conv3")(conv2)
    flatten = layers.Flatten()(conv3)
    z_mean = layers.Dense(z_dimension)(flatten)
    z_log_variance = layers.Dense(z_dimension)(flatten)
    return models.Model(inputs, [z_mean, z_log_variance], name="encoder")


def create_encoder(data_shape, z_dimension, conditional=None):
    inputs = layers.Input(shape=data_shape)
    flatten = layers.Flatten()(inputs)
    if conditional is not None:
        condition_inputs = layers.Input(shape=[10])
        inputs = [inputs, condition_inputs]
        flatten = layers.Concatenate()([flatten, condition_inputs])
    hidden1 = dense(256)(flatten)
    hidden2 = dense(256)(hidden1)
    z_mean = layers.Dense(z_dimension)(hidden2)
    z_log_variance = layers.Dense(z_dimension)(hidden2)
    outputs = [z_mean, z_log_variance]
    return models.Model(inputs, outputs, name="encoder")
