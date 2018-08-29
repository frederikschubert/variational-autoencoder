from tensorflow.keras import models, layers


def create_convolutional_encoder(data_shape, z_dimension):
    inputs = layers.Input(shape=data_shape)
    conv1 = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation="relu",
        padding="same",
        name="conv1",
    )(inputs)
    conv2 = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation="relu",
        padding="same",
        name="conv2",
    )(conv1)
    conv3 = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation="relu",
        padding="same",
        name="conv3",
    )(conv2)
    flatten = layers.Flatten()(conv3)
    z_mean = layers.Dense(z_dimension)(flatten)
    z_log_variance = layers.Dense(z_dimension)(flatten)
    return models.Model(inputs, [z_mean, z_log_variance], name="encoder")


def create_encoder(data_shape, z_dimension):
    inputs = layers.Input(shape=data_shape)
    flatten = layers.Flatten()(inputs)
    hidden1 = layers.Dense(256, activation="relu")(flatten)
    hidden2 = layers.Dense(256, activation="relu")(hidden1)
    z_mean = layers.Dense(z_dimension)(hidden2)
    z_log_variance = layers.Dense(z_dimension)(hidden2)
    return models.Model(inputs, [z_mean, z_log_variance], name="encoder")
