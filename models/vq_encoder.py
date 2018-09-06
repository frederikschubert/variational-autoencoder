from functools import partial
import tensorflow as tf


def make_encoder(latent_size, z_dimension):
    conv = partial(tf.keras.layers.Conv2D, padding="same", activation="relu")

    return tf.keras.Sequential(
        [
            conv(filters=32, kernel_size=5),
            conv(filters=32, kernel_size=5, strides=2),
            conv(filters=64, kernel_size=5),
            conv(filters=64, kernel_size=5, strides=2),
            conv(filters=64, kernel_size=7, padding="valid"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_size * z_dimension),
            tf.keras.layers.Reshape([latent_size, z_dimension]),
        ]
    )
