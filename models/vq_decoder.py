from functools import partial
import tensorflow as tf


def make_decoder(latent_size, z_dimension):
    deconv = partial(tf.keras.layers.Conv2DTranspose, padding="same", activation="relu")

    return tf.keras.Sequential(
        [
            tf.keras.layers.Reshape([1, 1, latent_size * z_dimension]),
            deconv(filters=64, kernel_size=7, padding="valid"),
            deconv(filters=64, kernel_size=5),
            deconv(filters=64, kernel_size=5, strides=2),
            deconv(filters=32, kernel_size=5),
            deconv(filters=32, kernel_size=5, strides=2),
            deconv(filters=32, kernel_size=5),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=5, padding="same", activation=None
            ),
        ]
    )
