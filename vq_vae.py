from logging import Logger
from typing import List
from functools import partial

import tensorflow as tf
import numpy as np

from sacred.stflow import LogFileWriter

from experiment import ex, create_dataset, run_training, create_writer


class VectorQuantizer:
    def __init__(self, num_codes, z_dimension):
        self.num_codes = num_codes
        self.z_dimension = z_dimension
        self.codebook = tf.get_variable(
            "codebook", shape=[num_codes, z_dimension], dtype=tf.float32
        )
        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.train_op = self.ema.apply([self.codebook])

    def get_codebook_entries(self, one_hot_assignments):
        return tf.reduce_sum(
            tf.expand_dims(
                one_hot_assignments, -1
            )  # [batch_size, latent_size, num_codes, 1]
            * tf.reshape(
                self.ema.average(self.codebook),
                [1, 1, self.num_codes, self.z_dimension],
            ),
            axis=2,
        )

    def __call__(self, codes):
        distances = tf.norm(
            tf.expand_dims(codes, 2)  # [batch_size, latent_size, 1, z_dimension]
            - tf.reshape(
                self.ema.average(self.codebook),
                [1, 1, self.num_codes, self.z_dimension],
            ),
            axis=-1,
        )
        code_assignments = tf.argmin(distances, -1)
        one_hot_code_assignments = tf.one_hot(code_assignments, depth=self.num_codes)
        nearest_codebook_entries = self.get_codebook_entries(one_hot_code_assignments)
        return nearest_codebook_entries, one_hot_code_assignments


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


@ex.automain
@LogFileWriter(ex)
def train(
    _log: Logger,
    num_codes: int,
    latent_size: int,
    z_dimension: int,
    batch_size: int,
    beta: float,
):
    train_dataset, _ = create_dataset()

    x = train_dataset.batch(batch_size).make_one_shot_iterator().get_next()

    with tf.name_scope("data"):
        tf.summary.image("mnist_image", x)

    writer = create_writer()

    _log.info("Building computation graph...")
    encoder = make_encoder(latent_size, z_dimension)
    decoder = make_decoder(latent_size, z_dimension)
    quantizer = VectorQuantizer(num_codes, z_dimension)

    codes = encoder(x)
    nearest_codebook_entries, one_hot_code_assignments = quantizer(codes)

    codes_straight_through = codes + tf.stop_gradient(nearest_codebook_entries - codes)
    decoder_distribution = tf.distributions.Bernoulli(
        logits=decoder(codes_straight_through)
    )

    print(encoder.summary())

    print(decoder.summary())

    with tf.variable_scope("posterior", reuse=True):
        posterior_sample = decoder_distribution.sample()
        tf.summary.image("posterior_sample", tf.cast(posterior_sample, tf.float32))

    reconstruction_loss = -tf.reduce_mean(decoder_distribution.log_prob(x))
    commitment_loss = tf.reduce_mean(
        tf.square(codes - tf.stop_gradient(nearest_codebook_entries))
    )

    # Uniform prior over codes
    prior_distribution = tf.distributions.Multinomial(
        total_count=1.0, logits=tf.zeros([latent_size, num_codes])
    )
    prior_loss = -tf.reduce_mean(
        tf.reduce_sum(prior_distribution.log_prob(one_hot_code_assignments), 1)
    )

    with tf.variable_scope("prior", reuse=True):
        prior_sample_codes = quantizer.get_codebook_entries(
            prior_distribution.sample(1)
        )
        prior_decoder_distribution = tf.distributions.Bernoulli(
            logits=decoder(prior_sample_codes)
        )
        prior_sample = prior_decoder_distribution.sample()
        tf.summary.image("prior_sample", tf.cast(prior_sample, tf.float32))

    loss = reconstruction_loss + beta * commitment_loss + prior_loss

    tf.summary.scalar("losses/total_loss", loss)
    tf.summary.scalar("losses/reconstruction_loss", reconstruction_loss)
    tf.summary.scalar("losses/commitment_loss", beta * commitment_loss)

    train_op = tf.group(quantizer.train_op, tf.train.AdamOptimizer().minimize(loss))
    summary_op = tf.summary.merge_all()

    run_training(train_op, summary_op, writer)

