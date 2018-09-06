from logging import Logger
from typing import List

import tensorflow as tf
import numpy as np

from sacred.stflow import LogFileWriter

from experiment import ex, create_dataset, run_training, create_writer
import models


def create_elbo(
    x,
    p_x_given_z: tf.distributions.Distribution,
    q_z_given_x: tf.distributions.Distribution,
    p_z: tf.distributions.Distribution,
    beta,
):
    kl_divergence = tf.reduce_sum(
        tf.distributions.kl_divergence(q_z_given_x, p_z), axis=1
    )
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), axis=[1, 2, 3])

    tf.summary.scalar("kl_divergence", tf.reduce_mean(beta * kl_divergence))
    tf.summary.scalar(
        "negative_log_likelihood", tf.reduce_mean(-expected_log_likelihood)
    )
    return tf.reduce_sum(expected_log_likelihood - beta * kl_divergence, axis=0)


@ex.automain
@LogFileWriter(ex)
def train(
    _log: Logger,
    z_dimension: int,
    data_shape: List[int],
    mode: str,
    beta: float,
    batch_size: int,
    binary: bool,
):
    train_dataset, _ = create_dataset()

    writer = create_writer()

    _log.info("Building computation graph...")

    if mode == "conditional":
        x, y = train_dataset.batch(batch_size).make_one_shot_iterator().get_next()
        y = tf.one_hot(y, 10, on_value=1.0, off_value=0.0)
    else:
        x = train_dataset.batch(batch_size).make_one_shot_iterator().get_next()
        y = None

    if mode == "convolutional":
        encoder = models.create_convolutional_encoder(data_shape, z_dimension)
        decoder = models.create_convolutional_decoder(data_shape)
    else:
        encoder = models.create_encoder(data_shape, z_dimension, y is not None)
        _log.info(encoder.summary())
        decoder = models.create_decoder(data_shape, z_dimension, y is not None)
        _log.info(decoder.summary())

    with tf.name_scope("data"):
        tf.summary.image("mnist_image", x)

    with tf.variable_scope("variational"):
        if mode == "conditional":
            z_mean, z_log_variance = encoder([x, y])
        else:
            z_mean, z_log_variance = encoder(x)

        q_z_given_x = tf.distributions.Normal(
            loc=z_mean, scale=tf.exp(0.5 * z_log_variance)
        )

    with tf.variable_scope("posterior", reuse=True):
        if mode == "conditional":
            p_x_given_z_theta = decoder([q_z_given_x.sample(), y])
        else:
            p_x_given_z_theta = decoder(q_z_given_x.sample())

        if binary:
            p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_theta)
        else:
            p_x_given_z = tf.distributions.Normal(
                loc=p_x_given_z_theta,
                scale=np.ones(shape=data_shape, dtype=np.float32) * 0.1,
            )
        posterior_sample = p_x_given_z.sample()
        # Plot a sample from the posterior for comparison with the input data
        tf.summary.image("posterior_sample", tf.cast(posterior_sample, tf.float32))

    with tf.variable_scope("prior", reuse=True):
        p_z = tf.distributions.Normal(
            loc=np.zeros(shape=z_dimension, dtype=np.float32),
            scale=np.ones(shape=z_dimension, dtype=np.float32),
        )
        if mode == "conditional":
            for n in range(10):
                condition_number = tf.expand_dims(
                    tf.one_hot(tf.constant(n), 10, on_value=1.0, off_value=0.0), axis=0
                )
                p_x_given_prior_z = decoder(
                    [tf.expand_dims(p_z.sample(), axis=0), condition_number]
                )

                if binary:
                    p_x_given_prior_z = tf.distributions.Bernoulli(
                        logits=p_x_given_prior_z
                    )
                else:
                    p_x_given_prior_z = tf.distributions.Normal(
                        loc=p_x_given_prior_z,
                        scale=np.ones(shape=data_shape, dtype=np.float32) * 0.1,
                    )
                p_x_given_prior_z_sample = p_x_given_prior_z.sample()
                # Plot a sample given a random prior to check whether it is similar to the input data
                tf.summary.image(
                    f"prior_sample_conditioned_on_{n}",
                    tf.cast(p_x_given_prior_z_sample, tf.float32),
                )
        else:
            p_x_given_prior_z = decoder(tf.expand_dims(p_z.sample(), axis=0))
            if binary:
                p_x_given_prior_z = tf.distributions.Bernoulli(logits=p_x_given_prior_z)
            else:
                p_x_given_prior_z = tf.distributions.Normal(
                    loc=p_x_given_prior_z,
                    scale=np.ones(shape=data_shape, dtype=np.float32) * 0.1,
                )

            p_x_given_prior_z_sample = p_x_given_prior_z.sample()
            # Plot a sample given a random prior to check whether it is similar to the input data
            tf.summary.image(
                "prior_sample", tf.cast(p_x_given_prior_z_sample, tf.float32)
            )

    elbo = create_elbo(x, p_x_given_z, q_z_given_x, p_z, beta)

    train_op = tf.train.AdamOptimizer().minimize(-elbo)
    summary_op = tf.summary.merge_all()

    run_training(train_op, summary_op, writer)

