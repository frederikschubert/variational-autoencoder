from logging import Logger
from typing import List

import tensorflow as tf
import numpy as np

from sacred.stflow import LogFileWriter

from experiment import ex, create_dataset, run_training
import models


@ex.automain
@LogFileWriter(ex)
def train(
    _log: Logger,
    z_dimension: int,
    data_shape: List[int],
    batch_size: int,
    mode: str,
    beta: float,  # pylint: disable=C0330
):
    dataset = create_dataset()

    _log.info("Building computation graph...")

    if mode == "conditional":
        x, y = dataset.make_one_shot_iterator().get_next()
        y = tf.one_hot(y, 10, on_value=1.0, off_value=0.0)
    else:
        x = dataset.make_one_shot_iterator().get_next()

    if mode == "convolutional":
        encoder = models.create_convolutional_encoder(data_shape, z_dimension)
        decoder = models.create_convolutional_decoder(data_shape)
    else:
        encoder = models.create_encoder(data_shape, z_dimension, y)
        decoder = models.create_decoder(data_shape, z_dimension, y)

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
            p_x_given_z_logits = decoder([q_z_given_x.sample(), y])
        else:
            p_x_given_z_logits = decoder(q_z_given_x.sample())

        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
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
                condition_number = tf.expand_dims(tf.one_hot(tf.constant(n), 10, on_value=1.0, off_value=0.0), axis=0)
                p_x_given_prior_z_logits = decoder(
                    [tf.expand_dims(p_z.sample(), axis=0), condition_number]
                )
                p_x_given_prior_z = tf.distributions.Bernoulli(logits=p_x_given_prior_z_logits)
                p_x_given_prior_z_sample = p_x_given_prior_z.sample()
                # Plot a sample given a random prior to check whether it is similar to the input data
                tf.summary.image(f"prior_sample_{n}", tf.cast(p_x_given_prior_z_sample, tf.float32))
        else:
            p_x_given_prior_z_logits = decoder(tf.expand_dims(p_z.sample(), axis=0))
            p_x_given_prior_z = tf.distributions.Bernoulli(logits=p_x_given_prior_z_logits)
            p_x_given_prior_z_sample = p_x_given_prior_z.sample()
            # Plot a sample given a random prior to check whether it is similar to the input data
            tf.summary.image("prior_sample", tf.cast(p_x_given_prior_z_sample, tf.float32))

    kl_divergence = tf.reduce_sum(
        tf.distributions.kl_divergence(q_z_given_x, p_z), axis=1
    )
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), axis=[1, 2, 3])

    elbo = tf.reduce_sum(expected_log_likelihood - beta * kl_divergence, axis=0)
    tf.summary.scalar("kl_divergence", tf.reduce_mean(beta * kl_divergence))
    tf.summary.scalar(
        "negative_log_likelihood", tf.reduce_mean(-expected_log_likelihood)
    )

    train_op = tf.train.AdamOptimizer().minimize(-elbo)
    summary_op = tf.summary.merge_all()

    run_training(train_op, summary_op)
