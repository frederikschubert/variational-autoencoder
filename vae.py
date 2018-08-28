import time

import tensorflow as tf
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

from encoder import Encoder
from decoder import Decoder

ex = Experiment("variational autoencoder")
ex.observers.append(FileStorageObserver.create("tmp"))


@ex.config
def config():
    log_dir = "./tmp"
    log_interval = 1000
    iterations = 100000
    output_shape = [28, 28, 1]
    z_dimension = 100
    batch_size = 64
    convolutional = False
    beta = 100


@ex.named_config
def cnn():
    convolutional = True


@ex.automain
@LogFileWriter(ex)
def train(
    _log,
    _run,
    log_dir,
    log_interval,
    batch_size,
    iterations,
    z_dimension,
    output_shape,
    convolutional,
    beta
):

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = (x_train > 0.5).astype(np.float32)
    x_train = np.expand_dims(x_train, -1)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(batch_size)

    x = dataset.make_one_shot_iterator().get_next()

    encoder = Encoder(z_dimension, convolutional)
    decoder = Decoder(output_shape, convolutional)

    with tf.name_scope("data"):
        tf.summary.image("mnist_image", x)

    with tf.variable_scope("variational"):
        z_mean, z_log_variance = encoder(x)
        q_z = tf.distributions.Normal(loc=z_mean, scale=tf.exp(0.5 * z_log_variance))
        assert q_z.reparameterization_type == tf.distributions.FULLY_REPARAMETERIZED

    with tf.variable_scope("model", reuse=True):
        p_z = tf.distributions.Normal(
            loc=np.zeros(shape=z_dimension, dtype=np.float32),
            scale=np.ones(shape=z_dimension, dtype=np.float32),
        )
        p_x_given_z_logits = decoder(tf.expand_dims(p_z.sample(), axis=0))
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        prior_sample = p_x_given_z.sample()
        tf.summary.image("prior_sample", tf.cast(prior_sample, tf.float32))

    with tf.variable_scope("model", reuse=True):
        p_x_given_z_logits = decoder(q_z.sample())
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        posterior_sample = p_x_given_z.sample()
        tf.summary.image("posterior_sample", tf.cast(posterior_sample, tf.float32))

    kl_divergence = tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1)
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), axis=[1, 2, 3])

    elbo = tf.reduce_sum(expected_log_likelihood - beta * kl_divergence, axis=0)
    tf.summary.scalar("elbo", elbo / batch_size)

    optimizer = tf.train.AdamOptimizer()

    train_op = optimizer.minimize(-elbo)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir=f"{log_dir}/{_run._id}")

        ts = time.time()
        _log.info("Training started.")
        for i in range(iterations):

            sess.run(train_op)

            if i % log_interval == 0:
                iteration_elbo, iteration_summary = sess.run([elbo, summary_op])
                writer.add_summary(iteration_summary, i)

                _log.info(
                    "Iteration: {0:d} ELBO: {1:.3f} seconds/batch: {2:.3e}".format(
                        i,
                        iteration_elbo / batch_size,
                        (time.time() - ts) / log_interval,
                    )
                )
                ts = time.time()
        _log.info("Training finished.")
