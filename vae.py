import tensorflow as tf
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from encoder import Encoder
from decoder import Decoder

ex = Experiment("variational autoencoder")
ex.observers.append(FileStorageObserver.create("tmp"))


@ex.config
def mnist():
    log_dir = "./tmp"
    log_interval = 1000
    iterations = 100000
    output_shape = [28, 28, 1]
    z_dimension = 2
    batch_size = 64


@ex.automain
def train(
    _log, log_dir, log_interval, z_dimension, output_shape, batch_size, iterations
):

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = (x_train < 0.5).astype(np.float32)
    x_train = np.expand_dims(x_train, -1)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(batch_size)

    x = dataset.make_one_shot_iterator().get_next()

    encoder = Encoder(z_dimension)
    decoder = Decoder(output_shape)

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
        p_x_given_z_logits = decoder(p_z.sample(1))
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        prior_sample = p_x_given_z.sample()
        tf.summary.image("prior_sample", tf.cast(prior_sample, tf.float32))

    with tf.variable_scope("model", reuse=True):
        p_x_given_z_logits = decoder(q_z.sample())
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        posterior_sample = p_x_given_z.sample()
        tf.summary.image("posterior_sample", tf.cast(posterior_sample, tf.float32))

    kl_divergence = tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z))
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), [1, 2, 3])

    elbo = tf.reduce_sum(expected_log_likelihood - kl_divergence, 0)

    optimizer = tf.train.AdamOptimizer()

    train_op = optimizer.minimize(-elbo)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        for i in range(iterations):

            sess.run(train_op)

            if i % log_interval == 0:
                iteration_elbo, iteration_summary = sess.run([elbo, summary_op])
                writer.add_summary(iteration_summary, i)

                _log.info(f"Iteration {i}\tELBO {iteration_elbo}")
