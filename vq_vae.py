from logging import Logger

import tensorflow as tf

from sacred.stflow import LogFileWriter

from experiment import ex, create_dataset, run_training, create_writer
from models import make_decoder, make_encoder, VectorQuantizer


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

    # Pass nearest codebook entries to decoder and pass its gradients
    # straigth through to the encoder.
    # => Forward pass: nearest_codebook_entries, Backpropagation: codes
    codes_straight_through = codes + tf.stop_gradient(nearest_codebook_entries - codes)
    decoder_distribution = tf.distributions.Bernoulli(
        logits=decoder(codes_straight_through)
    )

    print(encoder.summary())

    print(decoder.summary())

    with tf.variable_scope("posterior", reuse=True):
        posterior_sample = decoder_distribution.mean()
        tf.summary.image("posterior_sample", tf.cast(posterior_sample, tf.float32))

    reconstruction_loss = -tf.reduce_mean(decoder_distribution.log_prob(x))
    commitment_loss = tf.reduce_mean(
        tf.square(codes - tf.stop_gradient(nearest_codebook_entries))
    )
    embedding_loss = tf.reduce_mean(
        tf.square(tf.stop_gradient(nearest_codebook_entries) - codes)
    )

    # Uniform prior over codes
    prior_distribution = tf.distributions.Multinomial(
        total_count=1.0, logits=tf.zeros([latent_size, num_codes])
    )

    with tf.variable_scope("prior", reuse=True):
        prior_sample_codes = quantizer.get_codebook_entries(
            prior_distribution.sample(1)
        )
        prior_decoder_distribution = tf.distributions.Bernoulli(
            logits=decoder(prior_sample_codes)
        )
        prior_sample = prior_decoder_distribution.mean()
        tf.summary.image("prior_sample", tf.cast(prior_sample, tf.float32))

    loss = reconstruction_loss + embedding_loss + beta * commitment_loss

    tf.summary.scalar("losses/total_loss", loss)
    tf.summary.scalar("losses/embedding_loss", embedding_loss)
    tf.summary.scalar("losses/reconstruction_loss", reconstruction_loss)
    tf.summary.scalar("losses/commitment_loss", beta * commitment_loss)

    train_op = tf.train.AdamOptimizer().minimize(loss)
    summary_op = tf.summary.merge_all()

    run_training(train_op, summary_op, writer)

