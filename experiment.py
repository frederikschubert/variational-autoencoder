import tensorflow as tf
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("variational autoencoder")
ex.observers.append(FileStorageObserver.create("tmp"))


@ex.config
def config():
    log_dir = "./tmp"
    log_interval = 1000
    iterations = 50000
    output_shape = [28, 28, 1]
    z_dimension = 100
    batch_size = 64
    mode = "default"
    # See "Understanding disentangling in β-VAE" @ https://arxiv.org/abs/1804.03599
    beta = 1.15


@ex.named_config
def cnn():
    mode = "convolutional"


@ex.capture
def create_dataset(batch_size=64, _log=None):
    _log.info("Creating dataset...")
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = (x_train > 0.5).astype(np.float32)
    x_train = np.expand_dims(x_train, -1)
    return tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(batch_size)


@ex.capture
def create_writer(log_dir="./tmp", _run=None):
    return tf.summary.FileWriter(logdir=f"{log_dir}/{_run._id}")


@ex.capture
def run_training(train_op, summary_op, iterations=50000, log_interval=1000, _log=None):
    _log.info("Training started.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = create_writer()

        for i in range(iterations):
            sess.run(train_op)
            if i % log_interval == 0:
                iteration_summary = sess.run([summary_op])
                writer.add_summary(iteration_summary[0], i)
        _log.info("Training finished.")
