import time
from typing import Tuple
from logging import Logger
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
    data_shape = [28, 28, 1]
    z_dimension = 100
    batch_size = 64
    mode = "default"
    beta = 1.0


@ex.named_config
def conditional():
    mode = "conditional"


@ex.named_config
def cnn():
    mode = "convolutional"


@ex.capture
def create_dataset(
    _log: Logger = None, mode=None
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    _log.info("Creating dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = (
        (x_train > 0.5).astype(np.float32),
        (x_test > 0.5).astype(np.float32),
    )
    x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
    if mode == "conditional":
        return (
            tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat(),
            tf.data.Dataset.from_tensor_slices((x_test, y_test)).repeat(),
        )
    else:
        return (
            tf.data.Dataset.from_tensor_slices(x_train).repeat(),
            tf.data.Dataset.from_tensor_slices(x_test).repeat(),
        )


@ex.capture
def create_writer(log_dir=None, _run=None):
    return tf.summary.FileWriter(logdir=f"{log_dir}/{_run._id}")


@ex.capture
def run_training(
    train_op,
    summary_op,
    writer,
    iterations=None,
    log_interval=None,
    _log: Logger = None,
):
    _log.info("Training started.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, writer.get_logdir(), 1)

        ts = time.time()
        for i in range(iterations):
            sess.run(train_op)
            if i % log_interval == 0:
                _log.info(
                    f"iteration: {i} - seconds/batch: {round((time.time() - ts) / log_interval, 3)}"
                )
                ts = time.time()
                iteration_summary = sess.run(summary_op)
                writer.add_summary(iteration_summary, i)
                writer.flush()
        _log.info("Training finished.")
