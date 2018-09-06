import tensorflow as tf


class VectorQuantizer:
    def __init__(self, num_codes, z_dimension):
        self.num_codes = num_codes
        self.z_dimension = z_dimension
        self.codebook = tf.get_variable(
            "codebook",
            shape=[num_codes, z_dimension],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.uniform_unit_scaling_initializer(),
        )

    def get_codebook_entries(self, one_hot_assignments):
        return tf.reduce_sum(
            tf.expand_dims(
                one_hot_assignments, -1
            )  # [batch_size, latent_size, num_codes, 1]
            * tf.reshape(self.codebook, [1, 1, self.num_codes, self.z_dimension]),
            axis=2,
        )

    def __call__(self, codes):
        distances = tf.norm(
            tf.expand_dims(codes, 2)  # [batch_size, latent_size, 1, z_dimension]
            - tf.reshape(self.codebook, [1, 1, self.num_codes, self.z_dimension]),
            axis=-1,
        )
        code_assignments = tf.argmin(distances, -1)
        one_hot_code_assignments = tf.one_hot(code_assignments, depth=self.num_codes)
        nearest_codebook_entries = self.get_codebook_entries(one_hot_code_assignments)
        return nearest_codebook_entries, one_hot_code_assignments
