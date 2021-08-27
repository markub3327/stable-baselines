import tensorflow as tf


class RunningStats:
    def __init__(self, shape):
        self.mean = tf.Variable(tf.zeros(shape), trainable=False, name="running_mean")
        self.variance = tf.Variable(tf.ones(shape), trainable=False, name="running_variance")
        self.momentum = 0.999

    def update(self, batch):
        self.mean.assign(self.mean * self.momentum + tf.math.reduce_mean(batch, axis=0) * (1 - self.momentum))
        self.variance.assign(self.variance * self.momentum + tf.math.reduce_variance(batch, axis=0) * (1 - self.momentum))
        tf.print(f"mean: {self.mean}, variance: {self.variance}")

    def normalize(self, values):
        return (values - self.mean) / tf.sqrt(self.variance + 1e-6)