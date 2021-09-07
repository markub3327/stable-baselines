import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense


class Curiosity(Model):
    """
    Curiosity
    ===============

    Attributes:
        state_shape: the shape of next state

    References:
        - [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
    """

    def __init__(self, state_shape, **kwargs):
        super(Curiosity, self).__init__(**kwargs)

        # Input layer
        self.merged = Concatenate()

        # 1. layer
        self.fc1 = Dense(
            400,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # 2. layer
        self.fc2 = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # Output layer
        self.next_state = Dense(
            tf.reduce_prod(state_shape),
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="next_state",
        )

    def call(self, inputs):
        x = self.merged(inputs)

        # 1. layer
        x = self.fc1(x)

        # 2. layer
        x = self.fc2(x)

        # Output layer
        next_state = self.next_state(x)
        return next_state

    def get_reward(self, data):
        # Euclidean distance
        return tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    data["next_observation"]
                    - self([data["observation"], data["action"]])
                ),
                axis=-1,
                keepdims=True,
            )
        )

    def train_step(self, data):
        with tf.GradientTape() as tape:
            next_state = self([data["observation"], data["action"]])

            curiosity_loss = tf.nn.compute_average_loss(
                tf.keras.losses.huber(
                    y_true=data["next_observation"], y_pred=next_state
                )
            )

        gradients = tape.gradient(curiosity_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"curiosity_loss": curiosity_loss}

    def compile(self, optimizer):
        super(Curiosity, self).compile()
        self.optimizer = optimizer
