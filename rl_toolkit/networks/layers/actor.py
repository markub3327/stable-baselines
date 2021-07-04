import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Dense

from rl_toolkit.networks.activations import clipped_linear

from .noise import MultivariateGaussianNoise


class Actor(Model):
    """
    Actor
    ===============

    Attributes:
        n_outputs (int): number of outputs

    References:
        - [CrossNorm: On Normalization for Off-Policy TD Reinforcement Learning](https://arxiv.org/abs/1902.05605)
    """

    def __init__(self, n_outputs: int, **kwargs):
        super(Actor, self).__init__(**kwargs)

        # 1. layer
        self.fc1 = Dense(400, kernel_initializer="he_uniform")
        self.fc1_activ = Activation("relu")

        # 2. layer
        self.latent_sde = Dense(
            300,
            kernel_initializer="he_uniform",
        )
        self.latent_sde_activ = Activation("relu")

        # Deterministicke akcie
        self.mean = Dense(
            n_outputs,
            activation=clipped_linear,
            kernel_initializer="glorot_uniform",
            name="mean",
        )

        # Stochasticke akcie
        self.noise = MultivariateGaussianNoise(
            n_outputs,
            kernel_initializer=tf.keras.initializers.Constant(value=-3.0),
            name="noise",
        )

        # Vystupna prenosova funkcia
        self.bijector = tfp.bijectors.Tanh()

    def reset_noise(self):
        self.noise.sample_weights()

    def call(self, inputs, with_log_prob=True, deterministic=None):
        # 1. layer
        x = self.fc1(inputs)
        x = self.fc1_activ(x)

        # 2. layer
        latent_sde = self.latent_sde(x)
        latent_sde = self.latent_sde_activ(latent_sde)

        # Output layer
        mean = self.mean(latent_sde)

        if deterministic:
            action = self.bijector.forward(mean)
            log_prob = None
        else:
            noise = self.noise(latent_sde)
            action = self.bijector.forward(mean + noise)

            if with_log_prob:
                variance = tf.matmul(
                    tf.square(latent_sde), tf.square(self.noise.get_std())
                )
                pi_distribution = tfp.distributions.TransformedDistribution(
                    distribution=tfp.distributions.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.sqrt(variance + 1e-6)
                    ),
                    bijector=self.bijector,
                )
                log_prob = pi_distribution.log_prob(action)[..., tf.newaxis]
            else:
                log_prob = None

        return [action, log_prob]
