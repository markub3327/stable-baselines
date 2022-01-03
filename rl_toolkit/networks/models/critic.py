import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Activation, Add, Dense, LayerNormalization

uniform_initializer = VarianceScaling(
    distribution="uniform", mode="fan_out", scale=(1.0 / 3.0)
)


class Critic(Model):
    """
    Critic
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        n_quantiles (int): number of predicted quantiles

    References:
        - [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269)
    """

    def __init__(self, units: list, n_quantiles: int, **kwargs):
        super(Critic, self).__init__(**kwargs)

        # 1. layer
        self.fc_0 = Dense(
            units=units[0],
            kernel_initializer=uniform_initializer,
        )
        self.norm_0 = LayerNormalization()
        self.activ_0 = Activation("tanh")

        self.fc_1 = Dense(
            units=units[0],
            kernel_initializer=uniform_initializer,
        )
        self.norm_1 = LayerNormalization()
        self.activ_1 = Activation("tanh")

        # 2. layer
        self.fc_2 = Dense(
            units=units[1],
            kernel_initializer=uniform_initializer,
        )
        self.fc_3 = Dense(
            units=units[1],
            kernel_initializer=uniform_initializer,
        )
        self.add_0 = Add()
        self.activ_0 = Activation("relu")

        # Output layer
        self.quantiles = Dense(
            n_quantiles,
            activation="linear",
            kernel_initializer=uniform_initializer,
            name="quantiles",
        )

    def call(self, inputs):
        # 1. layer
        state = self.fc_0(inputs[0])
        state = self.norm_0(state)
        state = self.activ_0(state)

        action = self.fc_1(inputs[1])
        action = self.norm_1(action)
        action = self.activ_1(action)

        # 2. layer
        state = self.fc_2(state)
        action = self.fc_3(action)
        x = self.add_0([state, action])
        x = self.activ_0(x)

        # Output layer
        quantiles = self.quantiles(x)
        return quantiles


class MultiCritic(Model):
    """
    MultiCritic
    ===============

    Attributes:
        units (list): list of the numbers of units in each layer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
    """

    def __init__(
        self,
        units: list,
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        **kwargs
    ):
        super(MultiCritic, self).__init__(**kwargs)

        self.n_quantiles = n_quantiles
        self.quantiles_total = n_quantiles * n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # init critics
        self.models = []
        for _ in range(n_critics):
            self.models.append(Critic(units, n_quantiles))

    def call(self, inputs):
        quantiles = tf.stack(list(model(inputs) for model in self.models), axis=1)
        return quantiles

    def summary(self):
        for model in self.models:
            model.summary()
