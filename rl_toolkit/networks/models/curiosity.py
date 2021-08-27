from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Curiosity(Model):
    """
    Curiosity
    ===============

    Attributes:
        features_space (int): number of features

    References:
        - [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
    """

    def __init__(self, features_space: int, **kwargs):
        super(Curiosity, self).__init__(**kwargs)

        # Target
        # 1. layer
        self.target_fc1 = Dense(
            400,
            activation="relu",
            kernel_initializer="he_uniform",
            trainable=False,
        )

        # 2. layer
        self.target_fc2 = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
            trainable=False,
        )

        # Output layer
        self.target = Dense(
            features_space,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="target",
            trainable=False,
        )

        # Predicted
        # 1. layer
        self.predicted_fc1 = Dense(
            400,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # 2. layer
        self.predicted_fc2 = Dense(
            300,
            activation="relu",
            kernel_initializer="he_uniform",
        )

        # Output layer
        self.predicted = Dense(
            features_space,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="predicted",
        )

    def call(self, inputs):
        # Target
        # 1. layer
        target = self.target_fc1(inputs)

        # 2. layer
        target = self.target_fc2(target)

        # Output layer
        target = self.target(target)

        # Predicted
        # 1. layer
        predicted = self.predicted_fc1(inputs)

        # 2. layer
        predicted = self.predicted_fc2(predicted)

        # Output layer
        predicted = self.predicted(predicted)

        return [target, predicted]
