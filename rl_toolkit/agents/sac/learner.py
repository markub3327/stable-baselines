import os

import numpy as np
import reverb
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbMetricsLogger

import wandb
from rl_toolkit.networks.callbacks import SACAgentCallback
from rl_toolkit.networks.models import ActorCritic
from rl_toolkit.utils import make_reverb_dataset

from ...core.process import Process


class Learner(Process):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        db_server (str): database server name (IP or domain name)
        train_steps (int): number of training steps
        batch_size (int): size of mini-batch used for training
        actor_units (list): list of the numbers of units in each Actor's layer
        critic_units (list): list of the numbers of units in each Critic's layer
        actor_learning_rate (float): the learning rate for the Actor's optimizer
        critic_learning_rate (float): the learning rate for the Critic's optimizer
        alpha_learning_rate (float): the learning rate for the Alpha's optimizer
        n_quantiles (int): number of predicted quantiles
        top_quantiles_to_drop (int): number of quantiles to drop
        n_critics (int): number of critic networks
        clip_mean_min (float): the minimum value of mean
        clip_mean_max (float): the maximum value of mean
        gamma (float): the discount factor
        tau (float): the soft update coefficient for target networks
        init_alpha (float): initialization of alpha param
        init_noise (float): initialization of the Actor's noise
        save_path (str): path to the models for saving
    """

    def __init__(
        self,
        # ---
        env_name: str,
        db_server: str,
        # ---
        train_steps: int,
        batch_size: int,
        # ---
        actor_units: list,
        critic_units: list,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        # ---
        n_quantiles: int,
        top_quantiles_to_drop: int,
        n_critics: int,
        # ---
        clip_mean_min: float,
        clip_mean_max: float,
        # ---
        actor_global_clipnorm: float,
        critic_global_clipnorm: float,
        # ---
        gamma: float,
        tau: float,
        init_alpha: float,
        init_noise: float,
        merge_index: int,
        frame_stack: int,
        # ---
        save_path: str,
    ):
        super(Learner, self).__init__(env_name, False, frame_stack)

        tf.config.optimizer.set_jit(True)  # Enable XLA.

        self._train_steps = train_steps
        self._save_path = save_path
        self._db_server = db_server

        # Init actor-critic's network
        self.model = ActorCritic(
            actor_units=actor_units,
            critic_units=critic_units,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop=top_quantiles_to_drop,
            n_critics=n_critics,
            n_outputs=np.prod(self._env.action_space.shape),
            clip_mean_min=clip_mean_min,
            clip_mean_max=clip_mean_max,
            gamma=gamma,
            tau=tau,
            init_alpha=init_alpha,
            init_noise=init_noise,
            merge_index=merge_index,
        )
        self.model.build((None,) + self._env.observation_space.shape)
        self.model.compile(
            actor_optimizer=Adam(
                learning_rate=actor_learning_rate,
                global_clipnorm=actor_global_clipnorm,
            ),
            critic_optimizer=Adam(
                learning_rate=critic_learning_rate,
                global_clipnorm=critic_global_clipnorm,
            ),
            alpha_optimizer=Adam(learning_rate=alpha_learning_rate),
        )

        # Show models details
        self.model.summary()

        # Initializes the reverb's dataset
        self.dataset = make_reverb_dataset(
            server_address=self._db_server,
            table="experience",
            batch_size=batch_size,
        )

        # init Weights & Biases
        wandb.init(project="rl-toolkit", group=f"{env_name}")
        wandb.config.train_steps = train_steps
        wandb.config.batch_size = batch_size
        wandb.config.actor_units = actor_units
        wandb.config.critic_units = critic_units
        wandb.config.actor_learning_rate = actor_learning_rate
        wandb.config.critic_learning_rate = critic_learning_rate
        wandb.config.alpha_learning_rate = alpha_learning_rate
        wandb.config.actor_global_clipnorm = actor_global_clipnorm
        wandb.config.critic_global_clipnorm = critic_global_clipnorm
        wandb.config.n_quantiles = n_quantiles
        wandb.config.top_quantiles_to_drop = top_quantiles_to_drop
        wandb.config.n_critics = n_critics
        wandb.config.clip_mean_min = clip_mean_min
        wandb.config.clip_mean_max = clip_mean_max
        wandb.config.gamma = gamma
        wandb.config.tau = tau
        wandb.config.init_alpha = init_alpha
        wandb.config.init_noise = init_noise

    def run(self):
        self.model.fit(
            self.dataset,
            epochs=self._train_steps,
            steps_per_epoch=1,
            verbose=0,
            callbacks=[
                SACAgentCallback(self._db_server),
                WandbMetricsLogger(log_freq=10),
            ],
        )

    def save(self):
        if self._save_path:
            os.makedirs(self._save_path, exist_ok=True)
            # Save model
            self.model.save_weights(os.path.join(self._save_path, "actor_critic.h5"))

    def close(self):
        super(Learner, self).close()

        # create the checkpoint of the database
        client = reverb.Client(self._db_server)
        client.checkpoint()
