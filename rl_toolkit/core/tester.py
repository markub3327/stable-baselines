import numpy as np
import tensorflow as tf

import wandb
from rl_toolkit.networks.models import Actor

from .process import Process


class Tester(Process):
    """
    Tester
    =================

    Attributes:
        env_name (str): the name of environment
        render (bool): enable the rendering into the video file
        max_steps (int): maximum number of interactions do in environment
        model_path (str): path to the model
    """

    def __init__(
        self,
        # ---
        env_name: str,
        render: bool,
        # ---
        max_steps: int,
        # ---
        model_path: str,
    ):
        super(Tester, self).__init__(env_name)

        self._max_steps = max_steps

        self.actor = Actor(n_outputs=np.prod(self._env.action_space.shape))
        self.actor.build((None,) + self._env.observation_space.shape)

        if model_path is not None:
            self.actor.load_weights(model_path)

        # init Weights & Biases
        wandb.init(
            project="rl-toolkit",
            group=f"{env_name}",
            monitor_gym=render,
        )
        wandb.config.max_steps = max_steps

    @tf.function(jit_compile=True)
    def policy(self, input):
        action, _ = self.actor(
            tf.expand_dims(input, axis=0),
            with_log_prob=False,
            deterministic=True,
        )
        return tf.squeeze(action, axis=0)

    def run(self):
        self._total_steps = 0
        self._total_episodes = 0
        self._episode_reward = 0.0
        self._episode_steps = 0

        # init environment
        self._last_obs = self._env.reset()

        # hlavny cyklus hry
        while self._total_steps < self._max_steps:
            # Get the action
            action = self.policy(self._last_obs)
            action = np.array(action, copy=False, dtype="float32")

            # perform action
            new_obs, reward, terminal, _ = self._env.step(action)

            # update variables
            self._episode_reward += reward
            self._episode_steps += 1
            self._total_steps += 1

            # Check the end of episode
            if terminal:
                # logovanie
                print("=============================================")
                print(f"Epoch: {self._total_episodes}")
                print(f"Score: {self._episode_reward}")
                print(f"Steps: {self._episode_steps}")
                print(f"TotalInteractions: {self._total_steps}")
                print("=============================================")
                print(
                    f"Testing ... {(self._total_steps * 100) / self._max_steps} %"  # noqa
                )
                wandb.log(
                    {
                        "Epoch": self._total_episodes,
                        "Score": self._episode_reward,
                        "Steps": self._episode_steps,
                    },
                    step=self._total_steps,
                )

                # Init variables
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._total_episodes += 1

                # Init environment
                self._last_obs = self._env.reset()
            else:
                # super critical !!!
                self._last_obs = new_obs
