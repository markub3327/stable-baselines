from rl_toolkit.networks import Actor, TwinCritic
from rl_toolkit.policy import Policy
from rl_toolkit.utils import VariableContainer

import os
import reverb
import wandb

import tensorflow as tf


class Learner(Policy):
    """
    Learner (based on Soft Actor-Critic)
    =================

    Attributes:
        env: the instance of environment object
        max_steps (int): maximum number of interactions do in environment
        warmup_steps (int): number of interactions before using policy network
        buffer_capacity (int): the capacity of experiences replay buffer
        batch_size (int): size of mini-batch used for training
        actor_learning_rate (float): learning rate for actor's optimizer
        critic_learning_rate (float): learning rate for critic's optimizer
        alpha_learning_rate (float): learning rate for alpha's optimizer
        tau (float): the soft update coefficient for target networks
        gamma (float): the discount factor
        actor_path (str): path to the actor's model
        critic_path (str): path to the critic's model
        db_path (str): path to the database checkpoint
        save_path (str): path to the models for saving
        log_wandb (bool): log into WanDB cloud

    Paper: https://arxiv.org/pdf/1812.05905.pdf
    """

    def __init__(
        self,
        # ---
        env,
        max_steps: int,
        warmup_steps: int = 10000,
        # ---
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        # ---
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # ---
        tau: float = 0.01,
        gamma: float = 0.99,
        # ---
        actor_path: str = None,
        critic_path: str = None,
        db_path: str = None,
        save_path: str = None,
        # ---
        log_wandb: bool = False,
        log_interval: int = 64,
    ):
        super(Learner, self).__init__(env, log_wandb)

        self._max_steps = max_steps
        self._warmup_steps = warmup_steps
        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._save_path = save_path
        self._log_interval = log_interval

        # init param 'alpha' - Lagrangian constraint
        self._log_alpha = tf.Variable(0.0, trainable=True, name="log_alpha")
        self._alpha = tf.Variable(0.0, trainable=False, name="alpha")
        self._alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate, name="alpha_optimizer"
        )
        self._target_entropy = tf.cast(
            -tf.reduce_prod(self._env.action_space.shape), dtype=tf.float32
        )

        # Actor network (for learner)
        self._actor = Actor(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=actor_learning_rate,
            model_path=actor_path,
        )
        self._container = VariableContainer("localhost", self._actor)

        # Critic network & target network
        self._critic = TwinCritic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            learning_rate=critic_learning_rate,
            model_path=critic_path,
        )
        self._critic_target = TwinCritic(
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
        )
        self._update_target(self._critic, self._critic_target, tau=tf.constant(1.0))

        # load db from checkpoint or make a new one
        if db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=db_path)

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experience",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(warmup_steps),
                    max_size=buffer_capacity,
                    max_times_sampled=0,
                    signature={
                        "observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "action": tf.TensorSpec(
                            [*self._env.action_space.shape],
                            self._env.action_space.dtype,
                        ),
                        "reward": tf.TensorSpec([1], tf.float32),
                        "next_observation": tf.TensorSpec(
                            [*self._env.observation_space.shape],
                            self._env.observation_space.dtype,
                        ),
                        "terminal": tf.TensorSpec([1], tf.float32),
                    },
                ),
                reverb.Table(  # Variable container
                    name="variables",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature=self._container.variable_container_signature,
                ),
            ],
            port=8000,
            checkpointer=checkpointer,
        )

        # Initializes the reverb client and tf.dataset
        self.client = reverb.Client("localhost:8000")
        self.dataset_iterator = iter(
            reverb.TrajectoryDataset.from_table_signature(
                server_address="localhost:8000",
                table="experience",
                max_in_flight_samples_per_worker=10,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # init Weights & Biases
        if self._log_wandb:
            wandb.init(project="rl-toolkit")

            # Settings
            wandb.config.max_steps = max_steps
            wandb.config.warmup_steps = warmup_steps
            wandb.config.buffer_capacity = buffer_capacity
            wandb.config.batch_size = batch_size
            wandb.config.actor_learning_rate = actor_learning_rate
            wandb.config.critic_learning_rate = critic_learning_rate
            wandb.config.alpha_learning_rate = alpha_learning_rate
            wandb.config.tau = tau
            wandb.config.gamma = gamma

        # init actor's params in DB
        self._container.push_variables()

    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(
            net.model.trainable_variables, net_targ.model.trainable_variables
        ):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    # -------------------------------- update critic ------------------------------- #
    def _update_critic(self, batch):
        next_action, next_log_pi = self._actor.predict(batch.data["next_observation"])

        # target Q-values
        next_q_value = self._critic_target.model(
            [batch.data["next_observation"], next_action]
        )

        # Bellman Equation
        Q_target = tf.stop_gradient(
            batch.data["reward"]
            + (1 - batch.data["terminal"])
            * self._gamma
            * (next_q_value - self._alpha * next_log_pi)
        )

        # update critic
        with tf.GradientTape() as tape:
            q_value = self._critic.model(
                [batch.data["observation"], batch.data["action"]]
            )
            q_losses = tf.losses.huber(  # less sensitive to outliers in batch
                y_true=Q_target, y_pred=q_value
            )
            q_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q_loss, self._critic.model.trainable_variables)
        self._critic.optimizer.apply_gradients(
            zip(grads, self._critic.model.trainable_variables)
        )

        return q_loss

    # -------------------------------- update actor ------------------------------- #
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_pi = self._actor.predict(batch.data["observation"])

            # predict q value
            q_value = self._critic.model([batch.data["observation"], y_pred])

            policy_losses = self._alpha * log_pi - q_value
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        grads = tape.gradient(policy_loss, self._actor.model.trainable_variables)
        self._actor.optimizer.apply_gradients(
            zip(grads, self._actor.model.trainable_variables)
        )

        return policy_loss

    # -------------------------------- update alpha ------------------------------- #
    def _update_alpha(self, batch):
        _, log_pi = self._actor.predict(batch.data["observation"])

        self._alpha.assign(tf.exp(self._log_alpha))
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._log_alpha * tf.stop_gradient(log_pi + self._target_entropy)
            )
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    @tf.function
    def _update(self):
        # Get data from replay buffer
        sample = self.dataset_iterator.get_next()

        # Re-new noise matrix every update of 'log_std' params
        self._actor.reset_noise()

        # Alpha param update
        alpha_loss = self._update_alpha(sample)

        # Critic models update
        critic_loss = self._update_critic(sample)

        # Actor model update
        policy_loss = self._update_actor(sample)

        # Soft update target networks
        self._update_target(self._critic, self._critic_target, tau=self._tau)

        # Store new actor's params
        self._container.push_variables()

        return critic_loss, policy_loss, alpha_loss

    def run(self):
        for train_step in range(self._warmup_steps, self._max_steps):
            # update train_step (otlacok modelov)
            self._container.train_step.assign(train_step)

            # update models
            critic_loss, policy_loss, alpha_loss = self._update()

            # log metrics
            if (train_step % self._log_interval) == 0:
                print("=============================================")
                print(f"Step: {train_step}")
                print(f"Critic loss: {critic_loss}")
                print(f"Policy loss: {policy_loss}")
                print("=============================================")
                print(
                    f"Training ... {tf.floor(train_step * 100 / self._max_steps)} %"  # noqa
                )

            if self._log_wandb:
                # log of epoch's mean loss
                wandb.log(
                    {
                        "policy_loss": policy_loss,
                        "critic_loss": critic_loss,
                        "alpha_loss": alpha_loss,
                        "alpha": self._alpha,
                    },
                    step=train_step,
                )

        # Stop the agents
        self._container.stop_agents.assign(True)
        self._container.push_variables()

    def save(self):
        if self._save_path is not None:
            # store models
            self._actor.model.save(os.path.join(self._save_path, "actor"))
            self._critic.model.save(os.path.join(self._save_path, "critic"))

        # store checkpoint of DB
        self.client.checkpoint()

    def convert(self):
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(self._actor.model)
        tflite_model = converter.convert()

        # Save the model.
        with open("model_A.tflite", "wb") as f:
            f.write(tflite_model)
