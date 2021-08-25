import reverb
import tensorflow as tf

from .policy import Policy


class Server(Policy):
    """
    Learner
    =================

    Attributes:
        env_name (str): the name of environment
        min_replay_size (int): minimum number of samples in memory before learning starts
        samples_per_insert (int): samples per insert ratio (SPI) `= num_sampled_items / num_inserted_items`
        buffer_capacity (int): the capacity of experiences replay buffer
        db_path (str): path to the database checkpoint
    """

    def __init__(
        self,
        # ---
        env_name: str,
        # ---
        min_replay_size: int,
        samples_per_insert: int,
        buffer_capacity: int,
        # ---
        db_path: str,
    ):
        super(Server, self).__init__(env_name)

        # Load DB from checkpoint or make a new one
        if db_path is None:
            checkpointer = None
        else:
            checkpointer = reverb.checkpointers.DefaultCheckpointer(path=db_path)

        if samples_per_insert is None:
            limiter = reverb.rate_limiters.MinSize(min_replay_size)
        else:
            # 10% tolerance in rate
            samples_per_insert_tolerance = 0.1 * samples_per_insert
            error_buffer = min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=min_replay_size,
                samples_per_insert=samples_per_insert,
                error_buffer=error_buffer,
            )

        # Initialize the reverb server
        self.server = reverb.Server(
            tables=[
                reverb.Table(  # Replay buffer
                    name="experience",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=limiter,
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
                        "next_action": tf.TensorSpec(
                            [*self._env.action_space.shape],
                            self._env.action_space.dtype,
                        ),
                        "terminal": tf.TensorSpec([1], tf.bool),
                    },
                ),
                reverb.Table(  # Variables container
                    name="variables",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    max_size=1,
                    max_times_sampled=0,
                    signature=self._variable_container.signature,
                ),
            ],
            port=8000,
            checkpointer=checkpointer,
        )

    def run(self):
        self.server.wait()

    def close(self):
        super(Server, self).close()

        # create the checkpoint of DB
        client = reverb.Client("localhost:8000")
        client.checkpoint()
