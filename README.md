# RL toolkit

[![Release](https://img.shields.io/github/release/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/releases)
![Tag](https://img.shields.io/github/v/tag/markub3327/rl-toolkit)

[![Issues](https://img.shields.io/github/issues/markub3327/rl-toolkit)](https://github.com/markub3327/rl-toolkit/issues)
![Commits](https://img.shields.io/github/commit-activity/w/markub3327/rl-toolkit)

![Languages](https://img.shields.io/github/languages/count/markub3327/rl-toolkit)
![Size](https://img.shields.io/github/repo-size/markub3327/rl-toolkit)

## Papers

  * **Soft Actor-Critic** (https://arxiv.org/pdf/1812.05905.pdf)
  * **Twin Delayed DDPG** (https://arxiv.org/pdf/1802.09477.pdf)
  * **Generalized State-Dependent Exploration** (https://arxiv.org/pdf/2005.05719.pdf)

## Setting up container

##### YOU MUST HAVE INSTALLED DOCKER !!!

### Build the Docker image

```shell
docker build -t markub/rl-toolkit:cpu .
```

### Run the container

```shell
docker run -it --rm -v $PWD:/root/rl-toolkit markub/rl-toolkit:cpu python3 training.py / testing.py
```

## Using

```shell
# Run training
python3 training.py [-h] -alg sac -env ENV_NAME -s PATH_TO_MODEL_FOLDER [--wandb]

# Run testing
python3 testing.py [-h] -alg td3 -env ENV_NAME -f PATH_TO_MODEL_FOLDER [--wandb]
```

## Tested environments
  
  * MountainCarContinuous-v0
  * BipedalWalker-v3
  * BipedalWalkerHardcore-v3
  * LunarLanderContinuous-v2
  * Walker2DBulletEnv-v0
  * AntBulletEnv-v0
  * HalfCheetahBulletEnv-v0

<p align="center"><b>Summary</b></p>
<p align="center">
  <img src="img/results.png" alt="results">
</p>
<p align="center"><a href="https://wandb.ai/markub/rl-toolkit?workspace=user-markub" target="_blank">For more charts click here.</a></p>

----------------------------------

**Framework:** Tensorflow 2.4
<br>
**Languages:** Python 3.8
<br>
**Author**: Martin Kubovcik
