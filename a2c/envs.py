import os

import gym
from gym.spaces.box import Box
from logo.mnist_turtle_env import MnistTurtleEnv
from logo.connect_dots_env import ConnectDotsEnv
from baselines import bench

def make_env(env_name, seed=0, rank=0, digit=1, log_dir=None, use_patience=False):
    def _thunk():
        if env_name == 'mnist_turle':
            env = MnistTurtleEnv(digit=digit)
        elif env_name == 'connect_dots':
            env = ConnectDotsEnv(digit=digit, rank=rank, use_patience=use_patience)
        else:
            raise NotImplementedError("env not implemented")
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)
        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)