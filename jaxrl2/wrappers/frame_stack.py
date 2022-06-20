import collections

import gym
import numpy as np
from gym.spaces import Box


class FrameStack(gym.Wrapper):

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        self._env_dim = self.observation_space.shape[-1]

        low = np.repeat(self.observation_space.low[..., np.newaxis],
                        num_stack,
                        axis=-1)
        high = np.repeat(self.observation_space.high[..., np.newaxis],
                         num_stack,
                         axis=-1)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append(obs)
        return self.frames

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self.frames, reward, done, info
