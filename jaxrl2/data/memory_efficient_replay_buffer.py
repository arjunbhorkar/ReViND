import collections
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict
from gym.spaces import Box

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.data.replay_buffer import ReplayBuffer


class MemoryEfficientReplayBuffer(ReplayBuffer):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 capacity: int):

        self._num_stack = observation_space.shape[-1]
        self._unstacked_dim_size = observation_space.shape[-2]
        low = observation_space.low[..., 0]
        high = observation_space.high[..., 0]
        observation_space = Box(low=low,
                                high=high,
                                dtype=observation_space.dtype)

        self._first = True
        self._correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(observation_space,
                         action_space,
                         capacity,
                         add_next_observations=False)

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(
                self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._correct_index[self._insert_index] = False
                super().insert(element)

        obs = data_dict.pop('observations')
        next_obs = data_dict.pop('next_observations')

        if self._first:
            for i in range(self._num_stack):
                data_dict['observations'] = obs[..., i]
                self._correct_index[self._insert_index] = False
                super().insert(data_dict)

        data_dict['observations'] = next_obs[..., -1]

        self._first = data_dict['dones']

        self._correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._correct_index[indx] = False

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:

        if indx is None:
            indx = np.empty(batch_size, dtype=int)
            for i in range(batch_size):
                while True:
                    if hasattr(self.np_random, 'integers'):
                        indx[i] = self.np_random.integers(
                            self._num_stack, len(self))
                    else:
                        indx[i] = self.np_random.randint(
                            self._num_stack, len(self))
                    if self._correct_index[indx[i]]:
                        break
        else:
            raise ValueError()

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        keys = list(keys)
        keys.remove('observations')

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs = self.dataset_dict['observations']
        obs = np.lib.stride_tricks.sliding_window_view(obs,
                                                       self._num_stack + 1,
                                                       axis=0)
        obs = obs[indx - self._num_stack]
        batch['observations'] = obs

        return frozen_dict.freeze(batch)

    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
