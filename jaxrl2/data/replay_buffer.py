from typing import Iterable, Optional, Union

import flax.core.frozen_dict as frozen_dict
import gym
import gym.spaces
import numpy as np

from jaxrl2.data.dataset import Dataset, DatasetDict, _sample


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 add_next_observations: bool = True):
        observation_data = _init_replay_dict(observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
            trajectory_start_idx=np.empty((capacity, ), dtype=np.int32),
            trajectory_length=np.empty(
                (capacity, ), dtype=np.int32),  #Not guaranteed to be modulo!
        )

        if add_next_observations:
            next_observation_data = _init_replay_dict(observation_space,
                                                      capacity)
            dataset_dict['next_observations'] = next_observation_data

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self.current_trajectory_start_idx = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        data_dict.update(
            {'trajectory_start_idx': self.current_trajectory_start_idx})
        if (self._insert_index
            ) % self._capacity == self.current_trajectory_start_idx:
            data_dict.update({'trajectory_length': 0})
        else:
            data_dict.update({'trajectory_length': -1})  #Shouldn't be accessed

        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        assert self.dataset_dict['trajectory_length'][
            self.
            current_trajectory_start_idx] >= 0, "Problem with Replay Trajectory IDXs"
        self.dataset_dict['trajectory_length'][
            self.current_trajectory_start_idx] += 1

        if data_dict['dones']:
            self.current_trajectory_start_idx = (
                self._insert_index) % self._capacity


class HERReplayBuffer(ReplayBuffer):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 p_relabel: float,
                 reward_relabel_function,
                 p_random: Optional[float] = np.NaN,
                 p_geom: Optional[float] = np.NaN,
                 mask_on_reward: Optional[bool] = False,
                 add_next_observations: bool = True):

        #goal_data = _init_replay_dict(goal_space, capacity)

        super().__init__(observation_space,
                         action_space,
                         capacity,
                         add_next_observations=add_next_observations)

        self.p_relabel = p_relabel
        self.p_random = p_random
        self.p_geom = p_geom
        self.mask_on_reward = mask_on_reward
        self.reward_relabel_function = reward_relabel_function

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               her: Optional[bool] = True) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        if her:
            start_idxs = self.dataset_dict['trajectory_start_idx'][
                indx]  #Start of Trajectory for IDXs of Replay Buffer
            #temporal_distance_from_start = (indx - start_idxs) % len(self) #How many steps into the trajectory am I?
            traj_lengths = self.dataset_dict['trajectory_length'][
                start_idxs]  #Trajectory Length for IDXs of Replay
            end_idxs = (start_idxs + traj_lengths
                        )  #Start_idx INCLUDES first trajectory point

            #TODO: Fix Big when trajectory wraps over replay buffer

            assert all(traj_lengths >= 0), "Problem with trajectory lengths"
            #assert all(self.dataset_dict['dones'][end_idxs]), "Max Idx should be a termination transition"
            # future_relabel = np.random.geometric(p = self.p_geom, size = len(indx))
            # goal_relabels = np.minimum(indx + future_relabel, end_idxs) % len(self)
            if self.p_geom is not np.NaN and self.p_geom > 0:
                #Sample Geometrically
                future_relabel = np.random.geometric(p=self.p_geom,
                                                     size=len(indx))
                goal_relabels = (indx + future_relabel) % len(self)
                goal_relabels = np.minimum(goal_relabels,
                                           (end_idxs - 1) % len(self))
            else:
                #Sample Future Goals Uniformly
                trajectory_to_go = (end_idxs - indx) % len(self)
                future_relabel = np.random.randint(0, trajectory_to_go)
                goal_relabels = (indx + future_relabel) % len(self)

            if self.p_random is not np.NaN and self.p_random > 0:
                number_random_relabels = int(
                    len(indx) * self.p_relabel *
                    self.p_random)  #Not len(indx) * self.p_random
                random_relabels = np.random.randint(
                    np.zeros(indx.shape),
                    np.ones(indx.shape) * len(self))[:number_random_relabels]
                goal_relabels[:number_random_relabels] = random_relabels

            original_achieved_goal = self.dataset_dict['next_observations'][
                'achieved_goal'][indx]  #Current Achieved Goal
            relabeled_desired_goal = self.dataset_dict['next_observations'][
                'achieved_goal'][goal_relabels]  #Goal Relabels
            original_achieved_state = self.dataset_dict['next_observations'][
                'state_drop'][indx]  #Current Achieved Goal
            relabeled_desired_state = self.dataset_dict['next_observations'][
                'state_drop'][goal_relabels]  #Goal Relabels

            reward_relabel = self.reward_relabel_function(
                original_achieved_goal, relabeled_desired_goal, original_achieved_state, relabeled_desired_state)

            if self.mask_on_reward:
                mask_relabel = np.float32(reward_relabel == -1)
                done_relabel = (reward_relabel == 0)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        number_relabels = int(len(indx) * self.p_relabel)
        for k in keys:
            if 'observations' in k and her:  #Apply Goal-Relabeling to Observations and Next_Observations
                formal_batch = _sample(self.dataset_dict[k], indx)
                formal_batch[
                    'desired_goal'][:
                                    number_relabels] = relabeled_desired_goal[:
                                                                              number_relabels]
                batch[k] = formal_batch
            elif k == 'rewards' and her:
                formal_batch = self.dataset_dict[k][indx]
                formal_batch[:
                             number_relabels] = reward_relabel[:
                                                               number_relabels]
                batch[k] = formal_batch
            elif k == 'masks' and self.mask_on_reward and her:
                formal_batch = self.dataset_dict[k][indx]
                formal_batch[:number_relabels] = mask_relabel[:number_relabels]
                batch[k] = formal_batch
            elif k == 'dones' and self.mask_on_reward and her:
                formal_batch = self.dataset_dict[k][indx]
                formal_batch[:number_relabels] = done_relabel[:number_relabels]
                batch[k] = formal_batch
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)