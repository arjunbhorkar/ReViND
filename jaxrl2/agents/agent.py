import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.agents.common import (eval_actions_jit, eval_critic_jit, eval_value_jit, eval_log_prob_jit,
                                  sample_actions_jit)
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import PRNGKey


class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray, imageobservations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self._actor.apply_fn, self._actor.params,
                                   observations, imageobservations)
        # actions = np.asarray(actions)
        # return np.clip(actions, -1, 1)
        return np.asarray(actions)

    def eval_critic(self, observations: np.ndarray, imageobservations: np.ndarray, actions: np.ndarray) -> float:
        qval = eval_critic_jit(self._critic.apply_fn, self._critic.params,
                                   observations, imageobservations, actions)
        # qval = np.asarray(qval)
        return qval

    def eval_value(self, observations: np.ndarray, imageobservations: np.ndarray) -> float:
        val = eval_value_jit(self._value.apply_fn, self._value.params,
                                   observations, imageobservations)
        # qval = np.asarray(qval)
        return val

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params,
                                 batch)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(self._rng, self._actor.apply_fn,
                                          self._actor.params, observations)

        self._rng = rng

        # actions = np.asarray(actions)
        return np.asarray(actions)
        # return np.clip(actions, -1, 1)
