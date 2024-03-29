"""Implementations of algorithms for continuous control."""

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2 import networks
from jaxrl2.agents.agent import Agent
from jaxrl2.agents.bc.actor_updater import log_prob_update

_log_prob_update_jit = jax.jit(log_prob_update)


class BCLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-3,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 dropout_rate: Optional[float] = None,
                 distr: str = 'tanh_normal'):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = actions.shape[-1]
        if distr == 'tanh_normal':
            actor_def = networks.NormalTanhPolicy(hidden_dims,
                                                  action_dim,
                                                  dropout_rate=dropout_rate)
        elif distr == 'ar':
            actor_def = networks.MADEDiscretizedPolicy(
                hidden_dims, action_dim, dropout_rate=dropout_rate)
        elif distr == 'mog_ar':
            actor_def = networks.MADETanhMoGPolicy(hidden_dims,
                                                   action_dim,
                                                   dropout_rate=dropout_rate,
                                                   squash_tanh=False)
        elif distr == 'tanh_mog_ar':
            actor_def = networks.MADETanhMoGPolicy(hidden_dims,
                                                   action_dim,
                                                   dropout_rate=dropout_rate)
        else:
            raise NotImplementedError()

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        optimiser = optax.adam(learning_rate=actor_lr)

        params = actor_def.init(actor_key, observations)['params']
        self._actor = TrainState.create(apply_fn=actor_def.apply,
                                        params=params,
                                        tx=optimiser)
        self._rng = rng

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        self._rng, self._actor, info = _log_prob_update_jit(
            self._rng, self._actor, batch)
        return info
