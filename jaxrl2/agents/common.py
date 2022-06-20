from functools import partial
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_log_prob_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, batch: DatasetDict) -> float:
    dist = actor_apply_fn({'params': actor_params}, batch['observations'])
    log_probs = dist.log_prob(batch['actions'])
    return log_probs.mean()


@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_actions_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                     actor_params: Params,
                     observations: np.ndarray,
                     imageobservations: np.ndarray) -> jnp.ndarray:
    dist = actor_apply_fn({'params': actor_params}, observations, imageobservations)
    return dist.mode()

@partial(jax.jit, static_argnames='critic_apply_fn')
def eval_critic_jit(critic_apply_fn: Callable[..., distrax.Distribution],
                     critic_params: Params,
                     observations: np.ndarray,
                     imageobservations: np.ndarray,
                     actions: np.ndarray) -> float:
    qval = critic_apply_fn({'params': critic_params}, observations, imageobservations, actions)
    return qval

@partial(jax.jit, static_argnames='value_apply_fn')
def eval_value_jit(value_apply_fn: Callable[..., distrax.Distribution],
                     value_params: Params,
                     observations: np.ndarray,
                     imageobservations: np.ndarray) -> float:
    qval = value_apply_fn({'params': value_params}, observations, imageobservations)
    return qval

@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
