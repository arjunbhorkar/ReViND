from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp

from jaxrl2.types import PRNGKey


@dataclass
class Autoregressive(ABC, distrax.Distribution):
    _event_dim: int
    _event_dtype: jnp.dtype
    _batch_shape: Tuple[int]
    _beam_size: int

    @abstractmethod
    def _distr_fn(self, samples: jnp.ndarray) -> distrax.Distribution:
        pass

    def _sample_n(self, key: PRNGKey, n: int) -> jnp.ndarray:
        keys = jax.random.split(key, self._event_dim)

        samples = jnp.zeros((n, *self._batch_shape, self._event_dim),
                            self._event_dtype)

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(samples)
            dim_samples = dist.sample(seed=keys[i])
            samples = jax.ops.index_update(samples, jax.ops.index[..., i],
                                           dim_samples[..., i])

        return samples

    def log_prob(self, values: jnp.ndarray) -> jnp.ndarray:
        return self._distr_fn(values).log_prob(values)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self._event_dim, )

    @abstractmethod
    def mode(self):
        pass
