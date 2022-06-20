from typing import Callable, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax.scipy as jsp

from jaxrl2.networks.autoregressive import Autoregressive
from jaxrl2.networks.masked_mlp import MaskedMLP
from jaxrl2.types import PRNGKey


class MyUniform(distrax.Uniform):

    def prob(self, inputs):
        return super().prob(inputs) + 1e-8


class MADEDiscretizedUniform(Autoregressive):

    def __init__(self,
                 param_fn: Callable[..., Tuple[jnp.ndarray, jnp.ndarray,
                                               jnp.ndarray]],
                 batch_shape: Tuple[int],
                 event_dim: int,
                 beam_size: int = 20):
        self._param_fn = param_fn
        super().__init__(event_dim, jnp.float32, batch_shape, beam_size)

    def _distr_fn(self, samples: jnp.ndarray) -> distrax.Distribution:
        logits = self._param_fn(samples)

        num_components = logits.shape[-1]
        xs = jnp.linspace(-1, 1, num_components + 1)
        low = jnp.broadcast_to(xs[:-1], logits.shape)
        high = jnp.broadcast_to(xs[1:], logits.shape)

        dist = MyUniform(low=low, high=high)

        weights = distrax.Categorical(logits=logits)

        dist = distrax.MixtureSameFamily(weights, dist)
        return distrax.Independent(dist, reinterpreted_batch_ndims=1)

    def mode(self):
        beam = jnp.zeros(
            (self._beam_size, *self._batch_shape, self._event_dim),
            self._event_dtype)

        beam_log_probs = jnp.full((self._beam_size, *self._batch_shape), -100,
                                  jnp.float32)
        beam_log_probs = beam_log_probs.at[0].set(0)

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(beam)
            mix_dist = dist.distribution
            means = (mix_dist.components_distribution.high +
                     mix_dist.components_distribution.low) / 2
            log_probs = mix_dist._mixture_log_probs

            means = jnp.moveaxis(means, -1, 0)
            means = means[..., i]
            log_probs = jnp.moveaxis(log_probs, -1, 0)
            log_probs = log_probs[..., i]
            beam_log_probs = jnp.broadcast_to(beam_log_probs, log_probs.shape)

            means = jnp.reshape(means, (-1, *self._batch_shape))
            log_probs = jnp.reshape(log_probs, (-1, *self._batch_shape))
            beam_log_probs = jnp.reshape(beam_log_probs,
                                         (-1, *self._batch_shape))

            log_probs = jsp.special.logsumexp(jnp.stack(
                [beam_log_probs, log_probs], axis=0),
                                              axis=0)
            indx = jnp.argsort(log_probs, axis=0)[-self._beam_size:]

            beam = jnp.take_along_axis(beam,
                                       indx[..., jnp.newaxis] %
                                       self._beam_size,
                                       axis=0)
            means = jnp.take_along_axis(means, indx, axis=0)
            beam_log_probs = jnp.take_along_axis(log_probs, indx, axis=0)

            beam = beam.at[..., i].set(means)

        return beam[-1]


class MADEDiscretizedPolicy(nn.Module):
    features: Sequence[int]
    action_dim: int
    num_components: int = 100
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 states: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        is_initializing = len(self.variables) == 0
        masked_mlp = MaskedMLP(
            (*self.features, self.num_components * self.action_dim),
            dropout_rate=self.dropout_rate)

        if is_initializing:
            actions = jnp.zeros((*states.shape[:-1], self.action_dim),
                                states.dtype)
            masked_mlp(actions, states)

        def param_fn(actions: jnp.ndarray) -> distrax.Distribution:
            logits = masked_mlp(actions, states, training=training)

            new_shape = (*logits.shape[:-1], self.num_components,
                         actions.shape[-1])
            logits = jnp.reshape(logits, new_shape)
            logits = jnp.swapaxes(logits, -1, -2)

            return logits

        return MADEDiscretizedUniform(param_fn, states.shape[:-1],
                                      self.action_dim)
