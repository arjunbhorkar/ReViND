from typing import Callable, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.autoregressive import Autoregressive
from jaxrl2.networks.masked_mlp import MaskedMLP


class MADETanhMoG(Autoregressive):

    def __init__(self,
                 param_fn: Callable[..., Tuple[jnp.ndarray, jnp.ndarray,
                                               jnp.ndarray]],
                 batch_shape: Tuple[int],
                 event_dim: int,
                 beam_size: int = 20,
                 squash_tanh: bool = True):
        self._param_fn = param_fn
        self._squash_tanh = squash_tanh
        super().__init__(event_dim, jnp.float32, batch_shape, beam_size)

    def _distr_fn(self, samples: jnp.ndarray) -> distrax.Distribution:
        logits, means, log_scales = self._param_fn(samples)

        dist = distrax.Normal(loc=means, scale=jnp.exp(log_scales))

        dist = distrax.MixtureSameFamily(distrax.Categorical(logits=logits),
                                         dist)

        if self._squash_tanh:
            dist = distrax.Transformed(dist, distrax.Tanh())
        return distrax.Independent(dist, reinterpreted_batch_ndims=1)

    def mode(self):
        beam = jnp.zeros(
            (self._beam_size, *self._batch_shape, self._event_dim),
            self._event_dtype)

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(beam)
            mix_dist = dist.distribution
            if self._squash_tanh:
                mix_dist = mix_dist.distribution
            mean = mix_dist.components_distribution.mean()
            if self._squash_tanh:
                mean = jnp.tanh(mean)
            lim = 1 - 1e-5
            mean = jnp.clip(mean, -lim, lim)
            mean = jnp.moveaxis(mean, -1, 0)
            candidate = jnp.repeat(beam[jnp.newaxis], mean.shape[0], axis=0)
            candidate = candidate.at[..., i].set(mean[..., i])
            log_probs = jnp.sum(
                dist.distribution.log_prob(candidate)[..., :i + 1], -1)
            log_probs = jnp.reshape(log_probs, (-1, *self._batch_shape))
            indx = jnp.argsort(log_probs, axis=0)[-self._beam_size:]

            candidate = jnp.reshape(candidate,
                                    (-1, *self._batch_shape, self._event_dim))
            beam = jnp.take_along_axis(candidate,
                                       jnp.expand_dims(indx, axis=-1),
                                       axis=0)
        return beam[-1]


class MADETanhMoGPolicy(nn.Module):
    features: Sequence[int]
    action_dim: int
    num_components: int = 100
    dropout_rate: Optional[float] = None
    log_std_min: int = -5
    log_std_max: int = 5
    squash_tanh: bool = True

    @nn.compact
    def __call__(self,
                 states: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        is_initializing = len(self.variables) == 0

        if self.squash_tanh:
            num_outputs = (self.num_components + 2) * self.action_dim
        else:
            num_outputs = (self.num_components + 1) * self.action_dim
        masked_mlp = MaskedMLP((*self.features, num_outputs),
                               dropout_rate=self.dropout_rate)

        if is_initializing:
            actions = jnp.zeros((*states.shape[:-1], self.action_dim),
                                states.dtype)
            masked_mlp(actions, states)

        if not self.squash_tanh:
            _log_scales = self.param('log_scales', nn.initializers.zeros,
                                     (self.action_dim, ))

        def param_fn(actions: jnp.ndarray) -> distrax.Distribution:
            outputs = masked_mlp(actions, states, training=training)
            if self.squash_tanh:
                logits = outputs[..., :-2 * self.action_dim]
                means = outputs[..., -2 * self.action_dim:-self.action_dim]
                log_scales = outputs[..., -self.action_dim:]
                log_scales = jnp.clip(log_scales, self.log_std_min,
                                      self.log_std_max)
            else:
                logits = outputs[..., :-self.action_dim]
                means = outputs[..., -self.action_dim:]
                log_scales = _log_scales

            new_shape = (*logits.shape[:-1], self.num_components,
                         actions.shape[-1])
            logits = jnp.reshape(logits, new_shape)
            logits = jnp.swapaxes(logits, -1, -2)

            centroids = jnp.linspace(-1, 1, self.num_components + 1)
            centroids = (centroids[:-1] + centroids[1:]) / 2
            if self.squash_tanh:
                centroids = jnp.arctanh(centroids)
            means = means[..., jnp.newaxis] + centroids
            log_scales = log_scales[..., jnp.newaxis]

            means = jnp.broadcast_to(means, logits.shape)
            log_scales = jnp.broadcast_to(log_scales, logits.shape)

            return logits, means, log_scales

        return MADETanhMoG(param_fn,
                           states.shape[:-1],
                           self.action_dim,
                           squash_tanh=self.squash_tanh)
