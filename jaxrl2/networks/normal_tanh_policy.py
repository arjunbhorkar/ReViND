from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init


class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)
        super().__init__(distribution=distribution,
                         bijector=distrax.Block(distrax.Tanh(), 1))

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init())(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return TanhMultivariateNormalDiag(loc=means,
                                          scale_diag=jnp.exp(log_stds))
