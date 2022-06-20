from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        # print("img l", observations.shape)
        x = observations.astype(jnp.float32) / 255.0
        # print("img", x.shape)
        # x = jnp.reshape(x, (*x.shape[:-2], -1))
        # print("img", x.shape)

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            # print("img", x.shape)
            x = nn.relu(x)
            # print("img", x.shape)

        return x.reshape((*x.shape[:-3], -1))


class PixelCritic(nn.Module):
    encoder: nn.Module
    critic: nn.Module
    latent_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 imageobservations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # print("img len", imageobservations.shape)
        x = self.encoder(imageobservations)
        # print("n", x.shape)
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        # print("n", x.shape)
        x = nn.LayerNorm()(x)

        x = nn.tanh(x)

        # x = jnp.reshape(x, (1, 50))

        x = jnp.concatenate([observations, x], -1)

        # actions = actions.reshape((actions.shape[-1], )) 
        return self.critic(x, actions)


class PixelValue(nn.Module):
    encoder: nn.Module
    value: nn.Module
    latent_dim: int

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 imageobservations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = self.encoder(imageobservations)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)

        x = nn.LayerNorm()(x)

        x = nn.tanh(x)

        # x = jnp.reshape(x, (1, 50))

        x = jnp.concatenate([observations, x], -1)

        return self.value(x)


class PixelPolicy(nn.Module):
    encoder: nn.Module
    policy: nn.Module
    latent_dim: int

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 imageobservations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        x = self.encoder(imageobservations)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)

        x = nn.tanh(x)

        # x = jnp.reshape(x, (1, 50))

        x = jnp.concatenate([observations, x], -1)

        return self.policy(x, training=training)
