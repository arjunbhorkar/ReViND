import gym

from jaxrl2.wrappers.single_precision import SinglePrecision
from jaxrl2.wrappers.universal_seed import UniversalSeed


def wrap_gym(env: gym.Env) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    env = gym.wrappers.RescaleAction(env, -1, 1)

    return env