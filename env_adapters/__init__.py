from .base_env import BaseEnv

ENV_REGISTRY = {}

def register_env(name):
    def decorator(cls):
        if not issubclass(cls, BaseEnv):
            raise TypeError(f"{cls.__name__} must subclass BaseEnv")
        ENV_REGISTRY[name] = cls
        return cls
    return decorator

def make_env(name, cfg):
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {name}")
    return ENV_REGISTRY[name](cfg)

from . import libero_env