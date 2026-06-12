"""Vendored original Orca TensorFlow TD3 learner."""

from .tensorflow_compat import tf
from .agent import Agent

__all__ = ["Agent", "tf"]
