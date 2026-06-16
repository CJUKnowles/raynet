"""Vendored original Astraea TensorFlow learner."""

import re
import sys
from pathlib import Path

from .tensorflow_compat import tf


LEARNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEARNER_DIR))

from agent.agent import Actor, Agent, Critic  # noqa: E402


def _exact_train_var(self):
    """Return only variables from this network scope."""
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=f"^{re.escape(self.name)}/",
    )


Actor.train_var = _exact_train_var
Critic.train_var = _exact_train_var

__all__ = ["Agent", "tf"]
