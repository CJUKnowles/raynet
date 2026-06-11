import argparse
import math
import os
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


RAYNET_PATH = Path(os.getenv("RAYNET_PATH", "/home/james/raynet"))
TRAINING_DIR = RAYNET_PATH / "simlibs/Orca/src/training"
sys.path.insert(0, str(TRAINING_DIR))

from OrcaTraining_TD3 import (
    Agent,
    action_scalar,
    tf,
)


CHECKPOINT_DIR = RAYNET_PATH / "_models/Orca-papermodel"  # Path to the directory containing the checkpoint to evaluate
HIDDEN_SIZE = 256
STACKING = 10
SEED = 91456211


class OmnetOrcaEvalEnv:
    """Direct OMNeT++ adapter matching OrcaEval.py's lifecycle."""

    raw_obs_dim = 7
    ack_count_index = 3

    def __init__(self, env_config):
        sys.path.insert(0, str(RAYNET_PATH / "build"))
        from omnetbind import OmnetGymApi

        self.runner = OmnetGymApi()
        self.env_config = env_config
        self.stacking = int(env_config["stacking"])
        self.obs_history = {}
        self.closed = True

    @property
    def state_dim(self):
        return self.stacking * self.raw_obs_dim

    @property
    def action_dim(self):
        return 1

    def reset(self):
        self.obs_history = {}
        self.runner.initialise(
            self.env_config["iniPath"],
            self.env_config["config_section"],
        )
        self.closed = False

        observations = self.runner.reset()
        states = self._stack_observations(observations)
        if not states:
            self.close()
            raise RuntimeError(
                "runner.reset() did not return any Orca observations. "
                f"Returned keys were {list(observations.keys())}."
            )

        return states

    def step(self, learner_actions):
        actions = {
            agent_id: float(np.clip(action_scalar(action), -1.0, 1.0))
            for agent_id, action in learner_actions.items()
        }
        observations, rewards, terminateds, info = self.runner.step(actions)
        states = self._stack_observations(observations)

        cleaned_rewards = {}
        for agent_id, reward in rewards.items():
            reward = float(reward)
            if math.isnan(reward):
                print(f"Warning: NaN reward for {agent_id}; replacing with 0.0")
                reward = 0.0
            cleaned_rewards[agent_id] = reward

        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(info.get("simDone", False))
        if terminated:
            self.runner.shutdown()
            self.runner.cleanup()
            self.closed = True
        elif truncated:
            self.runner.cleanup()
            self.closed = True

        return states, cleaned_rewards, terminateds, terminated or truncated

    def close(self):
        if self.closed:
            return
        self.runner.shutdown()
        self.runner.cleanup()
        self.closed = True

    def _stack_observations(self, observations):
        states = {}
        for agent_id, observation in observations.items():
            if agent_id in {"__all__", "SIMULATION_END"}:
                continue
            if agent_id not in self.obs_history:
                # The paper actor was trained with zero-padded recurrent history.
                self.obs_history[agent_id] = deque(
                    np.zeros(self.state_dim, dtype=np.float32),
                    maxlen=self.state_dim,
                )
            self.obs_history[agent_id].extend(observation)
            states[agent_id] = np.asarray(
                list(self.obs_history[agent_id]),
                dtype=np.float32,
            )
        return states

    def latest_observation_has_acks(self, state):
        latest_obs = np.asarray(state)[-self.raw_obs_dim:]
        return bool(latest_obs[self.ack_count_index] > 0.0)


def resolve_checkpoint(path):
    candidate = Path(path).expanduser().resolve()

    if candidate.is_dir():
        direct = tf.train.latest_checkpoint(str(candidate))
        if direct:
            return direct

        nested = candidate / "checkpoints"
        if nested.is_dir():
            nested_checkpoint = tf.train.latest_checkpoint(str(nested))
            if nested_checkpoint:
                return nested_checkpoint

        raise FileNotFoundError(f"No TensorFlow checkpoint found under {candidate}")

    checkpoint_prefix = str(candidate)
    if candidate.suffix in {".meta", ".index"}:
        checkpoint_prefix = str(candidate.with_suffix(""))
    if Path(checkpoint_prefix + ".index").is_file():
        return checkpoint_prefix

    raise FileNotFoundError(f"Checkpoint prefix does not exist: {checkpoint_prefix}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an original Orca TD3 checkpoint")
    parser.add_argument(
        "ini_path",
        help="Path to the OMNeT++ ini file to run",
    )
    parser.add_argument(
        "config_section",
        nargs="?",
        default="Orca",
        help="OMNeT++ config section to run (default: Orca)",
    )
    return parser.parse_args()


def build_inference_agent(env, hidden_size):
    return Agent(
        env.state_dim,
        env.action_dim,
        h1_shape=hidden_size,
        h2_shape=hidden_size,
        batch_size=1,
        summary=None,
        mem_size=1,
        noise_type=5,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
    )


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(CHECKPOINT_DIR)
    print(f"Loading checkpoint: {checkpoint}", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)

    env_config = {
        "iniPath": args.ini_path,
        "config_section": args.config_section,
        "stacking": STACKING,
    }
    env = OmnetOrcaEvalEnv(env_config)

    total_rewards = defaultdict(float)
    episode_length = 0

    with tf.Graph().as_default():
        tf.set_random_seed(SEED)
        agent = build_inference_agent(env, HIDDEN_SIZE)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            agent.assign_sess(sess)
            saver.restore(sess, checkpoint)

            try:
                states = env.reset()
                actions = defaultdict(float)
                episode_done = False
                while not episode_done:
                    for agent_id, state in states.items():
                        if env.latest_observation_has_acks(state):
                            actions[agent_id] = action_scalar(
                                agent.get_action(state, use_noise=False)
                            )

                    states, rewards, _, episode_done = env.step(
                        {
                            agent_id: actions[agent_id]
                            for agent_id in states
                        }
                    )
                    for agent_id, reward in rewards.items():
                        total_rewards[agent_id] += reward
                    episode_length += 1
            finally:
                env.close()

    print(f"Evaluation complete: length={episode_length}")
    for agent_id, reward in sorted(total_rewards.items()):
        print(f"{agent_id}: return={reward:.6f}")


if __name__ == "__main__":
    main()
