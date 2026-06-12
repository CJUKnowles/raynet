import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ORCA_SRC_DIR = Path(__file__).resolve().parent
TRAINING_DIR = ORCA_SRC_DIR / "training"
sys.path.insert(0, str(ORCA_SRC_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from OrcaEnv import OrcaEnv, action_scalar  # noqa: E402
from learner import Agent, tf  # noqa: E402


CHECKPOINT_DIR = os.path.join(os.getenv('RAYNET_PATH'), "_models/Orca-papermodel") # Path to the directory containing the checkpoint to evaluate
HIDDEN_SIZE = 256
STACKING = 10
SEED = 91456211


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
    env = OrcaEnv(env_config)

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
                states, observation_valids = env.reset()
                actions = defaultdict(float)
                episode_done = False
                while not episode_done:
                    for agent_id, state in states.items():
                        if observation_valids.get(agent_id, False):
                            actions[agent_id] = action_scalar(
                                agent.get_action(state, use_noise=False)
                            )

                    result = env.step(
                        {
                            agent_id: actions[agent_id]
                            for agent_id in states
                        }
                    )
                    states = result.states
                    observation_valids = result.observation_valids
                    for agent_id, reward in result.rewards.items():
                        total_rewards[agent_id] += reward
                    episode_done = result.episode_done
                    episode_length += 1
            finally:
                env.close()

    print(f"Evaluation complete: length={episode_length}")
    for agent_id, reward in sorted(total_rewards.items()):
        print(f"{agent_id}: return={reward:.6f}")


if __name__ == "__main__":
    main()
