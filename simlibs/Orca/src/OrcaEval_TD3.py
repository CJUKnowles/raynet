import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from OrcaEnv import OrcaEnv, action_scalar
from training.learner import Agent, tf


CHECKPOINT_DIR = Path(os.environ["RAYNET_PATH"]) / "_models/Orca-papermodel"
HIDDEN_SIZE = 256
STACKING = 10
SEED = 91456211


def resolve_checkpoint(path):
    """Resolve a checkpoint directory or checkpoint file to its prefix."""
    # Search the provided directory and its checkpoints subdirectory.
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        for directory in (candidate, candidate / "checkpoints"):
            checkpoint = tf.train.latest_checkpoint(str(directory))
            if checkpoint:
                return checkpoint
        raise FileNotFoundError(f"No TensorFlow checkpoint found under {candidate}")

    # Remove TensorFlow metadata suffixes and validate the checkpoint prefix.
    checkpoint_prefix = str(candidate)
    if candidate.suffix in {".meta", ".index"}:
        checkpoint_prefix = str(candidate.with_suffix(""))
    if Path(checkpoint_prefix + ".index").is_file():
        return checkpoint_prefix

    raise FileNotFoundError(f"Checkpoint prefix does not exist: {checkpoint_prefix}")


def parse_args():
    """Parse the evaluation configuration."""
    parser = argparse.ArgumentParser(description="Evaluate an original Orca TD3 checkpoint")
    parser.add_argument("ini_path", help="Path to the OMNeT++ ini file to run")
    parser.add_argument("config_section", nargs="?", default="Orca", help="OMNeT++ config section to run")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="Maximum valid learner observations before ending early; zero runs until natural completion")
    return parser.parse_args()


def build_inference_agent(env):
    """Create the original Orca TD3 learner for inference."""
    return Agent(
        env.state_dim,
        env.action_dim,
        h1_shape=HIDDEN_SIZE,
        h2_shape=HIDDEN_SIZE,
        batch_size=1,
        summary=None,
        mem_size=1,
        noise_type=5,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
    )


def main():
    """Evaluate the original Orca paper model in one simulation."""
    # Resolve the requested simulation and paper-model checkpoint.
    args = parse_args()
    if args.max_episode_steps < 0:
        raise ValueError("--max-episode-steps must not be negative")
    checkpoint = resolve_checkpoint(CHECKPOINT_DIR)
    print(f"Loading checkpoint: {checkpoint}", flush=True)

    # Seed the learner and construct the protocol-specific environment.
    np.random.seed(SEED)
    env = OrcaEnv({"iniPath": args.ini_path, "config_section": args.config_section, "stacking": STACKING})
    total_rewards = defaultdict(float)
    episode_length = 0

    # Restore the learner and evaluate until the simulation completes.
    with tf.Graph().as_default():
        tf.set_random_seed(SEED)
        agent = build_inference_agent(env)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            agent.assign_sess(sess)
            saver.restore(sess, checkpoint)

            try:
                states = env.reset()
                episode_done = False
                while not episode_done:
                    actions = {agent_id: action_scalar(agent.get_action(state, use_noise=False)) for agent_id, state in states.items()}
                    result = env.step(actions)
                    states = result.states
                    for agent_id, reward in result.rewards.items():
                        total_rewards[agent_id] += reward
                    episode_length += int(bool(states))
                    reached_step_limit = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                    episode_done = result.episode_done or reached_step_limit
            finally:
                env.close()

    # Report the total return collected by each Orca agent.
    print(f"Evaluation complete: length={episode_length}")
    for agent_id, reward in sorted(total_rewards.items()):
        print(f"{agent_id}: return={reward:.6f}")


if __name__ == "__main__":
    main()
