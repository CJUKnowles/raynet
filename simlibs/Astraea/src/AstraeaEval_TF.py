"""Evaluate an original TensorFlow Astraea checkpoint in one OMNeT++ simulation."""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ASTRAEA_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASTRAEA_SRC_DIR))

from AstraeaEnv import AstraeaEnv, action_scalar  # noqa: E402
from training.learner import Agent, tf  # noqa: E402


CHECKPOINT_DIR = Path(os.getenv("RAYNET_PATH", "/home/james/raynet")) / "_models/astraea-papermodel"
H1_SIZE = 256
H2_SIZE = 128
STACKING = 5
SEED = 91456211


def parse_range(value, cast=float):
    """Parse MIN,MAX command-line ranges."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be MIN,MAX")
    return cast(parts[0]), cast(parts[1])


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
    parser = argparse.ArgumentParser(description="Evaluate an original Astraea TensorFlow checkpoint")
    parser.add_argument("ini_path", help="Path to the OMNeT++ ini file to run")
    parser.add_argument("config_section", nargs="?", default="General", help="OMNeT++ config section to run")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR), help="TensorFlow checkpoint directory or prefix")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="Maximum environment steps before ending early; zero runs until natural completion")
    parser.add_argument("--num-flows", type=lambda x: parse_range(x, int), default=(2, 2))
    parser.add_argument("--bandwidth", type=lambda x: parse_range(x, float), default=(10, 10))
    parser.add_argument("--rtt", type=lambda x: parse_range(x, float), default=(25, 25))
    parser.add_argument("--buffer-bits", type=lambda x: parse_range(x, int), default=(2_000_000, 2_000_000))
    parser.add_argument("--stacking", type=int, default=STACKING)
    return parser.parse_args()


def build_inference_agent(env):
    """Create the original Astraea learner for inference."""
    return Agent(
        env.state_dim,
        env.global_state_dim,
        env.action_dim,
        h1_shape=H1_SIZE,
        h2_shape=H2_SIZE,
        batch_size=1,
        summary=None,
        mem_size=1,
        noise_type=5,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
        is_global=True,
    )


def main():
    """Evaluate the original Astraea TensorFlow learner in one simulation."""
    # Resolve the requested simulation and model checkpoint.
    args = parse_args()
    if args.max_episode_steps < 0:
        raise ValueError("--max-episode-steps must not be negative")
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint}", flush=True)

    # Seed the learner and construct the protocol-specific environment.
    np.random.seed(SEED)
    env = AstraeaEnv(
        {
            "iniPath": str(Path(args.ini_path).resolve()),
            "section": args.config_section,
            "bottleneck_bw_range": args.bandwidth,
            "minimum_rtt_range": args.rtt,
            "bottleneck_buffer_range": args.buffer_bits,
            "max_steps_range": (args.max_episode_steps or 1_000_000_000, args.max_episode_steps or 1_000_000_000),
            "num_flows_range": args.num_flows,
            "stacking": args.stacking,
        }
    )
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
                states, _ = env.reset()
                episode_done = False
                while not episode_done:
                    actions = {agent_id: action_scalar(agent.get_action(state, use_noise=False)) for agent_id, state in states.items()}
                    result = env.step(actions)
                    states = result.states
                    for agent_id, reward in result.rewards.items():
                        total_rewards[agent_id] += reward
                    episode_length += 1
                    reached_step_limit = args.max_episode_steps > 0 and episode_length >= args.max_episode_steps
                    episode_done = result.episode_done or reached_step_limit
            finally:
                env.close()

    # Report the total return collected by each Astraea agent.
    print(f"Evaluation complete: length={episode_length}")
    for agent_id, reward in sorted(total_rewards.items()):
        print(f"{agent_id}: return={reward:.6f}")


if __name__ == "__main__":
    main()
