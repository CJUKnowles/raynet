"""
Train the OMNeT++ Astraea implementation with the original TensorFlow learner.

This is intentionally a small bridge rather than a rewrite of the original
learner. The vendored TensorFlow implementation stays separate from the
Astraea environment wrapper and this training loop.
"""

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ASTRAEA_SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASTRAEA_SRC_DIR))

from AstraeaEnv import AstraeaEnv, action_scalar  # noqa: E402
from training.learner import Agent, tf  # noqa: E402


RAYNET_PATH = Path(os.getenv("RAYNET_PATH", "/home/james/raynet"))
TRAINING_INI = RAYNET_PATH / "simlibs/Astraea/src/training/AstraeaTraining.ini"


# MARK: Logging -------------------------------------------------------------------------------------------------
def write_scalar(summary_writer, tag, value, step):
    """Write one scalar value to TensorBoard."""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=float(np.asarray(value).reshape(-1)[0]))
    summary_writer.add_summary(summary, global_step=step)


def log_rollout_step(summary_writer, agent, step, mean_reward, worker_actions, next_states):
    """Log one completed environment step."""
    write_scalar(summary_writer, "Rollout/mean_reward", mean_reward, step)
    write_scalar(summary_writer, "Rollout/num_agents", len(next_states), step)
    write_scalar(summary_writer, "Rollout/replay_size", len(agent.rp_buffer), step)
    for agent_id, action in worker_actions.items():
        write_scalar(summary_writer, "Rollout/action", action, step)


def log_training_performance(summary_writer, step, start_time):
    """Log aggregate environment-step throughput."""
    write_scalar(
        summary_writer,
        "Performance/environment_steps_per_second",
        step / max(time.time() - start_time, 1e-6),
        step,
    )


def log_completed_episode(summary_writer, agent, episode, worker_id, step, episode_steps, episode_return):
    """Log one completed training episode."""
    print(
        f"episode={episode} worker={worker_id} step={step} episode_steps={episode_steps} "
        f"return={episode_return:.4f} buffer={len(agent.rp_buffer)}",
        flush=True,
    )
    write_scalar(summary_writer, "Episode/return", episode_return, episode)
    write_scalar(summary_writer, "Episode/length", episode_steps, episode)
    summary_writer.flush()


# MARK: Utilities -----------------------------------------------------------------------------------------------
def parse_range(value, cast=float):
    """Parse MIN,MAX command-line ranges."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be MIN,MAX")
    return cast(parts[0]), cast(parts[1])


def resolve_checkpoint(path):
    """Resolve a checkpoint directory or checkpoint file to its prefix."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        checkpoint = tf.train.latest_checkpoint(str(candidate))
        if checkpoint is None:
            raise ValueError(f"No TensorFlow checkpoint found in {candidate}")
        return checkpoint
    return str(candidate)


def save_checkpoint(agent, checkpoint_dir, step, final=False):
    """Save and report one learner checkpoint."""
    agent.ckpt_dir = str(checkpoint_dir)
    agent.save_model(step)
    print(f"Saved {'final ' if final else ''}checkpoint at step {step}", flush=True)


def build_session_config():
    """Configure TensorFlow to grow GPU memory usage as needed."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


# MARK: Core Functionality --------------------------------------------------------------------------------------
def build_agent(env, args, summary_writer):
    """Create the original Astraea learner. (a single shared learning agent across all environments)"""
    return Agent(
        env.state_dim,
        env.global_state_dim,
        env.action_dim,
        h1_shape=args.h1_size,
        h2_shape=args.h2_size,
        batch_size=args.batch_size,
        summary=summary_writer,
        stddev=args.stddev,
        policy_delay=args.policy_delay,
        mem_size=args.replay_size,
        gamma=args.gamma,
        lr_c=args.critic_lr,
        lr_a=args.actor_lr,
        tau=args.tau,
        PER=False,
        LOSS_TYPE=args.loss_type,
        noise_type=args.noise_type,
        noise_exp=args.noise_exp,
        train_exp=args.train_exp,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
        is_global=not args.local_critic,
    )


# MARK: Configuration -------------------------------------------------------------------------------------------
def parse_args():
    """Parse the training configuration."""
    parser = argparse.ArgumentParser(description="Original TensorFlow Astraea learner for OMNeT++")

    # Define network parameter ranges.
    parser.add_argument("ini_path", nargs="?", default=str(TRAINING_INI))
    parser.add_argument("--section", default="General")
    parser.add_argument("--num-flows", type=lambda x: parse_range(x, int), default=(2, 5))
    parser.add_argument("--bandwidth", type=lambda x: parse_range(x, float), default=(5, 20))
    parser.add_argument("--rtt", type=lambda x: parse_range(x, float), default=(5, 100))
    parser.add_argument("--buffer-bits", type=lambda x: parse_range(x, int), default=(25_000, 2_000_000))
    parser.add_argument("--max-episode-steps", type=int, default=2000)

    # Define general training and output arguments.
    parser.add_argument("--log-dir", default=str(RAYNET_PATH / "_models" / "Astraea-original-tf"))
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--restore", default="")
    parser.add_argument("--quiet-env", action="store_true")

    # Define simulation orchestration.
    parser.add_argument("--seed", type=int, default=91456211)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--stacking", type=int, default=5)
    parser.add_argument("--train-after", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=20)
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--local-critic", action="store_true")
    parser.add_argument("--no-bootstrap-on-truncation", action="store_true")

    # Define original Astraea learner hyperparameters.
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--replay-size", type=int, default=400_000)
    parser.add_argument("--h1-size", type=int, default=256)
    parser.add_argument("--h2-size", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=0.00005)
    parser.add_argument("--critic-lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--stddev", type=float, default=1.0)
    parser.add_argument("--noise-type", type=int, default=0)
    parser.add_argument("--noise-exp", type=int, default=1_500_000)
    parser.add_argument("--train-exp", type=int, default=100_000)
    parser.add_argument("--policy-delay", type=int, default=20)
    parser.add_argument("--loss-type", choices=["MSE", "HUBER"], default="HUBER")
    return parser.parse_args()


def validate_args(args):
    """Validate training arguments before constructing simulations."""
    # Validate scalar counts and intervals.
    if args.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be at least 1")
    if args.train_every < 1:
        raise ValueError("--train-every must be at least 1")

    # Validate randomized training ranges.
    for name in ("num_flows", "bandwidth", "rtt", "buffer_bits"):
        lower, upper = getattr(args, name)
        if lower > upper:
            raise ValueError(f"--{name.replace('_', '-')} minimum must not exceed its maximum")


# MARK: Training Loop -------------------------------------------------------------------------------------------
def main():
    """Train the original Astraea learner across parallel simulations."""
    # Parse arguments and seed the learner.
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Construct the shared environment configuration.
    env_config = {
        "iniPath": str(Path(args.ini_path).resolve()),
        "section": args.section,
        "bottleneck_bw_range": args.bandwidth,
        "minimum_rtt_range": args.rtt,
        "bottleneck_buffer_range": args.buffer_bits,
        "max_steps_range": (args.max_episode_steps, args.max_episode_steps),
        "num_flows_range": args.num_flows,
        "stacking": args.stacking,
        "randomize_ini": True,
    }

    # Create output directories and persistent environment wrappers.
    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    envs = [
        AstraeaEnv(
            dict(env_config, worker_id=worker_id),
            bootstrap_on_truncation=not args.no_bootstrap_on_truncation,
            verbose=not args.quiet_env,
        )
        for worker_id in range(args.num_simulations)
    ]

    print(f"Running {args.num_simulations} parallel simulation(s) for {args.total_steps} aggregate steps.", flush=True)

    # Build and initialize the learner graph.
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        summary_writer = tf.summary.FileWriter(str(log_dir))
        agent = build_agent(envs[0], args, summary_writer)
        agent.build_learn()
        agent.create_tf_summary()

        with tf.Session(config=build_session_config()) as sess:
            agent.assign_sess(sess)
            sess.run(tf.global_variables_initializer())
            if args.restore:
                agent.saver.restore(sess, resolve_checkpoint(args.restore))
            else:
                agent.init_target()

            with ThreadPoolExecutor(max_workers=args.num_simulations) as executor:
                # Reset every environment and initialize persistent rollout state.
                reset_results = list(executor.map(lambda env: env.reset(), envs))
                states = [item[0] for item in reset_results]
                globals_ = [item[1] for item in reset_results]
                episode_returns = [0.0] * args.num_simulations
                episode_steps = [0] * args.num_simulations
                completed_episodes = 0
                step = 0
                train_step = 0
                start_time = time.time()

                try:
                    while step < args.total_steps:
                        # Select actions and advance all active simulations in parallel.
                        active_count = min(args.num_simulations, args.total_steps - step)
                        active_workers = list(range(active_count))
                        actions = [
                            {
                                agent_id: action_scalar(agent.get_action(state, use_noise=True))
                                for agent_id, state in states[worker_id].items()
                            }
                            for worker_id in active_workers
                        ]
                        futures = {
                            worker_id: executor.submit(envs[worker_id].step, actions[index])
                            for index, worker_id in enumerate(active_workers)
                        }
                        reset_workers = []

                        # Process each completed simulation step.
                        for index, worker_id in enumerate(active_workers):
                            old_states = states[worker_id]
                            old_global = globals_[worker_id]
                            result = futures[worker_id].result()
                            worker_actions = actions[index]
                            step += 1
                            episode_steps[worker_id] += 1

                            # Store transitions for agents that have both current and next observations.
                            step_rewards = []
                            common_agents = old_states.keys() & result.states.keys() & worker_actions.keys()
                            for agent_id in common_agents:
                                reward = float(result.rewards.get(agent_id, 0.0))
                                if math.isnan(reward):
                                    reward = 0.0
                                step_rewards.append(reward)
                                agent.store_experience(
                                    old_states[agent_id],
                                    old_global,
                                    np.asarray([worker_actions[agent_id]], dtype=np.float32),
                                    np.asarray([reward], dtype=np.float32),
                                    result.states[agent_id],
                                    result.global_state,
                                    np.asarray([float(result.terminals.get(agent_id, False))], dtype=np.float32),
                                )

                            # Log rollout state and update persistent worker state.
                            mean_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
                            episode_returns[worker_id] += mean_reward
                            states[worker_id] = result.states
                            globals_[worker_id] = result.global_state
                            log_rollout_step(summary_writer, agent, step, mean_reward, worker_actions, result.states)
                            log_training_performance(summary_writer, step, start_time)

                            # Train after enough valid learner transitions have been collected.
                            if len(agent.rp_buffer) >= args.train_after and step % args.train_every == 0:
                                for _ in range(args.updates_per_train):
                                    train_step += 1
                                    agent.train_step(train_step)

                            # Schedule completed environments for reset.
                            if result.episode_done:
                                log_completed_episode(
                                    summary_writer,
                                    agent,
                                    completed_episodes,
                                    worker_id,
                                    step,
                                    episode_steps[worker_id],
                                    episode_returns[worker_id],
                                )
                                completed_episodes += 1
                                episode_returns[worker_id] = 0.0
                                episode_steps[worker_id] = 0
                                if step < args.total_steps:
                                    reset_workers.append(worker_id)

                            # Save periodic checkpoints after the completed step.
                            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                                save_checkpoint(agent, checkpoint_dir, step)

                        # Reset environments after all current results are processed.
                        if reset_workers:
                            reset_futures = {
                                worker_id: executor.submit(envs[worker_id].reset)
                                for worker_id in reset_workers
                            }
                            for worker_id, future in reset_futures.items():
                                states[worker_id], globals_[worker_id] = future.result()
                            if agent.actor_noise is not None:
                                agent.actor_noise.reset()
                        summary_writer.flush()

                finally:
                    # Save the final learner state and close every environment.
                    save_checkpoint(agent, checkpoint_dir, step, final=True)
                    list(executor.map(lambda env: env.close(), envs))
                    summary_writer.close()


if __name__ == "__main__":
    main()
