"""
Train the OMNeT++ Orca implementation with the original Orca TD3 learner.

This is intentionally a small bridge rather than a rewrite of the original
learner. The vendored TensorFlow TD3 implementation stays separate from the
Orca environment wrapper and this training loop.
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ORCA_SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCA_SRC_DIR))

from OrcaEnv import OrcaEnv, action_scalar  # noqa: E402
from learner import Agent, tf  # noqa: E402


def write_scalar(summary_writer, tag, value, step):
    """Write one scalar value to TensorBoard."""
    value = np.asarray(value).reshape(-1)[0]
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=float(value))
    summary_writer.add_summary(summary, global_step=step)


def prepare_training_ini(training_config, rng, worker_id):
    """Create one randomized INI variant for a training episode."""
    # Sample the randomized link and episode parameters.
    bw = round(rng.uniform(*training_config["bottleneck_bw_range"]))
    base_rtt = round(rng.uniform(*training_config["minimum_rtt_range"]), 2)
    buffer_size = round(bw * base_rtt * rng.uniform(0.2, 4.0) * 1000)
    max_steps = round(rng.uniform(*training_config["max_steps_range"]))

    # Create a unique INI variant for this training worker.
    original_ini_file = Path(training_config["iniPath"])
    ini_variants_dir = original_ini_file.parent / "ini_variants"
    ini_variants_dir.mkdir(parents=True, exist_ok=True)
    worker_ini_file = ini_variants_dir / f"{original_ini_file.name}.worker{os.getpid()}-{worker_id}"

    # Replace the training placeholders and write the generated INI.
    ini_string = original_ini_file.read_text(encoding="utf-8")
    ini_string = ini_string.replace("HOME", os.getenv("HOME", str(Path.home())))
    ini_string = ini_string.replace("ORCA_BOTTLENECK_BW", f"{bw}Mbps")
    ini_string = ini_string.replace("ORCA_BASE_RTT", f"{base_rtt / 2.0}ms")
    ini_string = ini_string.replace("ORCA_BOTTLENECK_BUFFER_SIZE", f"{buffer_size}b")
    ini_string = ini_string.replace("MAX_RL_STEPS", str(max_steps))
    worker_ini_file.write_text(ini_string, encoding="utf-8")
    return worker_ini_file


def terminal_for_training(step, agent_id, bootstrap_on_truncation=True):
    """Return whether a step should be terminal in the replay buffer."""
    terminated = bool(step.terminateds.get(agent_id, step.terminateds.get("__all__", False)))
    return terminated or (step.truncated and not bootstrap_on_truncation)


def build_agent(env, args, summary_writer):
    """Create the original Orca TD3 learner."""
    return Agent(
        env.state_dim,
        env.action_dim,
        h1_shape=args.hidden_size,
        h2_shape=args.hidden_size,
        batch_size=args.batch_size,
        summary=summary_writer,
        stddev=args.stddev,
        mem_size=args.replay_size,
        gamma=args.gamma,
        lr_c=args.critic_lr,
        lr_a=args.actor_lr,
        tau=args.tau,
        PER=False,
        CDQ=True,
        LOSS_TYPE=args.loss_type,
        noise_type=args.noise_type,
        noise_exp=args.noise_exp,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Original Orca TD3 learner for OMNeT++")
    parser.add_argument("--seed", type=int, default=91456211)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=16,
        help="Number of OMNeT++ simulations to run in parallel",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-after", type=int, default=512)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--log-dir", type=str, default=os.getenv('RAYNET_PATH') + "/_models_training" + "/Orca-TD3")
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--bootstrap-on-truncation", dest="bootstrap_on_truncation", action="store_true")
    parser.add_argument("--no-bootstrap-on-truncation", dest="bootstrap_on_truncation", action="store_false")
    parser.set_defaults(bootstrap_on_truncation=True)
    parser.add_argument("--quiet-env", action="store_true")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=2_553_600)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--stddev", type=float, default=0.2)
    parser.add_argument("--noise-type", type=int, default=3)
    parser.add_argument("--noise-exp", type=int, default=50_000)
    parser.add_argument("--loss-type", choices=["MSE", "HUBER"], default="MSE")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    training_config = {
        "iniPath": os.getenv('RAYNET_PATH') + "/simlibs/Orca/src/training/OrcaTraining.ini",
        "bottleneck_bw_range": (5, 20),
        "minimum_rtt_range": (5, 100),
        "bottleneck_buffer_range": (25_000, 4_000_000),
        "max_steps_range": (args.max_episode_steps, args.max_episode_steps),
    }
    env_config = {
        "iniPath": training_config["iniPath"],
        "config_section": "General",
        "stacking": 10,
    }

    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    envs = [OrcaEnv(env_config, verbose=not args.quiet_env) for _ in range(args.num_simulations)]
    env_rngs = [np.random.default_rng(args.seed + worker_id) for worker_id in range(args.num_simulations)]

    # Generate a randomized INI before resetting each persistent environment.
    def reset_training_env(worker_id):
        ini_path = prepare_training_ini(training_config, env_rngs[worker_id], worker_id)
        try:
            return envs[worker_id].reset(ini_path)
        finally:
            ini_path.unlink(missing_ok=True)

    print(
        f"Running {args.num_simulations} parallel simulation(s) "
        f"for {args.total_steps} aggregate steps.",
        flush=True,
    )

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        summary_writer = tf.summary.FileWriter(str(log_dir))
        agent = build_agent(envs[0], args, summary_writer)
        agent.build_learn()
        agent.create_tf_summary()
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            agent.assign_sess(sess)
            sess.run(tf.global_variables_initializer())

            if args.restore:
                saver.restore(sess, args.restore)
            else:
                agent.init_target()

            with ThreadPoolExecutor(max_workers=args.num_simulations) as executor:
                reset_results = list(executor.map(reset_training_env, range(args.num_simulations)))
                states = [result["Orca"] for result in reset_results]
                decision_states = states.copy()
                actions = [0.0] * args.num_simulations
                episode_returns = [0.0] * args.num_simulations
                episode_steps = [0] * args.num_simulations
                completed_episodes = 0
                step = 0
                start_time = time.time()

                try:
                    while step < args.total_steps:
                        active_count = min(args.num_simulations, args.total_steps - step)
                        active_workers = list(range(active_count))

                        for worker_id in active_workers:
                            decision_state = decision_states[worker_id]
                            if decision_state is not None:
                                actions[worker_id] = action_scalar(
                                    agent.get_action(decision_state, use_noise=True)
                                )

                        futures = {
                            worker_id: executor.submit(
                                envs[worker_id].step,
                                {"Orca": actions[worker_id]} if decision_states[worker_id] is not None else {},
                            )
                            for worker_id in active_workers
                        }
                        reset_workers = []

                        for worker_id in active_workers:
                            env = envs[worker_id]
                            state = states[worker_id]
                            action = actions[worker_id]
                            result = futures[worker_id].result()
                            next_state = result.states.get("Orca", state)
                            reward = result.rewards.get("Orca", 0.0)
                            terminal = terminal_for_training(
                                result,
                                "Orca",
                                bootstrap_on_truncation=args.bootstrap_on_truncation,
                            )
                            episode_done = result.episode_done
                            next_observation_valid = "Orca" in result.states
                            step += 1

                            if next_observation_valid:
                                agent.store_experience(
                                    state,
                                    np.array([action], dtype=np.float32),
                                    np.array([reward], dtype=np.float32),
                                    next_state,
                                    np.array([float(terminal)], dtype=np.float32),
                                )

                            episode_returns[worker_id] += reward
                            episode_steps[worker_id] += 1
                            if next_observation_valid:
                                states[worker_id] = next_state
                                decision_states[worker_id] = next_state
                            else:
                                decision_states[worker_id] = None
                            latest_obs = next_state[-env.raw_obs_dim:]

                            write_scalar(summary_writer, "Rollout/step_reward", reward, step)
                            write_scalar(summary_writer, "Rollout/learner_action", action, step)
                            write_scalar(summary_writer, "Rollout/sim_action", pow(4.0, action), step)
                            write_scalar(
                                summary_writer,
                                "Rollout/observation_valid",
                                float(next_observation_valid),
                                step,
                            )
                            write_scalar(
                                summary_writer,
                                "Rollout/worker_id",
                                worker_id,
                                step,
                            )
                            write_scalar(
                                summary_writer,
                                "Rollout/replay_size",
                                agent.rp_buffer.length_buf,
                                step,
                            )
                            write_scalar(summary_writer, "Rollout/throughput", latest_obs[0], step)
                            write_scalar(summary_writer, "Rollout/pacing_rate", latest_obs[1], step)
                            write_scalar(summary_writer, "Rollout/loss_rate", latest_obs[2], step)
                            write_scalar(summary_writer, "Rollout/delay_metric", latest_obs[6], step)
                            write_scalar(
                                summary_writer,
                                "Performance/environment_steps_per_second",
                                step / max(time.time() - start_time, 1e-6),
                                step,
                            )

                            can_train = agent.rp_buffer.length_buf >= args.train_after
                            if can_train and step % args.train_every == 0:
                                for _ in range(args.updates_per_train):
                                    agent.train_step()
                                    agent.target_update()

                            hit_template_limit = (
                                episode_steps[worker_id] >= args.max_episode_steps
                            )
                            if episode_done or hit_template_limit:
                                print(
                                    "episode={} worker={} step={} episode_steps={} "
                                    "return={:.4f} buffer={} elapsed={:.1f}s".format(
                                        completed_episodes,
                                        worker_id,
                                        step,
                                        episode_steps[worker_id],
                                        episode_returns[worker_id],
                                        agent.rp_buffer.length_buf,
                                        time.time() - start_time,
                                    )
                                )
                                write_scalar(
                                    summary_writer,
                                    "Episode/return",
                                    episode_returns[worker_id],
                                    completed_episodes,
                                )
                                write_scalar(
                                    summary_writer,
                                    "Episode/length",
                                    episode_steps[worker_id],
                                    completed_episodes,
                                )
                                write_scalar(
                                    summary_writer,
                                    "Episode/worker_id",
                                    worker_id,
                                    completed_episodes,
                                )
                                summary_writer.flush()
                                completed_episodes += 1
                                episode_returns[worker_id] = 0.0
                                episode_steps[worker_id] = 0
                                actions[worker_id] = 0.0
                                if step < args.total_steps:
                                    reset_workers.append(worker_id)

                            if (
                                args.checkpoint_every > 0
                                and step % args.checkpoint_every == 0
                            ):
                                path = saver.save(
                                    sess,
                                    str(checkpoint_dir / "model.ckpt"),
                                    global_step=step,
                                )
                                print(f"Saved checkpoint: {path}")

                        if reset_workers:
                            reset_futures = {
                                worker_id: executor.submit(reset_training_env, worker_id)
                                for worker_id in reset_workers
                            }
                            for worker_id, future in reset_futures.items():
                                reset_states = future.result()
                                states[worker_id] = reset_states["Orca"]
                                decision_states[worker_id] = reset_states["Orca"]
                            if agent.actor_noise is not None:
                                agent.actor_noise.reset()

                finally:
                    final_path = saver.save(
                        sess,
                        str(checkpoint_dir / "model.ckpt"),
                        global_step=args.total_steps,
                    )
                    list(executor.map(lambda env: env.close(), envs))
                    summary_writer.close()
                    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
