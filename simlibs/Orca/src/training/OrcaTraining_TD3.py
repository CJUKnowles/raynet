"""
Train the OMNeT++ Orca implementation with the original Orca TD3 learner.

This is intentionally a small bridge rather than a rewrite of the original
learner. The vendored TensorFlow TD3 implementation stays separate from the
Orca environment wrapper and this training loop.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import GPUtil
import numpy as np
import psutil

ORCA_SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCA_SRC_DIR))

from OrcaEnv import OrcaEnv, action_scalar  # noqa: E402
from training.learner import Agent, tf  # noqa: E402


AGENT_ID = "Orca"
STACKING = 10
TRAINING_INI = Path(os.environ["RAYNET_PATH"]) / "simlibs/Orca/src/training/OrcaTraining.ini"


# MARK: Logging -------------------------------------------------------------------------------------------------
def write_scalars(summary_writer, values, step):
    """Write multiple scalar values to one TensorBoard step."""
    summary = tf.Summary()
    for tag, value in values.items():
        value = np.asarray(value).reshape(-1)[0]
        summary.value.add(tag=tag, simple_value=float(value))
    summary_writer.add_summary(summary, global_step=step)


def get_utilization_percentages():
    """Return current system CPU and maximum visible GPU utilization percentages."""
    cpu_utilization = psutil.cpu_percent(interval=None)
    try:
        gpus = GPUtil.getGPUs()
    except Exception:
        gpus = []
    gpu_utilization = max((gpu.load * 100.0 for gpu in gpus), default=0.0)
    return cpu_utilization, gpu_utilization


def log_rollout_step(summary_writer, agent, env, worker_id, step, reward, action, observation_valid, state):
    """Log one completed environment step."""
    latest_obs = state[-env.raw_obs_dim:]
    write_scalars(
        summary_writer,
        {
            "Rollout/step_reward": reward,
            "Rollout/learner_action": action,
            "Rollout/sim_action": pow(4.0, action),
            "Rollout/observation_valid": observation_valid,
            "Rollout/worker_id": worker_id,
            "Rollout/replay_size": agent.rp_buffer.length_buf,
            "Rollout/throughput": latest_obs[0],
            "Rollout/pacing_rate": latest_obs[1],
            "Rollout/loss_rate": latest_obs[2],
            "Rollout/delay_metric": latest_obs[6],
        },
        step,
    )


def log_training_performance(summary_writer, step, start_time, timings):
    """Log utilization, throughput, and phase timings for one simulation wave."""
    cpu_utilization, gpu_utilization = get_utilization_percentages()
    values = {
        "Performance/environment_steps_per_second": step / max(time.time() - start_time, 1e-6),
        "Performance/cpu_utilization_percent": cpu_utilization,
        "Performance/gpu_utilization_percent": gpu_utilization,
    }
    values.update({f"Performance/{name}_seconds": value for name, value in timings.items()})
    write_scalars(summary_writer, values, step)


def log_completed_episode(summary_writer, agent, episode, worker_id, step, episode_steps, episode_return, start_time):
    """Log one completed training episode."""
    print(
        f"episode={episode} worker={worker_id} step={step} episode_steps={episode_steps} "
        f"return={episode_return:.4f} buffer={agent.rp_buffer.length_buf} elapsed={time.time() - start_time:.1f}s"
    )
    write_scalars(
        summary_writer,
        {
            "Episode/return": episode_return,
            "Episode/length": episode_steps,
            "Episode/worker_id": worker_id,
        },
        episode,
    )
    summary_writer.flush()

# MARK: Utilities -----------------------------------------------------------------------------------------------
def prepare_training_ini(args, rng, worker_id):
    """Create one randomized INI variant for a training episode."""
    # Sample the randomized bandwidth and RTT parameters.
    bw = round(rng.uniform(*args.bottleneck_bw_range))
    base_rtt = round(rng.uniform(*args.minimum_rtt_range), 2)

    # Calculate the buffer in bits from either a BDP multiplier or a raw byte size.
    if args.buffers_use_bdp_ranges:
        buffer_size_bits = round(bw * base_rtt * rng.uniform(*args.buffer_bdp_range) * 1000)
    else:
        buffer_size_bits = round(rng.uniform(*args.buffer_size_range) * 8)

    # Create a unique INI variant for this training worker.
    original_ini_file = args.training_ini
    ini_variants_dir = original_ini_file.parent / "ini_variants"
    ini_variants_dir.mkdir(parents=True, exist_ok=True)
    worker_ini_file = ini_variants_dir / f"{original_ini_file.name}.worker{os.getpid()}-{worker_id}"

    # Replace the training placeholders and write the generated INI.
    ini_string = original_ini_file.read_text(encoding="utf-8")
    ini_string = ini_string.replace("!HOME!", os.getenv("HOME", str(Path.home())))
    ini_string = ini_string.replace("!BW!", f"{bw}Mbps")
    ini_string = ini_string.replace("!DELAY!", f"{base_rtt / 2.0}ms")
    ini_string = ini_string.replace("!QSIZE!", f"{buffer_size_bits}b")
    worker_ini_file.write_text(ini_string, encoding="utf-8")
    return worker_ini_file


def reset_training_env(env, args, rng, worker_id):
    """Reset one training environment with a randomized temporary INI."""
    ini_path = prepare_training_ini(args, rng, worker_id)
    try:
        return env.reset(ini_path)
    finally:
        ini_path.unlink(missing_ok=True)


def terminal_for_training(step, agent_id, bootstrap_on_truncation=True):
    """Return whether a step should be terminal in the replay buffer."""
    terminated = bool(step.terminateds.get(agent_id, step.terminateds.get("__all__", False)))
    return terminated or (step.truncated and not bootstrap_on_truncation)


def save_checkpoint(saver, session, checkpoint_dir, step, final=False):
    """Save and report one learner checkpoint."""
    path = saver.save(session, str(checkpoint_dir / "model.ckpt"), global_step=step)
    print(f"Saved {'final ' if final else ''}checkpoint: {path}")


def build_session_config():
    """Configure TensorFlow to grow GPU memory usage as needed."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


# MARK: Core Functionality --------------------------------------------------------------------------------------
def build_agent(env, args, summary_writer):
    """Create the original Orca TD3 learner. (a single shared learning agent across all environments)"""
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


# MARK: Configuration -------------------------------------------------------------------------------------------
def parse_args():
    """Parse the training configuration."""
    parser = argparse.ArgumentParser(description="Original Orca TD3 learner for OMNeT++")

    # Define network parameter ranges
    parser.add_argument("--bottleneck-bw-range", type=float, nargs=2, default=(5, 20), metavar=("MIN_MBPS", "MAX_MBPS"))   # 5-20, 48-48
    parser.add_argument("--minimum-rtt-range", type=float, nargs=2, default=(5, 100), metavar=("MIN_MS", "MAX_MS"))         # 5-100, 20-20
    parser.add_argument("--buffer-bdp-range", type=float, nargs=2, default=(.2, 10), metavar=("MIN_BDP", "MAX_BDP"))          # .2-10, 2-2
    parser.add_argument("--buffer-size-range", type=float, nargs=2, default=(240_000, 240_000), metavar=("MIN_BUFFER_BYTES", "MAX_BUFFER_BYTES"))
    parser.add_argument("--buffers-use-bdp-ranges", action=argparse.BooleanOptionalAction, default=True, help="Derive buffer sizes from BDP multipliers instead of sampling raw byte sizes")


    # Define general training and output arguments.
    parser.add_argument("--log-dir", type=str, default=str(Path(os.environ["RAYNET_PATH"]) / "_models_training/Orca-TD3-GPU-v3-expanded"))
    parser.add_argument("--num-simulations", type=int, default=16, help="Number of OMNeT++ simulations to run in parallel")
    parser.add_argument("--checkpoint-every-seconds", type=float, default=15.0, help="Seconds between periodic learner checkpoints; zero disables them")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Aggregate steps between rollout-step logs; zero disables them")
    parser.add_argument("--restore", type=str, default="", help="TensorFlow checkpoint prefix to restore before training")
    parser.add_argument("--quiet-env", action="store_true", help="Suppress OrcaEnv status messages")
    parser.add_argument("--max-episode-steps", type=int, default=64)

    # Define simulation orchestration
    parser.add_argument("--training-ini", type=Path, default=TRAINING_INI)
    parser.add_argument("--seed", type=int, default=91456211)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--train-after", type=int, default=512)
    parser.add_argument("--train-every", type=int, default=1) # 1 in original Orca, but that is distributed and real-time. 16 increases step throughput but reduces learning rate.
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--bootstrap-on-truncation", dest="bootstrap_on_truncation", action="store_true")
    parser.add_argument("--no-bootstrap-on-truncation", dest="bootstrap_on_truncation", action="store_false")
    parser.set_defaults(bootstrap_on_truncation=True)

    # Define original Orca TD3 learner hyperparameters.
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


def validate_args(args):
    """Validate training arguments before constructing simulations."""
    # Validate scalar counts and intervals.
    if args.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be at least 1")
    if args.log_every_steps < 0:
        raise ValueError("--log-every-steps must not be negative")

    # Validate randomized training ranges.
    for name in ("bottleneck_bw_range", "minimum_rtt_range", "buffer_bdp_range", "buffer_size_range"):
        lower, upper = getattr(args, name)
        if lower > upper:
            raise ValueError(f"--{name.replace('_', '-')} minimum must not exceed its maximum")


# MARK: Training Loop -----------------------------------------------------------------------------------
def main():
    """Train the original Orca TD3 learner across parallel simulations."""
    # Parse arguments and seed the learner.
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Construct the shared environment configuration.
    env_config = {
        "iniPath": str(args.training_ini),
        "config_section": "General",
        "stacking": STACKING,
    }

    # Create output directories and persistent environment wrappers.
    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    envs = [OrcaEnv(env_config, verbose=not args.quiet_env) for _ in range(args.num_simulations)]
    env_rngs = [np.random.default_rng(args.seed + worker_id) for worker_id in range(args.num_simulations)]

    print(f"Running {args.num_simulations} parallel simulation(s) for {args.total_steps} aggregate steps.", flush=True)

    # Build and initialize the learner graph.
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        summary_writer = tf.summary.FileWriter(str(log_dir))
        agent = build_agent(envs[0], args, summary_writer)
        agent.build_learn()
        agent.create_tf_summary()
        saver = tf.train.Saver(max_to_keep=5) # Maximum number of checkpoints to keep on disk

        with tf.Session(config=build_session_config()) as sess:
            agent.assign_sess(sess)
            sess.run(tf.global_variables_initializer())

            if args.restore:
                saver.restore(sess, args.restore)
            else:
                agent.init_target()

            with ThreadPoolExecutor(max_workers=args.num_simulations) as executor:
                # Reset every environment and initialize persistent rollout state.
                reset_futures = [
                    executor.submit(reset_training_env, envs[worker_id], args, env_rngs[worker_id], worker_id)
                    for worker_id in range(args.num_simulations)
                ]
                reset_results = [future.result() for future in reset_futures]
                states = [result[AGENT_ID] for result in reset_results]
                decision_states = states.copy()
                actions = [0.0] * args.num_simulations
                episode_returns = [0.0] * args.num_simulations
                episode_steps = [0] * args.num_simulations
                completed_episodes = 0
                step = 0
                start_time = time.time()
                last_checkpoint_time = time.monotonic()
                psutil.cpu_percent(interval=None)

                try:
                    while step < args.total_steps:
                        # Measure each synchronized wave through action selection, simulation, learning, and logging.
                        wave_start_time = time.perf_counter()
                        wave_start_step = step
                        action_selection_seconds = 0.0
                        simulation_wait_seconds = 0.0
                        learner_update_seconds = 0.0
                        logging_seconds = 0.0
                        reset_seconds = 0.0

                        # Select new actions only for environments with valid observations.
                        phase_start_time = time.perf_counter()
                        active_count = min(args.num_simulations, args.total_steps - step)
                        active_workers = list(range(active_count))
                        for worker_id in active_workers:
                            decision_state = decision_states[worker_id]
                            if decision_state is not None:
                                actions[worker_id] = action_scalar(agent.get_action(decision_state, use_noise=True))
                        action_selection_seconds = time.perf_counter() - phase_start_time

                        # Advance all active simulations in parallel.
                        futures = {
                            worker_id: executor.submit(
                                envs[worker_id].step,
                                {AGENT_ID: actions[worker_id]} if decision_states[worker_id] is not None else {},
                            )
                            for worker_id in active_workers
                        }
                        reset_workers = []

                        # Process each completed simulation step.
                        for worker_id in active_workers:
                            env = envs[worker_id]
                            state = states[worker_id]
                            action = actions[worker_id]
                            phase_start_time = time.perf_counter()
                            result = futures[worker_id].result()
                            simulation_wait_seconds += time.perf_counter() - phase_start_time
                            next_state = result.states.get(AGENT_ID, state)
                            reward = result.rewards.get(AGENT_ID, 0.0)
                            terminal = terminal_for_training(result, AGENT_ID, bootstrap_on_truncation=args.bootstrap_on_truncation)
                            episode_done = result.episode_done
                            next_observation_valid = AGENT_ID in result.states

                            # Treat the Python valid-step limit as a truncation for replay bootstrapping.
                            hit_template_limit = next_observation_valid and episode_steps[worker_id] + 1 >= args.max_episode_steps
                            if hit_template_limit and not args.bootstrap_on_truncation:
                                terminal = True

                            # Store valid transitions and advance learner-facing step counters.
                            if next_observation_valid:
                                agent.store_experience(
                                    state,
                                    np.array([action], dtype=np.float32),
                                    np.array([reward], dtype=np.float32),
                                    next_state,
                                    np.array([float(terminal)], dtype=np.float32),
                                )
                                step += 1
                                episode_returns[worker_id] += reward
                                episode_steps[worker_id] += 1
                                states[worker_id] = next_state
                                decision_states[worker_id] = next_state
                            else:
                                decision_states[worker_id] = None

                            # Train and log only after valid learner observations.
                            can_train = agent.rp_buffer.length_buf >= args.train_after
                            log_step = next_observation_valid and args.log_every_steps > 0 and step % args.log_every_steps == 0
                            if next_observation_valid and can_train and step % args.train_every == 0:
                                phase_start_time = time.perf_counter()
                                for _ in range(args.updates_per_train):
                                    agent.train_step()
                                    agent.target_update()
                                learner_update_seconds += time.perf_counter() - phase_start_time

                            # Schedule completed environments for reset.
                            completed_episode = None
                            if episode_done or hit_template_limit:
                                completed_episode = (
                                    completed_episodes,
                                    worker_id,
                                    step,
                                    episode_steps[worker_id],
                                    episode_returns[worker_id],
                                    start_time,
                                )
                                completed_episodes += 1
                                episode_returns[worker_id] = 0.0
                                episode_steps[worker_id] = 0
                                actions[worker_id] = 0.0
                                if step < args.total_steps:
                                    reset_workers.append(worker_id)

                            # Log the completed step after all core processing.
                            phase_start_time = time.perf_counter()
                            if log_step:
                                log_rollout_step(summary_writer, agent, env, worker_id, step, reward, action, next_observation_valid, next_state)
                            if completed_episode is not None:
                                log_completed_episode(summary_writer, agent, *completed_episode)
                            logging_seconds += time.perf_counter() - phase_start_time

                            # Save periodic checkpoints after the completed step.
                            current_checkpoint_time = time.monotonic()
                            if args.checkpoint_every_seconds > 0 and current_checkpoint_time - last_checkpoint_time >= args.checkpoint_every_seconds:
                                save_checkpoint(saver, sess, checkpoint_dir, step)
                                last_checkpoint_time = current_checkpoint_time

                        # Reset environments after all current results are processed.
                        if reset_workers:
                            phase_start_time = time.perf_counter()
                            reset_futures = {
                                worker_id: executor.submit(
                                    reset_training_env,
                                    envs[worker_id],
                                    args,
                                    env_rngs[worker_id],
                                    worker_id,
                                )
                                for worker_id in reset_workers
                            }
                            for worker_id, future in reset_futures.items():
                                reset_states = future.result()
                                states[worker_id] = reset_states[AGENT_ID]
                                decision_states[worker_id] = reset_states[AGENT_ID]
                            if agent.actor_noise is not None:
                                agent.actor_noise.reset()
                            reset_seconds = time.perf_counter() - phase_start_time

                        # Log synchronized-wave performance after all work is complete.
                        wave_seconds = time.perf_counter() - wave_start_time
                        crossed_logging_interval = args.log_every_steps > 0 and step // args.log_every_steps > wave_start_step // args.log_every_steps
                        if crossed_logging_interval:
                            timings = {
                                "wave": wave_seconds,
                                "action_selection": action_selection_seconds,
                                "simulation_wait": simulation_wait_seconds,
                                "learner_update": learner_update_seconds,
                                "logging": logging_seconds,
                                "reset": reset_seconds,
                            }
                            log_training_performance(summary_writer, step, start_time, timings)

                finally:
                    # Save the final learner state and close every environment.
                    save_checkpoint(saver, sess, checkpoint_dir, args.total_steps, final=True)
                    list(executor.map(lambda env: env.close(), envs))
                    summary_writer.close()


if __name__ == "__main__":
    main()
