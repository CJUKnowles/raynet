"""
Train the OMNeT++ Orca implementation with the original Orca TD3 learner.

This is intentionally a small bridge rather than a rewrite of the original
learner. The TensorFlow TD3 implementation lives in ~/orca/rl-module/agent.py;
this file supplies the simulator rollout loop that Ray/RLlib used to hide.
"""

import argparse
import math
import multiprocessing
import os
import random
import sys
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The original Orca learner is small enough that CPU execution is the least
# surprising default. TensorFlow 2.20 can otherwise select GPU and fail during
# XLA/kernel generation if the local CUDA libdevice files are not installed.
if os.getenv("ORCA_TD3_USE_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _unique_layer_name(base_name):
    """Return a tf.layers-style unnamed-layer name within the current scope."""
    graph = tf.get_default_graph()
    counters = getattr(graph, "_orca_layer_name_counters", None)
    if counters is None:
        counters = {}
        graph._orca_layer_name_counters = counters

    key = (tf.get_variable_scope().name, base_name)
    index = counters.get(key, 0)
    counters[key] = index + 1
    return base_name if index == 0 else f"{base_name}_{index}"


def _dense(inputs, units, activation=None, name=None):
    """Small tf.layers.dense replacement for TensorFlow 2 + Keras 3."""
    input_dim = inputs.shape.as_list()[-1]
    if input_dim is None:
        raise ValueError("Dense input dimension must be statically known")

    with tf.variable_scope(name or "dense", reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(
            "kernel",
            shape=[input_dim, units],
            initializer=tf.glorot_uniform_initializer(),
        )
        bias = tf.get_variable(
            "bias",
            shape=[units],
            initializer=tf.zeros_initializer(),
        )
        output = tf.matmul(inputs, kernel) + bias
        if activation is not None:
            output = activation(output)
        return output


def _batch_normalization(inputs, training=False, scale=True, name=None):
    """Small tf.layers.batch_normalization replacement for this learner."""
    dim = inputs.shape.as_list()[-1]
    if dim is None:
        raise ValueError("Batch normalization input dimension must be statically known")

    # tf.layers assigned a fresh name to every unnamed layer. Reusing the same
    # default scope here would incorrectly make Orca's two actor BN layers
    # share parameters and would not restore batch_normalization_1 checkpoints.
    layer_name = name or _unique_layer_name("batch_normalization")
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable("beta", shape=[dim], initializer=tf.zeros_initializer())
        gamma = None
        if scale:
            gamma = tf.get_variable("gamma", shape=[dim], initializer=tf.ones_initializer())
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=[dim],
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=[dim],
            initializer=tf.ones_initializer(),
            trainable=False,
        )

        mean, variance = tf.nn.moments(inputs, axes=[0])
        mean_update = tf.assign(moving_mean, moving_mean * 0.99 + mean * 0.01)
        variance_update = tf.assign(
            moving_variance,
            moving_variance * 0.99 + variance * 0.01,
        )
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)

        train_output = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-3)
        infer_output = tf.nn.batch_normalization(
            inputs,
            moving_mean,
            moving_variance,
            beta,
            gamma,
            1e-3,
        )

        if isinstance(training, bool):
            return train_output if training else infer_output
        return tf.cond(training, lambda: train_output, lambda: infer_output)


tf.layers = types.SimpleNamespace(
    dense=_dense,
    batch_normalization=_batch_normalization,
)
sys.modules["tensorflow"] = tf

sys.path.insert(0, os.path.join(os.getenv('RAYNET_PATH'), "build"))

from learner import Agent  # noqa: E402
from OrcaEpisodeWorker import run_episode  # noqa: E402


def write_scalar(summary_writer, tag, value, step):
    value = np.asarray(value).reshape(-1)[0]
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=float(value))
    summary_writer.add_summary(summary, global_step=step)


def action_scalar(action):
    """Normalize the original learner's nested action arrays to one float."""
    return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])


class OmnetOrcaRolloutEnv:
    """Minimal environment adapter for the original Orca learner.

    The learner acts in [-1, 1]. The original emulation bridge converted that
    to a multiplicative pacing-rate command with 4 ** action. Your simulator
    expects a log2 cwnd multiplier action, so 2 * learner_action preserves the
    same multiplier range: 2 ** (2a) == 4 ** a.
    """

    agent_name = "Orca"
    raw_obs_dim = 7
    ack_count_index = 3

    def __init__(self, env_config, bootstrap_on_truncation=True, verbose=True):
        self.mp_context = multiprocessing.get_context("spawn")
        self.worker_process = None
        self.worker_connection = None
        self.env_config = env_config
        self.worker_id = int(env_config.get("worker_id", 0))
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.verbose = verbose
        self.stacking = int(env_config["stacking"])
        self.obs_history = deque(
            np.zeros(self.stacking * self.raw_obs_dim, dtype=np.float32),
            maxlen=self.stacking * self.raw_obs_dim,
        )
        self.rng = np.random.default_rng(os.getpid() + self.worker_id)
        self.worker_ini_file = None
        self.closed = True

    @property
    def state_dim(self):
        return self.stacking * self.raw_obs_dim

    @property
    def action_dim(self):
        return 1

    def reset(self):
        self._log("reset: preparing new OMNeT++ run")
        self._cleanup_after_previous_episode()
        self.obs_history.clear()
        self.obs_history.extend(np.zeros(self.state_dim, dtype=np.float32))

        worker_ini_file = self._write_worker_ini()
        self._log(f"reset: spawning episode process for {worker_ini_file}")

        print("--------------------------------------------------------------------")
        print("Initializing new OMNeT++ run with ini file:\n", worker_ini_file, flush=True)
        print("--------------------------------------------------------------------\n")
        self._spawn_episode_process(worker_ini_file)
        message_type, reset_obs = self._receive_worker_message()
        if message_type != "reset":
            raise RuntimeError(f"Expected reset message from episode worker, got {message_type}")

        self._log(f"reset: episode worker returned keys={list(reset_obs.keys())}")
        if self.agent_name not in reset_obs:
            self.closed = True
            raise RuntimeError(
                "runner.reset() did not return an Orca observation. "
                f"Returned keys were {list(reset_obs.keys())}. "
                "If this contains SIMULATION_END, the simulation ended before the first RL observation."
            )

        obs = reset_obs[self.agent_name]
        for _ in range(self.stacking):
            self.obs_history.extend(obs)
        self._log(f"reset: initial stacked state shape={self._state().shape}")
        return self._state()

    def step(self, learner_action):
        action = action_scalar(learner_action)
        print(f"ACTION: {action}")
        action = float(np.clip(action, -1.0, 1.0))
        self._log(f"step: learner_action={action:.6f}, sim_action={action:.6f}")
        self.worker_connection.send(("step", {self.agent_name: action}))
        message_type, result = self._receive_worker_message()
        if message_type != "step":
            raise RuntimeError(f"Expected step message from episode worker, got {message_type}")
        obs, rewards, terminateds, info = result
        self._log(
            "step: returned "
            f"obs_keys={list(obs.keys())}, reward_keys={list(rewards.keys())}, "
            f"terminated_keys={list(terminateds.keys())}, info={info}"
        )
        if self.agent_name not in obs:
            if info.get("simDone", False):
                self.closed = True
                self._join_episode_process()
                return self._state(), 0.0, False, True
            raise RuntimeError(
                f"runner.step() did not return an Orca observation. Returned keys were {list(obs.keys())}."
            )

        self.obs_history.extend(obs[self.agent_name])

        reward = float(rewards[self.agent_name])
        if math.isnan(reward):
            print("Warning: NaN reward returned; replacing with 0.0")
            reward = 0.0

        terminated = bool(terminateds[self.agent_name])
        truncated = bool(info.get("simDone", False))
        terminal_for_training = terminated or (truncated and not self.bootstrap_on_truncation)

        if terminated or truncated:
            self.closed = True
            self._join_episode_process()

        return self._state(), reward, terminal_for_training, terminated or truncated

    def close(self):
        print("! Closing environment and cleaning up runner !")
        self._cleanup_after_previous_episode()

    def _state(self):
        return np.asarray(list(self.obs_history), dtype=np.float32)

    def latest_observation_has_acks(self, state):
        latest_obs = np.asarray(state)[-self.raw_obs_dim:]
        return bool(latest_obs[self.ack_count_index] > 0.0)

    def _cleanup_after_previous_episode(self):
        if self.worker_process is None:
            return

        if not self.closed:
            try:
                self.worker_connection.send(("close", None))
                message_type, _ = self._receive_worker_message()
                if message_type != "closed":
                    print(f"Warning: expected closed message from episode worker, got {message_type}")
            except Exception as exc:
                print(f"Warning: episode worker cleanup failed: {exc}")

        self._join_episode_process()
        self._log("reset: discarded previous episode process")
        self.closed = True

    def _spawn_episode_process(self, worker_ini_file):
        parent_connection, child_connection = self.mp_context.Pipe()
        self.worker_connection = parent_connection
        self.worker_process = self.mp_context.Process(
            target=run_episode,
            args=(child_connection, str(worker_ini_file), "General", self.agent_name),
            daemon=True,
        )
        self.worker_process.start()
        child_connection.close()
        self.closed = False
        self._log(f"reset: spawned episode process pid={self.worker_process.pid}")

    def _receive_worker_message(self):
        try:
            message_type, payload = self.worker_connection.recv()
        except EOFError as exc:
            exit_code = self.worker_process.exitcode
            raise RuntimeError(
                f"OMNeT++ episode process exited unexpectedly with code {exit_code}"
            ) from exc

        if message_type == "error":
            raise RuntimeError(f"OMNeT++ episode process failed:\n{payload}")
        return message_type, payload

    def _join_episode_process(self):
        if self.worker_process is None:
            return

        self.worker_process.join(timeout=10)
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
        if self.worker_connection is not None:
            self.worker_connection.close()
        self.worker_process = None
        self.worker_connection = None

    def _write_worker_ini(self):
        bottleneck_bw_range = self.env_config["bottleneck_bw_range"]
        base_rtt_range = self.env_config["minimum_rtt_range"]
        max_steps_range = self.env_config["max_steps_range"]

        bw = round(self.rng.uniform(bottleneck_bw_range[0], bottleneck_bw_range[1]))
        base_rtt = round(self.rng.uniform(base_rtt_range[0], base_rtt_range[1]), 2)
        buffer_size = round(bw * base_rtt * self.rng.uniform(0.2, 4.0) * 1000)
        max_steps = round(self.rng.uniform(max_steps_range[0], max_steps_range[1]))

        original_ini_file = Path(self.env_config["iniPath"])
        ini_variants_dir = original_ini_file.parent / "ini_variants"
        ini_variants_dir.mkdir(parents=True, exist_ok=True)
        worker_ini_file = (
            ini_variants_dir
            / f"{original_ini_file.name}.worker{os.getpid()}-{self.worker_id}"
        )

        ini_string = original_ini_file.read_text(encoding="utf-8")
        ini_string = ini_string.replace("HOME", os.getenv("HOME", str(Path.home())))
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BW", f"{bw}Mbps")
        ini_string = ini_string.replace("ORCA_BASE_RTT", f"{base_rtt / 2.0}ms")
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BUFFER_SIZE", f"{buffer_size}b")
        ini_string = ini_string.replace("MAX_RL_STEPS", f"{max_steps}")
        worker_ini_file.write_text(ini_string, encoding="utf-8")
        self.worker_ini_file = worker_ini_file
        self._log(
            "reset: wrote ini variant "
            f"bw={bw}Mbps base_rtt={base_rtt}ms buffer={buffer_size}b max_steps={max_steps}"
        )
        return worker_ini_file

    def _log(self, message):
        if self.verbose:
            print(f"[td3-env] {message}", flush=True)


def build_agent(env, args, summary_writer):
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

    base_env_config = {
        "iniPath": os.getenv('RAYNET_PATH') + "/simlibs/Orca/src/training/OrcaTraining.ini",
        "bottleneck_bw_range": (5, 20),
        "minimum_rtt_range": (5, 100),
        "bottleneck_buffer_range": (25_000, 4_000_000),
        "max_steps_range": (args.max_episode_steps, args.max_episode_steps),
        "stacking": 10,
    }

    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    envs = []
    for worker_id in range(args.num_simulations):
        env_config = dict(base_env_config, worker_id=worker_id)
        envs.append(
            OmnetOrcaRolloutEnv(
                env_config,
                bootstrap_on_truncation=args.bootstrap_on_truncation,
                verbose=not args.quiet_env,
            )
        )
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
                states = list(executor.map(lambda env: env.reset(), envs))
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
                        observations_have_acks = {}

                        for worker_id in active_workers:
                            env = envs[worker_id]
                            state = states[worker_id]
                            has_acks = env.latest_observation_has_acks(state)
                            observations_have_acks[worker_id] = has_acks
                            if has_acks:
                                actions[worker_id] = action_scalar(
                                    agent.get_action(state, use_noise=True)
                                )

                        futures = {
                            worker_id: executor.submit(
                                envs[worker_id].step,
                                actions[worker_id],
                            )
                            for worker_id in active_workers
                        }
                        reset_workers = []

                        for worker_id in active_workers:
                            env = envs[worker_id]
                            state = states[worker_id]
                            action = actions[worker_id]
                            has_acks = observations_have_acks[worker_id]
                            next_state, reward, terminal, episode_done = futures[
                                worker_id
                            ].result()
                            step += 1

                            if has_acks:
                                agent.store_experience(
                                    state,
                                    np.array([action], dtype=np.float32),
                                    np.array([reward], dtype=np.float32),
                                    next_state,
                                    np.array([float(terminal)], dtype=np.float32),
                                )

                            episode_returns[worker_id] += reward
                            episode_steps[worker_id] += 1
                            states[worker_id] = next_state
                            latest_obs = next_state[-env.raw_obs_dim:]

                            write_scalar(summary_writer, "Rollout/step_reward", reward, step)
                            write_scalar(summary_writer, "Rollout/learner_action", action, step)
                            write_scalar(summary_writer, "Rollout/sim_action", pow(4.0, action), step)
                            write_scalar(
                                summary_writer,
                                "Rollout/observation_has_acks",
                                float(has_acks),
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
                                worker_id: executor.submit(envs[worker_id].reset)
                                for worker_id in reset_workers
                            }
                            for worker_id, future in reset_futures.items():
                                states[worker_id] = future.result()
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
