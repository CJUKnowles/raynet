"""
Train RayNet's multi-agent Astraea simulation with the original TensorFlow
Astraea learner.

Each connection is an actor with a local observation history. All connections
share one policy, one replay buffer, and the global reward. As in the original
Astraea implementation, the actor sees only local state while the critic also
receives a compact global state (centralized training, decentralized execution).
"""

import argparse
import math
import multiprocessing
import os
import random
import re
import sys
import time
import types
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if os.getenv("ASTRAEA_TF_USE_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/astraea-matplotlib")

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _unique_layer_name(base_name):
    graph = tf.get_default_graph()
    counters = getattr(graph, "_astraea_layer_name_counters", None)
    if counters is None:
        counters = {}
        graph._astraea_layer_name_counters = counters
    key = (tf.get_variable_scope().name, base_name)
    index = counters.get(key, 0)
    counters[key] = index + 1
    return base_name if index == 0 else f"{base_name}_{index}"


def _dense(inputs, units, activation=None, name=None):
    input_dim = inputs.shape.as_list()[-1]
    if input_dim is None:
        raise ValueError("Dense input dimension must be statically known")
    # The original critic is built twice under AUTO_REUSE and expects its
    # unnamed output layer to be shared by both calls.
    layer_name = name or "dense"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(
            "kernel",
            shape=[input_dim, units],
            initializer=tf.glorot_uniform_initializer(),
        )
        bias = tf.get_variable("bias", shape=[units], initializer=tf.zeros_initializer())
        output = tf.matmul(inputs, kernel) + bias
        return activation(output) if activation is not None else output


def _batch_normalization(inputs, training=False, scale=True, name=None):
    dim = inputs.shape.as_list()[-1]
    if dim is None:
        raise ValueError("Batch normalization input dimension must be statically known")
    layer_name = name or _unique_layer_name("batch_normalization")
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable("beta", shape=[dim], initializer=tf.zeros_initializer())
        gamma = (
            tf.get_variable("gamma", shape=[dim], initializer=tf.ones_initializer())
            if scale
            else None
        )
        moving_mean = tf.get_variable(
            "moving_mean", shape=[dim], initializer=tf.zeros_initializer(), trainable=False
        )
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=[dim],
            initializer=tf.ones_initializer(),
            trainable=False,
        )
        mean, variance = tf.nn.moments(inputs, axes=[0])
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(moving_mean, moving_mean * 0.99 + mean * 0.01),
        )
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(moving_variance, moving_variance * 0.99 + variance * 0.01),
        )
        train_output = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-3)
        infer_output = tf.nn.batch_normalization(
            inputs, moving_mean, moving_variance, beta, gamma, 1e-3
        )
        if isinstance(training, bool):
            return train_output if training else infer_output
        return tf.cond(training, lambda: train_output, lambda: infer_output)


tf.layers = types.SimpleNamespace(dense=_dense, batch_normalization=_batch_normalization)
sys.modules["tensorflow"] = tf

RAYNET_PATH = Path(os.getenv("RAYNET_PATH", "/home/james/raynet"))
ASTRAEA_PYTHON = Path(
    os.getenv("ASTRAEA_PYTHON", "/home/james/astraea-open-source/python")
)
sys.path.insert(0, str(ASTRAEA_PYTHON))

# helpers.utils creates a project-local tmp directory during import. Keep the
# original repository read-only by giving that helper a writable scratch root.
helpers_context = types.ModuleType("helpers.context")
helpers_context.base_dir = "/tmp/astraea-open-source"
helpers_context.log_dir = "/tmp/astraea-open-source/log"
helpers_context.src_dir = str(ASTRAEA_PYTHON)
helpers_context.helper_dir = str(ASTRAEA_PYTHON / "helpers")
sys.modules["helpers.context"] = helpers_context

from agent.agent import Actor, Agent, Critic  # noqa: E402
from AstraeaEpisodeWorker import run_episode  # noqa: E402


def _exact_train_var(self):
    # Modern TensorFlow treats collection scopes as regular expressions, so
    # "actor" also matches "target_actor" in the original implementation.
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=f"^{re.escape(self.name)}/",
    )


Actor.train_var = _exact_train_var
Critic.train_var = _exact_train_var


def action_scalar(action):
    return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])


def write_scalar(writer, tag, value, step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=float(np.asarray(value).reshape(-1)[0]))
    writer.add_summary(summary, global_step=step)


class OmnetAstraeaRolloutEnv:
    """Multi-agent process adapter for the original Astraea learner."""

    raw_obs_dim = 8
    global_state_dim = 12

    def __init__(self, env_config, bootstrap_on_truncation=True, verbose=True):
        self.env_config = env_config
        self.worker_id = int(env_config.get("worker_id", 0))
        self.stacking = int(env_config["stacking"])
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.verbose = verbose
        self.mp_context = multiprocessing.get_context("spawn")
        self.rng = np.random.default_rng(os.getpid() + self.worker_id)
        self.histories = defaultdict(self._new_history)
        self.raw_observations = {}
        self.worker_process = None
        self.worker_connection = None
        self.closed = True
        self.bw_mbps = 0.0
        self.base_rtt_ms = 0.0
        self.buffer_bits = 0

    @property
    def state_dim(self):
        return self.stacking * self.raw_obs_dim

    @property
    def action_dim(self):
        return 1

    def _new_history(self):
        return deque(
            [np.zeros(self.raw_obs_dim, dtype=np.float32) for _ in range(self.stacking)],
            maxlen=self.stacking,
        )

    def reset(self):
        self._cleanup_after_previous_episode()
        self.histories.clear()
        self.raw_observations = {}
        worker_ini = self._write_worker_ini()
        self._spawn_episode_process(worker_ini)
        message_type, observations = self._receive_worker_message()
        if message_type != "reset":
            raise RuntimeError(f"Expected reset message, got {message_type}")
        if not observations:
            raise RuntimeError("runner.reset() returned no Astraea observations")
        self._record_observations(observations, fill_history=True)
        self._log(f"reset agents={sorted(self.raw_observations)}")
        return self.states(), self.global_state()

    def step(self, actions):
        sim_actions = {
            agent_id: float(np.clip(action_scalar(action), -1.0, 1.0))
            for agent_id, action in actions.items()
        }
        self.worker_connection.send(("step", sim_actions))
        message_type, result = self._receive_worker_message()
        if message_type != "step":
            raise RuntimeError(f"Expected step message, got {message_type}")
        observations, rewards, terminateds, info = result
        if observations:
            self._record_observations(observations)

        truncated = bool(info.get("simDone", False))
        episode_done = bool(terminateds.get("__all__", False) or truncated)
        terminals = {}
        for agent_id in self.raw_observations:
            terminated = bool(terminateds.get(agent_id, terminateds.get("__all__", False)))
            terminals[agent_id] = terminated or (
                truncated and not self.bootstrap_on_truncation
            )

        if episode_done:
            self.closed = True
            self._join_episode_process()

        return self.states(), rewards, terminals, episode_done, self.global_state()

    def states(self):
        return {
            agent_id: np.concatenate(self.histories[agent_id]).astype(np.float32)
            for agent_id in self.raw_observations
        }

    def global_state(self):
        """Reconstruct the original learner's 12-value centralized critic input."""
        if not self.raw_observations:
            return np.zeros(self.global_state_dim, dtype=np.float32)

        obs = np.asarray(list(self.raw_observations.values()), dtype=np.float64)
        throughput_bytes = obs[:, 0] * obs[:, 1]
        latency_seconds = obs[:, 2] * obs[:, 3]
        cwnd_bytes = obs[:, 4] * obs[:, 1] * obs[:, 3]
        loss_rate = obs[:, 5] * obs[:, 1]

        # Original Astraea normalized throughput as bits/s, latency as us, and
        # cwnd as packets. RayNet exposes bytes/s, seconds, and bytes.
        throughput = throughput_bytes * 8.0 / 5e7
        latency = latency_seconds * 1e6 / 5e5
        cwnd_packets = cwnd_bytes / 1024.0
        bdp_packets = self.bw_mbps * self.base_rtt_ms / (1024.0 * 8.0)
        return np.asarray(
            [
                np.sum(throughput),
                np.min(throughput),
                np.max(throughput),
                np.mean(latency),
                np.min(cwnd_packets) / 1000.0,
                np.max(cwnd_packets) / 1000.0,
                np.mean(cwnd_packets) / 1000.0,
                np.mean(loss_rate) / 1e6,
                len(obs) / 10.0,
                (self.base_rtt_ms / 2.0) / 500.0,
                bdp_packets / 10.0,
                self.bw_mbps / 500.0,
            ],
            dtype=np.float32,
        )

    def close(self):
        self._cleanup_after_previous_episode()

    def _record_observations(self, observations, fill_history=False):
        for agent_id, observation in observations.items():
            if agent_id == "__all__":
                continue
            raw = np.asarray(observation, dtype=np.float32)
            if raw.shape != (self.raw_obs_dim,):
                raise ValueError(
                    f"{agent_id} returned observation shape {raw.shape}; "
                    f"expected ({self.raw_obs_dim},)"
                )
            self.raw_observations[agent_id] = raw
            if fill_history:
                # Original Astraea begins recurrent state with zero history and
                # shifts the first real observation into the newest slot.
                self.histories[agent_id] = self._new_history()
                self.histories[agent_id].append(raw)
            else:
                self.histories[agent_id].append(raw)

    def _write_worker_ini(self):
        self.bw_mbps = round(
            self.rng.uniform(*self.env_config["bottleneck_bw_range"])
        )
        self.base_rtt_ms = round(
            self.rng.uniform(*self.env_config["minimum_rtt_range"]), 2
        )
        self.buffer_bits = round(
            self.rng.uniform(*self.env_config["bottleneck_buffer_range"])
        )
        max_steps = round(self.rng.uniform(*self.env_config["max_steps_range"]))
        num_flows = round(self.rng.uniform(*self.env_config["num_flows_range"]))

        source = Path(self.env_config["iniPath"])
        variants = source.parent / "ini_variants"
        variants.mkdir(parents=True, exist_ok=True)
        destination = variants / f"{source.name}.tf-worker{os.getpid()}-{self.worker_id}"
        contents = source.read_text(encoding="utf-8")
        contents = contents.replace("HOME", os.getenv("HOME", str(Path.home())))
        contents = contents.replace("BOTTLENECK_BW", f"{self.bw_mbps}Mbps")
        contents = contents.replace("BASE_RTT", f"{self.base_rtt_ms / 2.0}ms")
        contents = contents.replace("BOTTLENECK_BUFFER_SIZE", f"{self.buffer_bits}b")
        contents = contents.replace("MAX_RL_STEPS", str(max_steps))
        contents = contents.replace("NUM_FLOWS", str(num_flows))
        destination.write_text(contents, encoding="utf-8")
        self._log(
            f"ini bw={self.bw_mbps}Mbps rtt={self.base_rtt_ms}ms "
            f"buffer={self.buffer_bits}b flows={num_flows} steps={max_steps}"
        )
        return destination

    def _spawn_episode_process(self, ini_path):
        parent, child = self.mp_context.Pipe()
        self.worker_connection = parent
        self.worker_process = self.mp_context.Process(
            target=run_episode,
            args=(child, str(ini_path), self.env_config["section"]),
            daemon=True,
        )
        self.worker_process.start()
        child.close()
        self.closed = False

    def _receive_worker_message(self):
        try:
            message_type, payload = self.worker_connection.recv()
        except EOFError as exc:
            code = self.worker_process.exitcode
            raise RuntimeError(f"OMNeT++ episode process exited with code {code}") from exc
        if message_type == "error":
            raise RuntimeError(f"OMNeT++ episode process failed:\n{payload}")
        return message_type, payload

    def _cleanup_after_previous_episode(self):
        if self.worker_process is None:
            return
        if not self.closed:
            try:
                self.worker_connection.send(("close", None))
                self._receive_worker_message()
            except Exception as exc:
                print(f"Warning: Astraea episode cleanup failed: {exc}", flush=True)
        self._join_episode_process()
        self.closed = True

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

    def _log(self, message):
        if self.verbose:
            print(f"[astraea-tf-env {self.worker_id}] {message}", flush=True)


def build_agent(env, args, writer):
    return Agent(
        env.state_dim,
        env.global_state_dim,
        env.action_dim,
        h1_shape=args.h1_size,
        h2_shape=args.h2_size,
        batch_size=args.batch_size,
        summary=writer,
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


def parse_range(value, cast=float):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be MIN,MAX")
    return cast(parts[0]), cast(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Original TensorFlow Astraea learner for RayNet OMNeT++"
    )
    parser.add_argument(
        "ini_path",
        nargs="?",
        default=str(RAYNET_PATH / "simlibs/Astraea/src/training/AstraeaTraining.ini"),
    )
    parser.add_argument("--section", default="General")
    parser.add_argument("--seed", type=int, default=91456211)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--stacking", type=int, default=5)
    parser.add_argument("--num-flows", type=lambda x: parse_range(x, int), default=(2, 5))
    parser.add_argument("--bandwidth", type=lambda x: parse_range(x, float), default=(5, 20))
    parser.add_argument("--rtt", type=lambda x: parse_range(x, float), default=(5, 100))
    parser.add_argument(
        "--buffer-bits", type=lambda x: parse_range(x, int), default=(25_000, 2_000_000)
    )
    parser.add_argument("--train-after", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=20)
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument(
        "--log-dir", default=str(RAYNET_PATH / "_models" / "Astraea-original-tf")
    )
    parser.add_argument("--restore", default="")
    parser.add_argument("--quiet-env", action="store_true")
    parser.add_argument("--local-critic", action="store_true")
    parser.add_argument("--no-bootstrap-on-truncation", action="store_true")

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


def main():
    args = parse_args()
    if args.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    base_config = {
        "iniPath": str(Path(args.ini_path).resolve()),
        "section": args.section,
        "bottleneck_bw_range": args.bandwidth,
        "minimum_rtt_range": args.rtt,
        "bottleneck_buffer_range": args.buffer_bits,
        "max_steps_range": (args.max_episode_steps, args.max_episode_steps),
        "num_flows_range": args.num_flows,
        "stacking": args.stacking,
    }
    envs = [
        OmnetAstraeaRolloutEnv(
            dict(base_config, worker_id=worker_id),
            bootstrap_on_truncation=not args.no_bootstrap_on_truncation,
            verbose=not args.quiet_env,
        )
        for worker_id in range(args.num_simulations)
    ]
    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        writer = tf.summary.FileWriter(str(log_dir))
        agent = build_agent(envs[0], args, writer)
        agent.build_learn()
        agent.create_tf_summary()

        with tf.Session() as sess:
            agent.assign_sess(sess)
            sess.run(tf.global_variables_initializer())
            if args.restore:
                restore_path = Path(args.restore)
                if restore_path.is_dir():
                    checkpoint = tf.train.latest_checkpoint(str(restore_path))
                    if checkpoint is None:
                        raise ValueError(f"No TensorFlow checkpoint found in {restore_path}")
                else:
                    checkpoint = str(restore_path)
                agent.saver.restore(sess, checkpoint)
            else:
                agent.init_target()

            step = 0
            train_step = 0
            completed_episodes = 0
            episode_returns = [0.0] * args.num_simulations
            episode_steps = [0] * args.num_simulations
            start = time.time()

            with ThreadPoolExecutor(max_workers=args.num_simulations) as executor:
                resets = list(executor.map(lambda env: env.reset(), envs))
                states = [item[0] for item in resets]
                globals_ = [item[1] for item in resets]
                try:
                    while step < args.total_steps:
                        active_count = min(args.num_simulations, args.total_steps - step)
                        active = list(range(active_count))
                        actions = []
                        for worker_id in active:
                            actions.append(
                                {
                                    agent_id: action_scalar(
                                        agent.get_action(state, use_noise=True)
                                    )
                                    for agent_id, state in states[worker_id].items()
                                }
                            )
                        futures = {
                            worker_id: executor.submit(envs[worker_id].step, actions[index])
                            for index, worker_id in enumerate(active)
                        }
                        reset_workers = []

                        for index, worker_id in enumerate(active):
                            old_states = states[worker_id]
                            old_global = globals_[worker_id]
                            next_states, rewards, terminals, done, next_global = futures[
                                worker_id
                            ].result()
                            worker_actions = actions[index]
                            step += 1
                            episode_steps[worker_id] += 1

                            common_agents = (
                                old_states.keys()
                                & next_states.keys()
                                & worker_actions.keys()
                            )
                            step_rewards = []
                            for agent_id in common_agents:
                                reward = float(rewards.get(agent_id, 0.0))
                                if math.isnan(reward):
                                    reward = 0.0
                                step_rewards.append(reward)
                                agent.store_experience(
                                    old_states[agent_id],
                                    old_global,
                                    np.asarray([worker_actions[agent_id]], dtype=np.float32),
                                    np.asarray([reward], dtype=np.float32),
                                    next_states[agent_id],
                                    next_global,
                                    np.asarray(
                                        [float(terminals.get(agent_id, False))],
                                        dtype=np.float32,
                                    ),
                                )
                                write_scalar(
                                    writer,
                                    "Rollout/action",
                                    worker_actions[agent_id],
                                    step,
                                )

                            mean_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
                            episode_returns[worker_id] += mean_reward
                            states[worker_id] = next_states
                            globals_[worker_id] = next_global
                            write_scalar(writer, "Rollout/mean_reward", mean_reward, step)
                            write_scalar(writer, "Rollout/num_agents", len(next_states), step)
                            write_scalar(writer, "Rollout/replay_size", len(agent.rp_buffer), step)
                            write_scalar(
                                writer,
                                "Performance/environment_steps_per_second",
                                step / max(time.time() - start, 1e-6),
                                step,
                            )

                            if len(agent.rp_buffer) >= args.train_after and step % args.train_every == 0:
                                for _ in range(args.updates_per_train):
                                    train_step += 1
                                    agent.train_step(train_step)

                            if done:
                                print(
                                    f"episode={completed_episodes} worker={worker_id} "
                                    f"step={step} episode_steps={episode_steps[worker_id]} "
                                    f"return={episode_returns[worker_id]:.4f} "
                                    f"buffer={len(agent.rp_buffer)}",
                                    flush=True,
                                )
                                write_scalar(
                                    writer,
                                    "Episode/return",
                                    episode_returns[worker_id],
                                    completed_episodes,
                                )
                                write_scalar(
                                    writer,
                                    "Episode/length",
                                    episode_steps[worker_id],
                                    completed_episodes,
                                )
                                completed_episodes += 1
                                episode_returns[worker_id] = 0.0
                                episode_steps[worker_id] = 0
                                if step < args.total_steps:
                                    reset_workers.append(worker_id)

                            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                                agent.ckpt_dir = str(checkpoint_dir)
                                agent.save_model(step)

                        if reset_workers:
                            reset_futures = {
                                worker_id: executor.submit(envs[worker_id].reset)
                                for worker_id in reset_workers
                            }
                            for worker_id, future in reset_futures.items():
                                states[worker_id], globals_[worker_id] = future.result()
                            if agent.actor_noise is not None:
                                agent.actor_noise.reset()
                        writer.flush()
                finally:
                    agent.ckpt_dir = str(checkpoint_dir)
                    agent.save_model(step)
                    list(executor.map(lambda env: env.close(), envs))
                    writer.close()


if __name__ == "__main__":
    main()
