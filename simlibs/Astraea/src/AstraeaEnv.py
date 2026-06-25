"""Protocol-specific OMNeT++ environment wrapper for Astraea."""

import multiprocessing
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.AstraeaEpisodeWorker import run_episode


IGNORED_AGENT_IDS = {"__all__", "SIMULATION_END"}


def action_scalar(action):
    """Normalize a learner action to the scalar expected by the simulator."""
    return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])


@dataclass
class AstraeaStep:
    states: dict
    rewards: dict
    terminals: dict
    episode_done: bool
    global_state: np.ndarray


# MARK: AstraeaEnv ----------------------------------------------------------------------------------------------
class AstraeaEnv:
    """Complete Astraea environment interface used by training and evaluation."""

    raw_obs_dim = 10
    learner_obs_dim = 10
    global_state_dim = 12
    action_dim = 1
    avg_thr_index = 0
    max_tput_index = 1
    avg_urtt_index = 2
    min_rtt_index = 3
    srtt_us_index = 4
    cwnd_index = 5
    loss_ratio_index = 6
    packets_out_index = 7
    pacing_rate_index = 8
    retrans_out_index = 9
    raw_observation_fields = (
        "avg_thr",
        "max_tput",
        "avg_urtt",
        "min_rtt",
        "srtt_us",
        "cwnd",
        "loss_rate",
        "packets_out",
        "pacing_rate",
        "retrans_out",
    )
    paper_mss_bytes = 1460.0
    reward_delay_coefficient = 0.5
    throughput_weight = 0.1
    latency_weight = 0.02
    loss_weight = 1.0
    fairness_weight = 0.02
    stability_weight = 0.01

    def __init__(self, env_config, bootstrap_on_truncation=True, verbose=True):
        """Initialize a persistent Astraea environment wrapper."""
        # Store the environment configuration and protocol-specific settings.
        self.env_config = env_config
        self.worker_id = int(env_config.get("worker_id", 0))
        self.stacking = int(env_config["stacking"])
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.verbose = verbose
        self.randomize_ini = bool(env_config.get("randomize_ini", False))

        # Initialize the transient simulation process and IPC state.
        self.mp_context = multiprocessing.get_context("spawn")
        self.rng = np.random.default_rng(os.getpid() + self.worker_id)
        self.worker_process = None
        self.worker_connection = None
        self.closed = True

        # Initialize persistent observation histories and scenario metadata.
        self.histories = defaultdict(self._new_history)
        self.throughput_histories = defaultdict(lambda: deque(maxlen=self.stacking))
        self.raw_observations = {}
        self.current_agent_ids = set()
        self.episode_steps = 0
        self.max_episode_steps = int(env_config.get("max_episode_steps", 0))
        self.bw_mbps = float(env_config.get("bottleneck_bw_range", (0.0, 0.0))[0])
        self.base_rtt_ms = float(env_config.get("minimum_rtt_range", (0.0, 0.0))[0])
        self.buffer_bits = float(env_config.get("bottleneck_buffer_range", (0.0, 0.0))[0])
        if self.max_episode_steps == 0 and "max_steps_range" in env_config:
            self.max_episode_steps = int(env_config["max_steps_range"][0])

    @property
    def state_dim(self):
        """Return the flattened recurrent-state dimension."""
        return self.stacking * self.learner_obs_dim

    def reset(self):
        """Start a fresh simulation episode and return its initial states."""
        # Clean up the previous episode and reset all agent histories.
        self._cleanup_after_previous_episode()
        self.histories.clear()
        self.throughput_histories.clear()
        self.raw_observations = {}
        self.current_agent_ids = set()
        self.episode_steps = 0

        # Spawn a new simulation and receive its initial observations.
        worker_ini = self._write_worker_ini() if self.randomize_ini else self.env_config["iniPath"]
        self._spawn_episode_process(worker_ini)
        message_type, observations = self._receive_worker_message()

        # Validate and process the reset response.
        if message_type != "reset":
            raise RuntimeError(f"Expected reset message from episode worker, got {message_type}")
        if not observations:
            raise RuntimeError("runner.reset() returned no Astraea observations")
        self._record_observations(observations, fill_history=True)
        if not self.raw_observations:
            self.closed = True
            self._join_episode_process()
            raise RuntimeError("Simulation ended before any Astraea observation was returned")
        self._log(f"reset agents={sorted(self.raw_observations)}")
        return self.states(), self.global_state()

    def step(self, learner_actions):
        """Apply learner actions and return the processed simulation step."""
        # Astraea makes one decision for each fresh observation; never replay stale actions for quiet agents.
        unexpected_agent_ids = set(learner_actions) - self.current_agent_ids
        if unexpected_agent_ids:
            raise ValueError(f"Received actions for agents without fresh Astraea observations: {sorted(unexpected_agent_ids)}")

        # Normalize learner actions before forwarding them to OMNeT++.
        actions = {}
        for agent_id, action in learner_actions.items():
            actions[agent_id] = float(np.clip(action_scalar(action), -1.0, 1.0))

        self.worker_connection.send(("step", actions))
        message_type, result = self._receive_worker_message()
        if message_type != "step":
            raise RuntimeError(f"Expected step message from episode worker, got {message_type}")

        # Derive learner states and terminal flags from the simulator response.
        observations, rewards, terminateds, info = result
        if observations:
            self._record_observations(observations)
        else:
            self.current_agent_ids = set()

        truncated = bool(info.get("simDone", False))
        self.episode_steps += 1
        reached_step_limit = self.max_episode_steps > 0 and self.episode_steps >= self.max_episode_steps
        episode_done = bool(terminateds.get("__all__", False) or truncated or reached_step_limit)
        terminals = {}
        for agent_id in self.raw_observations:
            terminated = bool(terminateds.get(agent_id, terminateds.get("__all__", False)))
            terminals[agent_id] = terminated or ((truncated or reached_step_limit) and not self.bootstrap_on_truncation)

        # Discard simulation processes that ended through termination or truncation.
        if episode_done:
            if terminateds.get("__all__", False) or truncated:
                self.closed = True
                self._join_episode_process()
            else:
                self._cleanup_after_previous_episode()

        # Return the complete processed result to the learner or evaluator.
        shaped_rewards = self.rewards()
        return AstraeaStep(
            states=self.states(),
            rewards=shaped_rewards,
            terminals=terminals,
            episode_done=episode_done,
            global_state=self.global_state(),
        )

    def states(self):
        """Return stacked local states for agents with fresh observations."""
        return {
            agent_id: np.concatenate(self.histories[agent_id]).astype(np.float32)
            for agent_id in sorted(self.current_agent_ids)
        }

    def global_state(self):
        """Reconstruct the original learner's 12-value centralized critic input."""
        if not self.raw_observations:
            return np.zeros(self.global_state_dim, dtype=np.float32)

        obs = np.asarray(list(self.raw_observations.values()), dtype=np.float64)
        throughput = obs[:, self.avg_thr_index] / 5e7
        latency = obs[:, self.avg_urtt_index] / 5e5
        cwnd_packets = obs[:, self.cwnd_index]
        loss = obs[:, self.loss_ratio_index] / 1e6
        bdp_packets = self.bw_mbps * self.base_rtt_ms * 1000.0 / (self.paper_mss_bytes * 8.0)
        return np.asarray(
            [
                np.sum(throughput),
                np.min(throughput),
                np.max(throughput),
                np.mean(latency),
                np.min(cwnd_packets) / 1000.0,
                np.max(cwnd_packets) / 1000.0,
                np.mean(cwnd_packets) / 1000.0,
                np.mean(loss),
                len(obs) / 10.0,
                (self.base_rtt_ms / 2.0) / 500.0,
                bdp_packets / 10.0,
                self.bw_mbps / 500.0,
            ],
            dtype=np.float32,
        )

    def rewards(self):
        """Compute the original global reward from the latest raw metrics."""
        if not self.raw_observations:
            return {}

        # Convert raw per-flow metrics to the units used by the original reward formula.
        obs = np.asarray(list(self.raw_observations.values()), dtype=np.float64)
        throughput_bytes = obs[:, self.avg_thr_index]
        latency_seconds = obs[:, self.avg_urtt_index] / 1e6
        loss_bytes = obs[:, self.loss_ratio_index]
        pacing_packets = obs[:, self.pacing_rate_index] / self.paper_mss_bytes

        num_flows = len(obs)
        bandwidth_bytes = max(self.bw_mbps * 125000.0, 1e-9)
        link_delay_seconds = (self.base_rtt_ms / 2.0) / 1000.0
        throughput_metric = float(np.sum(throughput_bytes) / bandwidth_bytes)
        latency_threshold = (1.0 + self.reward_delay_coefficient) * link_delay_seconds
        latency_metric = max(float(np.mean(latency_seconds)) - latency_threshold, 0.0) * float(np.mean(pacing_packets))
        loss_terms = [loss / throughput for loss, throughput in zip(loss_bytes, throughput_bytes) if throughput > 0.0]
        loss_metric = float(np.mean(loss_terms)) if loss_terms else 0.0

        average_throughputs = []
        stabilities = []
        for agent_id in self.raw_observations:
            history = list(self.throughput_histories[agent_id])
            average_throughput = float(np.mean(history)) if history else 0.0
            average_throughputs.append(average_throughput)
            denominator = len(history) * average_throughput ** 2
            if denominator > 0.0:
                stabilities.append(float(np.sqrt(np.sum((np.asarray(history) - average_throughput) ** 2) / denominator)))
            else:
                stabilities.append(0.0)

        global_average_throughput = float(np.mean(average_throughputs)) if average_throughputs else 0.0
        fairness_denominator = num_flows * float(np.sum(average_throughputs)) ** 2
        if fairness_denominator > 0.0:
            fairness_metric = float(np.sqrt(np.sum((np.asarray(average_throughputs) - global_average_throughput) ** 2) / fairness_denominator))
        else:
            fairness_metric = 0.0
        stability_metric = float(np.mean(stabilities)) if stabilities else 0.0

        reward = self.throughput_weight * throughput_metric
        reward -= self.latency_weight * latency_metric
        reward -= self.loss_weight * loss_metric
        reward -= self.fairness_weight * fairness_metric
        reward -= self.stability_weight * stability_metric
        reward = float(np.clip(reward, -self.throughput_weight, self.throughput_weight))
        return {agent_id: reward for agent_id in self.raw_observations}

    def close(self):
        """Close the active simulation episode process."""
        self._cleanup_after_previous_episode()

    def _new_history(self):
        """Create one zero-filled observation history."""
        return deque(
            [np.zeros(self.learner_obs_dim, dtype=np.float32) for _ in range(self.stacking)],
            maxlen=self.stacking,
        )

    def _record_observations(self, observations, fill_history=False):
        """Validate raw simulator observations and update agent histories."""
        # Process each returned agent observation independently.
        current_agent_ids = set()
        for agent_id, observation in observations.items():
            if agent_id in IGNORED_AGENT_IDS:
                continue
            raw = self._raw_observation_array(observation)
            if raw.shape != (self.raw_obs_dim,):
                raise ValueError(f"{agent_id} returned observation shape {raw.shape}; expected ({self.raw_obs_dim},)")

            # Original Astraea begins recurrent state with zero history and shifts the first real observation into the newest slot.
            self.raw_observations[agent_id] = raw
            learner_observation = self._derive_learner_observation(raw)
            if fill_history:
                self.histories[agent_id] = self._new_history()
            self.histories[agent_id].append(learner_observation)
            self.throughput_histories[agent_id].append(float(raw[self.avg_thr_index]))
            current_agent_ids.add(agent_id)
        self.current_agent_ids = current_agent_ids

    def _derive_learner_observation(self, raw):
        """Reproduce the original Astraea 10-feature local state transform."""
        avg_thr = float(raw[self.avg_thr_index])
        max_tput = float(raw[self.max_tput_index])
        avg_urtt = float(raw[self.avg_urtt_index])
        min_rtt = float(raw[self.min_rtt_index])
        srtt_us = float(raw[self.srtt_us_index])
        cwnd = float(raw[self.cwnd_index])
        loss_ratio = float(raw[self.loss_ratio_index])
        packets_out = float(raw[self.packets_out_index])
        pacing_rate = float(raw[self.pacing_rate_index])
        retrans_out = float(raw[self.retrans_out_index])

        state = np.asarray(
            [
                0.5 if avg_thr == 0.0 else avg_thr / max_tput if max_tput > 0.0 else 0.0,
                2.0 if avg_urtt == 0.0 else avg_urtt / min_rtt if min_rtt > 0.0 else 0.0,
                2.0 if srtt_us == 0.0 else srtt_us / 8.0 / min_rtt if min_rtt > 0.0 else 0.0,
                cwnd * self.paper_mss_bytes * 8.0 / (min_rtt / 1e6) / max_tput / 10.0 if min_rtt > 0.0 and max_tput > 0.0 else 0.0,
                max_tput / 1e7,
                min_rtt / 5e5,
                loss_ratio / max_tput if max_tput > 0.0 else 0.0,
                packets_out / cwnd if cwnd > 0.0 else 0.0,
                pacing_rate / max_tput if max_tput > 0.0 else 0.0,
                retrans_out / cwnd if cwnd > 0.0 else 0.0,
            ],
            dtype=np.float32,
        )
        state[[1, 2, 3, 8]] = np.minimum(state[[1, 2, 3, 8]], 2.0)
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    def _raw_observation_array(self, observation):
        """Return Astraea raw metrics in the learner's canonical order."""
        if isinstance(observation, dict):
            return np.asarray(
                [float(observation[name]) for name in self.raw_observation_fields],
                dtype=np.float32,
            )
        return np.asarray(observation, dtype=np.float32)

    def _write_worker_ini(self):
        """Create one randomized INI variant for a training episode."""
        # Sample the randomized network parameters.
        self.bw_mbps = round(self.rng.uniform(*self.env_config["bottleneck_bw_range"]))
        self.base_rtt_ms = round(self.rng.uniform(*self.env_config["minimum_rtt_range"]), 2)
        self.buffer_bits = round(self.rng.uniform(*self.env_config["bottleneck_buffer_range"]))
        max_steps = round(self.rng.uniform(*self.env_config["max_steps_range"]))
        self.max_episode_steps = max_steps
        num_flows = round(self.rng.uniform(*self.env_config["num_flows_range"]))

        # Replace the training placeholders and write the generated INI.
        source = Path(self.env_config["iniPath"])
        variants = source.parent if source.parent.name == "ini_variants" else source.parent / "ini_variants"
        variants.mkdir(parents=True, exist_ok=True)
        destination = variants / f"{source.name}.tf-worker{os.getpid()}-{self.worker_id}"
        contents = source.read_text(encoding="utf-8")
        contents = contents.replace("!HOME!", os.getenv("HOME", str(Path.home())))
        contents = contents.replace("!BW!", f"{self.bw_mbps}Mbps")
        contents = contents.replace("!DELAY!", f"{self.base_rtt_ms / 2.0}ms")
        contents = contents.replace("!QSIZE!", f"{self.buffer_bits}b")
        contents = contents.replace("!MAX_RL_STEPS!", str(max_steps))
        contents = contents.replace("NUM_FLOWS", str(num_flows))
        destination.write_text(contents, encoding="utf-8")
        self._log(f"ini bw={self.bw_mbps}Mbps rtt={self.base_rtt_ms}ms buffer={self.buffer_bits}b flows={num_flows} steps={max_steps}")
        return destination

    def _spawn_episode_process(self, ini_path):
        """Spawn a simulation process for one episode."""
        # Create a duplex Pipe for communicating with the simulation worker.
        parent_connection, child_connection = self.mp_context.Pipe()
        self.worker_connection = parent_connection

        # Spawn the worker process and transfer ownership of its Pipe endpoint.
        self.worker_process = self.mp_context.Process(
            target=run_episode,
            args=(child_connection, str(ini_path), self.env_config["section"]),
            daemon=True,
        )
        self.worker_process.start()
        child_connection.close()
        self.closed = False

    def _receive_worker_message(self):
        """Receive and validate one message from the simulation worker."""
        # Receive the next message or report an unexpected worker exit.
        try:
            message_type, payload = self.worker_connection.recv()
        except EOFError as exc:
            exit_code = self.worker_process.exitcode
            raise RuntimeError(f"OMNeT++ episode process exited unexpectedly with code {exit_code}") from exc

        # Promote worker errors into exceptions in the parent process.
        if message_type == "error":
            raise RuntimeError(f"OMNeT++ episode process failed:\n{payload}")
        return message_type, payload

    def _cleanup_after_previous_episode(self):
        """Close and discard the previous episode process."""
        # Return immediately when there is no active process to clean up.
        if self.worker_process is None:
            return

        # Ask unfinished simulations to shut down cleanly.
        if not self.closed:
            try:
                self.worker_connection.send(("close", None))
                message_type, _ = self._receive_worker_message()
                if message_type != "closed":
                    print(f"Warning: expected closed message, got {message_type}", flush=True)
            except Exception as exc:
                print(f"Warning: Astraea episode cleanup failed: {exc}", flush=True)

        # Join the worker and mark the environment as closed.
        self._join_episode_process()
        self.closed = True

    def _join_episode_process(self):
        """Join the episode process and close its Pipe endpoint."""
        # Return immediately when there is no active process to join.
        if self.worker_process is None:
            return

        # Wait for a clean exit before terminating a stuck worker.
        self.worker_process.join(timeout=10)
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)

        # Close the parent Pipe endpoint and discard process references.
        if self.worker_connection is not None:
            self.worker_connection.close()
        self.worker_process = None
        self.worker_connection = None

    def _log(self, message):
        """Print an environment log message when verbose output is enabled."""
        if self.verbose:
            print(f"[astraea-env {self.worker_id}] {message}", flush=True)
