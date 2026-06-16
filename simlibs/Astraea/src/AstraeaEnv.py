"""Protocol-specific OMNeT++ environment wrapper for Astraea."""

import multiprocessing
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.AstraeaEpisodeWorker import run_episode


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

    raw_obs_dim = 8
    global_state_dim = 12
    action_dim = 1

    def __init__(self, env_config, bootstrap_on_truncation=True, verbose=True):
        """Initialize a persistent Astraea environment wrapper."""
        # Store the environment configuration and protocol-specific settings.
        self.env_config = env_config
        self.worker_id = int(env_config.get("worker_id", 0))
        self.stacking = int(env_config["stacking"])
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.verbose = verbose

        # Initialize the transient simulation process and IPC state.
        self.mp_context = multiprocessing.get_context("spawn")
        self.rng = np.random.default_rng(os.getpid() + self.worker_id)
        self.worker_process = None
        self.worker_connection = None
        self.closed = True

        # Initialize persistent observation histories and scenario metadata.
        self.histories = defaultdict(self._new_history)
        self.raw_observations = {}
        self.bw_mbps = 0.0
        self.base_rtt_ms = 0.0
        self.buffer_bits = 0

    @property
    def state_dim(self):
        """Return the flattened recurrent-state dimension."""
        return self.stacking * self.raw_obs_dim

    def reset(self):
        """Start a fresh simulation episode and return its initial states."""
        # Clean up the previous episode and reset all agent histories.
        self._cleanup_after_previous_episode()
        self.histories.clear()
        self.raw_observations = {}

        # Spawn a new simulation and receive its initial observations.
        worker_ini = self._write_worker_ini()
        self._spawn_episode_process(worker_ini)
        message_type, observations = self._receive_worker_message()

        # Validate and process the reset response.
        if message_type != "reset":
            raise RuntimeError(f"Expected reset message from episode worker, got {message_type}")
        if not observations:
            raise RuntimeError("runner.reset() returned no Astraea observations")
        self._record_observations(observations, fill_history=True)
        self._log(f"reset agents={sorted(self.raw_observations)}")
        return self.states(), self.global_state()

    def step(self, learner_actions):
        """Apply learner actions and return the processed simulation step."""
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

        truncated = bool(info.get("simDone", False))
        episode_done = bool(terminateds.get("__all__", False) or truncated)
        terminals = {}
        for agent_id in self.raw_observations:
            terminated = bool(terminateds.get(agent_id, terminateds.get("__all__", False)))
            terminals[agent_id] = terminated or (truncated and not self.bootstrap_on_truncation)

        # Discard simulation processes that ended through termination or truncation.
        if episode_done:
            self.closed = True
            self._join_episode_process()

        # Return the complete processed result to the learner or evaluator.
        return AstraeaStep(
            states=self.states(),
            rewards=rewards,
            terminals=terminals,
            episode_done=episode_done,
            global_state=self.global_state(),
        )

    def states(self):
        """Return the stacked local state for every currently observed agent."""
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

        # Original Astraea normalized throughput as bits/s, latency as us, and cwnd as packets.
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
        """Close the active simulation episode process."""
        self._cleanup_after_previous_episode()

    def _new_history(self):
        """Create one zero-filled observation history."""
        return deque(
            [np.zeros(self.raw_obs_dim, dtype=np.float32) for _ in range(self.stacking)],
            maxlen=self.stacking,
        )

    def _record_observations(self, observations, fill_history=False):
        """Validate raw simulator observations and update agent histories."""
        # Process each returned agent observation independently.
        for agent_id, observation in observations.items():
            if agent_id == "__all__":
                continue
            raw = np.asarray(observation, dtype=np.float32)
            if raw.shape != (self.raw_obs_dim,):
                raise ValueError(f"{agent_id} returned observation shape {raw.shape}; expected ({self.raw_obs_dim},)")

            # Original Astraea begins recurrent state with zero history and shifts the first real observation into the newest slot.
            self.raw_observations[agent_id] = raw
            if fill_history:
                self.histories[agent_id] = self._new_history()
            self.histories[agent_id].append(raw)

    def _write_worker_ini(self):
        """Create one randomized INI variant for a training episode."""
        # Sample the randomized network parameters.
        self.bw_mbps = round(self.rng.uniform(*self.env_config["bottleneck_bw_range"]))
        self.base_rtt_ms = round(self.rng.uniform(*self.env_config["minimum_rtt_range"]), 2)
        self.buffer_bits = round(self.rng.uniform(*self.env_config["bottleneck_buffer_range"]))
        max_steps = round(self.rng.uniform(*self.env_config["max_steps_range"]))
        num_flows = round(self.rng.uniform(*self.env_config["num_flows_range"]))

        # Replace the training placeholders and write the generated INI.
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
