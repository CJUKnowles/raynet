"""Protocol-specific OMNeT++ environment wrapper for Orca."""

import math
import multiprocessing
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORED_AGENT_IDS = {"__all__", "SIMULATION_END"}


def action_scalar(action):
    """Normalize a learner action to the scalar expected by the simulator."""
    return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])


def _serialize_observations(observations):
    """Convert OmnetBind objects to lists for sending over the Pipe."""
    serialized = {}

    # Convert each OmnetBind observation into a serializable Python list.
    for agent_id, observation in observations.items():
        if hasattr(observation, "to_list"):
            serialized[agent_id] = observation.to_list()
        else:
            serialized[agent_id] = list(observation)

    return serialized


def run_episode(connection, ini_path, section_name, primary_agent_id):
    """Create and run a single OMNeT++ episode."""
    # Import OmnetBind inside the worker process that will own the simulation.
    raynet_path = Path(os.environ["RAYNET_PATH"])
    import sys

    sys.path.insert(0, str(raynet_path / "build"))
    from omnetbind import OmnetGymApi

    # Create the simulator and track its finalization state.
    runner = OmnetGymApi()
    cleaned = False
    shutdown_required = True

    try:
        # Initialize the simulation and return its first observations.
        runner.initialise(ini_path, section_name)
        connection.send(("reset", _serialize_observations(runner.reset())))

        # Continuously receive and perform commands from the OrcaEnv until the episode is complete
        while True:
            command, payload = connection.recv()

            if command == "step":
                # Advance the simulation and return serializable step results.
                observations, rewards, terminateds, info = runner.step(payload)
                result = (
                    _serialize_observations(observations),
                    {key: float(value) for key, value in rewards.items()},
                    {key: bool(value) for key, value in terminateds.items()},
                    {key: bool(value) for key, value in info.items()},
                )

                # Finalize natural simulation completion with cleanup only.
                simulation_ended = info.get("simDone", False)
                agents_terminated = terminateds.get("__all__", terminateds.get(primary_agent_id, False))
                if simulation_ended:
                    shutdown_required = False
                    runner.cleanup()
                    cleaned = True

                # Finalize agent-requested termination with shutdown and cleanup.
                elif agents_terminated:
                    runner.shutdown()
                    runner.cleanup()
                    cleaned = True

                # Return the step only after finalization has completed.
                connection.send(("step", result))
                if simulation_ended or agents_terminated:
                    break

            elif command == "close":
                # Clean up and acknowledge an explicit close request.
                runner.shutdown()
                runner.cleanup()
                cleaned = True
                connection.send(("closed", None))
                break

            else:
                raise ValueError(f"Unknown episode-worker command: {command}")

    except BaseException:
        # Forward worker failures to the parent before allowing the process to fail.
        try:
            connection.send(("error", traceback.format_exc()))
        except BaseException:
            pass
        raise

    finally:
        # Clean up unfinished simulations and always close the worker connection.
        if not cleaned:
            if shutdown_required:
                try:
                    runner.shutdown()
                except BaseException:
                    pass
            try:
                runner.cleanup()
            except BaseException:
                pass
        connection.close()


@dataclass
class OrcaStep:
    states: dict
    rewards: dict
    terminateds: dict
    truncated: bool

    @property
    def episode_done(self):
        """Return whether the episode has terminated or truncated."""
        return bool(self.terminateds.get("__all__", False) or self.truncated)





# MARK: OrcaEnv -------------------------------------------------------------------------------------------------
class OrcaEnv:
    """Complete Orca environment interface used by training and evaluation.

    This class is responsible for a single simulation process at a time, and spawns a new one upon each reset.
    Think of this class as representing a single CPU core, repeatedly spawning and running new simulations as needed.
    This is synonymous with a Ray worker. In the future, Ray may be reintegrated without RLlib.
    """

    simulator_obs_dim = 15
    raw_obs_dim = 7
    action_dim = 1
    delay_index = 0
    samples_index = 2
    cwnd_index = 5
    srtt_index = 8
    ssthresh_index = 9
    min_rtt_index = 14
    default_delay_margin_coefficient = 1.25
    default_loss_coefficient = 5.0

    def __init__(self, env_config, verbose=True):
        """Initialize a persistent Orca environment wrapper."""
        # Store the environment configuration and protocol-specific settings.
        self.env_config = env_config
        self.verbose = verbose
        self.stacking = int(env_config.get("stacking", 10))
        self.primary_agent_id = env_config.get("agent_id", "Orca")
        self.delay_margin_coefficient = float(env_config.get("delay_margin_coefficient", self.default_delay_margin_coefficient))
        self.loss_coefficient = float(env_config.get("loss_coefficient", self.default_loss_coefficient))

        # Initialize the transient simulation process and IPC state.
        self.mp_context = multiprocessing.get_context("spawn")
        self.worker_process = None
        self.worker_connection = None

        # Initialize persistent observation histories.
        self.obs_histories = {}
        self.max_throughputs = {}
        self.learner_ready_agent_ids = set()
        self.pending_agent_ids = set()
        self.closed = True

    @property
    def state_dim(self):
        """Return the flattened recurrent-state dimension."""
        return self.stacking * self.raw_obs_dim

    def reset(self, ini_path=None):
        """Start a fresh simulation episode and return its initial states."""
        # Clean up the previous episode and reset all agent histories.
        self._log("reset: preparing new OMNeT++ run")
        self._cleanup_after_previous_episode()
        self.obs_histories = {}
        self.max_throughputs = {}
        self.learner_ready_agent_ids = set()
        self.pending_agent_ids = set()

        # Spawn a new simulation and receive its initial observations.
        ini_path = ini_path if ini_path is not None else self.env_config["iniPath"]
        section_name = self.env_config.get("config_section", "General")
        self._spawn_episode_process(ini_path, section_name)
        message_type, observations = self._receive_worker_message()

        # Validate and process the reset response.
        if message_type != "reset":
            raise RuntimeError(f"Expected reset message from episode worker, got {message_type}")
        states, _ = self._process_observations(observations)

        # Keep Cubic in control until the first agent has completed initial slow start.
        while not states:
            actions = {agent_id: 0.0 for agent_id in self.pending_agent_ids}
            self.worker_connection.send(("step", actions))
            message_type, result = self._receive_worker_message()
            if message_type != "step":
                raise RuntimeError(f"Expected step message while waiting for Orca startup, got {message_type}")
            observations, _, terminateds, info = result
            if info.get("simDone", False) or terminateds.get("__all__", False):
                self.closed = True
                self._join_episode_process()
                raise RuntimeError("Simulation ended before any Orca agent completed initial slow start")
            states, _ = self._process_observations(observations)

        return states

    def step(self, learner_actions):
        """Apply learner actions and return the processed simulation step."""
        # Reject actions for agents that did not produce the latest observations.
        unexpected_agent_ids = set(learner_actions) - self.pending_agent_ids
        if unexpected_agent_ids:
            raise ValueError(f"Received actions for agents without pending observations: {sorted(unexpected_agent_ids)}")

        # Normalize only the new actions requested by the learner.
        actions = {}
        for agent_id, action in learner_actions.items():
            actions[agent_id] = float(np.clip(action_scalar(action), -1.0, 1.0))

        self._log(f"step: actions={actions}")
        self.worker_connection.send(("step", actions)) # Perform the step
        message_type, result = self._receive_worker_message()
        if message_type != "step":
            raise RuntimeError(f"Expected step message from episode worker, got {message_type}")

        # Derive learner states and rewards from the simulator's raw metrics.
        observations, _, terminateds, info = result
        states, rewards = self._process_observations(observations)

        # Discard simulation processes that ended through termination or truncation.
        truncated = bool(info.get("simDone", False))
        terminated = bool(terminateds.get("__all__", terminateds.get(self.primary_agent_id, False)))
        if terminated or truncated:
            self.closed = True
            self._join_episode_process()

        # Return the complete processed result to the learner or evaluator.
        return OrcaStep(
            states=states,
            rewards=rewards,
            terminateds=terminateds,
            truncated=truncated,
        )

    def close(self):
        """Close the active simulation episode process."""
        self._cleanup_after_previous_episode()

    def _process_observations(self, observations):
        """Validate raw simulator observations and derive learner states and rewards."""
        states = {}
        rewards = {}
        self.pending_agent_ids = set()

        # Process each returned agent observation independently.
        for agent_id, observation in observations.items():
            if agent_id in IGNORED_AGENT_IDS:
                continue
            self.pending_agent_ids.add(agent_id)
            observation = np.asarray(observation, dtype=np.float32)

            # Validate the raw shape and create a zero-padded history for new agents.
            if observation.size != self.simulator_obs_dim:
                raise ValueError(f"Expected {self.simulator_obs_dim} raw Orca metrics for {agent_id}, got {observation.size}")
            if agent_id not in self.obs_histories:
                self.obs_histories[agent_id] = np.zeros((self.stacking, self.raw_obs_dim), dtype=np.float32,)

            # Permanently enable learner interaction after this agent first exits Cubic slow start.
            if self.observation_has_valid_rtt(observation) and observation[self.cwnd_index] > observation[self.ssthresh_index]:
                self.learner_ready_agent_ids.add(agent_id)

            # Omit pre-handoff and invalid observations so Cubic remains responsible for startup.
            if agent_id not in self.learner_ready_agent_ids or not self.observation_has_valid_rtt(observation):
                continue

            # Derive the original Orca learner observation and reward from the raw metrics.
            learner_observation, reward = self._derive_observation_and_reward(agent_id, observation)

            # Add the derived observation and return a copy that cannot be mutated by future steps.
            history = self.obs_histories[agent_id]
            history[:-1] = history[1:]
            history[-1] = learner_observation
            states[agent_id] = self.obs_histories[agent_id].reshape(-1).copy()
            rewards[agent_id] = reward

        return states, rewards

    def _derive_observation_and_reward(self, agent_id, raw_observation):
        """Reproduce the original Orca wrapper's seven-feature state and reward."""
        # Return a neutral state for startup observations that contain no valid RTT measurement.
        if not self.observation_has_valid_rtt(raw_observation):
            return np.zeros(self.raw_obs_dim, dtype=np.float32), 0.0

        # Read the raw metrics used by the original Orca state calculation.
        throughput = float(raw_observation[1])
        samples = float(raw_observation[2])
        interval_duration = float(raw_observation[3])
        cwnd = float(raw_observation[5])
        pacing_rate = float(raw_observation[6])
        loss_rate = float(raw_observation[7])
        srtt = float(raw_observation[8])
        min_rtt = float(raw_observation[14])

        # Track the maximum throughput independently for each Orca flow.
        max_throughput = max(self.max_throughputs.get(agent_id, 0.0), throughput)
        self.max_throughputs[agent_id] = max_throughput

        # Compute the original delay metric and throughput-normalized values.
        delay_metric = min(1.0, min_rtt * self.delay_margin_coefficient / srtt)
        if max_throughput > 0.0:
            normalized_throughput = throughput / max_throughput
            normalized_pacing_rate = min(10.0, pacing_rate / max_throughput)
            normalized_loss_rate = self.loss_coefficient * loss_rate / max_throughput
            reward = (throughput - self.loss_coefficient * loss_rate) / max_throughput * delay_metric
        else:
            normalized_throughput = 0.0
            normalized_pacing_rate = 0.0
            normalized_loss_rate = 0.0
            reward = 0.0

        # Assemble the seven features consumed by the original Orca learner.
        observation = np.asarray([
            normalized_throughput,
            normalized_pacing_rate,
            normalized_loss_rate,
            samples / cwnd if cwnd > 0.0 else 0.0,
            interval_duration,
            min_rtt / srtt,
            delay_metric,
        ], dtype=np.float32)

        # Replace non-finite results defensively before exposing them to the learner.
        if not np.all(np.isfinite(observation)) or not math.isfinite(reward):
            print(f"Warning: non-finite derived observation or reward for {agent_id}; replacing invalid values with 0.0")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            reward = reward if math.isfinite(reward) else 0.0

        return observation, float(reward)

    def observation_has_valid_rtt(self, observation):
        """Return whether a raw Orca observation contains valid RTT data."""
        return bool(observation[self.delay_index] > 0.0 and observation[self.samples_index] > 0.0 and observation[self.srtt_index] > 0.0 and observation[self.min_rtt_index] > 0.0)

    def _spawn_episode_process(self, ini_path, section_name):
        """Spawn a simulation process for one episode."""
        # Create a duplex Pipe for communicating with the simulation worker.
        parent_connection, child_connection = self.mp_context.Pipe()
        self.worker_connection = parent_connection

        # Spawn the worker process and transfer ownership of its Pipe endpoint.
        self.worker_process = self.mp_context.Process(
            target=run_episode,
            args=(child_connection, str(ini_path), section_name, self.primary_agent_id),
            daemon=True,
        )
        self.worker_process.start()
        child_connection.close()
        self.closed = False
        self._log(f"reset: spawned episode process pid={self.worker_process.pid}")

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
                    print(f"Warning: expected closed message, got {message_type}")
            except Exception as exc:
                print(f"Warning: episode worker cleanup failed: {exc}")

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
            print(f"[orca-env] {message}", flush=True)
