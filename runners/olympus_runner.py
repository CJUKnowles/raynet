#!/usr/bin/env python3
"""RayNet-owned OMNeT++ episode runner for Olympus.

This process is the boundary between RayNet and external training systems. It
owns RayNet environment setup, imports ``omnetbind`` from the RayNet build
tree, materializes any INI template placeholders, and exposes one simulation
episode over a JSON-lines control socket.
"""

import argparse
import json
import os
import re
import socket
import tempfile
import traceback
from pathlib import Path

from raynet import experimentTools, iniTools, obsTools, unitTools

# MARK: Simulator Step Handling ---------------------------------------------------------------------------------
def _serialize_observations(observations, observation_fields):
    """Format raw simulator observations for Olympus worker flow backends."""
    return obsTools.serialize_observations(
        observations,
        value_type=float,
        key_type=str,
        fields=observation_fields,
    )


def _serialize_info(info):
    """Format simulator metadata without changing timing semantics."""
    serialized = obsTools.info_dict(info)
    serialized["simDone"] = bool(serialized.get("simDone", False))
    if "group_step" not in serialized and "step_id" in serialized:
        serialized["group_step"] = serialized["step_id"]
    return serialized


def _step_message(runner, actions, observation_fields):
    """Advance the simulator and format one step response."""
    observations, rewards, terminateds, info = runner.step(actions or {})
    info = dict(info or {})
    info.setdefault("time_s", runner.sim_time())
    return {
        "type": "step",
        "observations": _serialize_observations(observations, observation_fields),
        "rewards": obsTools.float_dict(rewards),
        "terminateds": obsTools.bool_dict(terminateds),
        "info": _serialize_info(info),
    }


# MARK: Observation Plotting -------------------------------------------------------------------------------------
def _safe_name(value):
    """Return a filesystem-safe fragment for generated plot names."""
    if value is None:
        value = "raynet"
    text = str(value).strip() or "raynet"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "raynet"


def _observation_plot_config(episode):
    """Return normalized plot options from an episode config."""
    raw = (
        episode.get("observation_plots")
        if "observation_plots" in episode
        else episode.get("plot_observations")
    )
    if raw is None:
        raw = episode.get("observation_plot")
    if raw in (None, False):
        return None

    if isinstance(raw, dict):
        enabled = iniTools.as_bool(raw.get("enabled", True), True)
        options = dict(raw)
    else:
        enabled = iniTools.as_bool(raw, False)
        options = {}
    if not enabled:
        return None

    output_path = options.get("path") or options.get("output_path")
    output_dir = options.get("dir") or options.get("output_dir")
    if output_path:
        path = Path(str(output_path)).expanduser()
    else:
        if output_dir:
            directory = Path(str(output_dir)).expanduser()
        else:
            directory = Path(tempfile.gettempdir()) / "raynet_observation_plots"
        protocol = _safe_name(episode.get("protocol") or episode.get("label"))
        episode_id = _safe_name(episode.get("episode", "episode"))
        slot_id = _safe_name(episode.get("slot", "slot"))
        filename = options.get("filename")
        if filename:
            path = directory / str(filename)
        else:
            path = directory / f"{protocol}_ep{episode_id}_slot{slot_id}_observations.pdf"
    return {"path": path}


class ObservationTrace:
    """Collect per-flow observations and render one metric per PDF page."""

    def __init__(self, config):
        self.path = Path(config["path"]).expanduser()
        self.rows = []

    def record(self, time_s, observations):
        for agent_id, raw in sorted((observations or {}).items()):
            if not isinstance(raw, dict):
                continue
            for name, value in sorted(raw.items()):
                try:
                    y = float(value)
                except (TypeError, ValueError):
                    continue
                self.rows.append({
                    "time_s": float(time_s),
                    "agent_id": str(agent_id),
                    "name": str(name),
                    "value": y,
                })

    def write_pdf(self):
        if not self.rows:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except BaseException:
            return None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted({row["name"] for row in self.rows})
        agents = sorted({row["agent_id"] for row in self.rows})

        with PdfPages(self.path) as pdf:
            for name in names:
                fig, ax = plt.subplots(figsize=(11, 6))
                for agent_id in agents:
                    points = [
                        row for row in self.rows
                        if row["name"] == name and row["agent_id"] == agent_id
                    ]
                    if not points:
                        continue
                    points.sort(key=lambda row: row["time_s"])
                    ax.plot(
                        [row["time_s"] for row in points],
                        [row["value"] for row in points],
                        label=f"flow {agent_id}",
                        linewidth=1.2,
                    )
                ax.set_title(name)
                ax.set_xlabel("sim time (s)")
                ax.set_ylabel(name)
                ax.grid(True, alpha=0.25)
                if len(agents) > 1:
                    ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return str(self.path)


# MARK: IPC Serialization ---------------------------------------------------------------------------------------
def _send(writer, message):
    """Send one JSON message over the control socket."""
    writer.write(json.dumps(message, separators=(",", ":")) + "\n")
    writer.flush()


def _recv(reader):
    """Receive one JSON message from the control socket."""
    line = reader.readline()
    if not line:
        raise EOFError("control socket closed")
    return json.loads(line)


# MARK: Logging Control -------------------------------------------------------------------------------------------
def _silence_process_output():
    """Redirect stdout/stderr away from external orchestrator logs."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)


# MARK: INI Materialization -------------------------------------------------------------------------------------
def _create_ini_wrapper(episode):
    """Create an INI materializer for one Olympus-requested episode."""
    ini_path = Path(str(episode["ini_path"])).expanduser()
    ini_name = str(episode.get("protocol") or episode.get("label") or "raynet").lower()
    wrapper = experimentTools.create_experiment_wrapper(
        ini_path,
        episode,
        prefix=f"{ini_name}_",
    )

    # The caller owns semantic values. RayNet owns how they become INI/XML files.
    wrapper.add_replacements(episode.get("replacements") or episode.get("template_replacements"))
    wrapper.add_overrides(episode.get("overrides"))

    duration_s = None if episode.get("duration") is None else unitTools.to_seconds(episode.get("duration"))
    if duration_s is not None:
        wrapper.add_override("sim-time-limit", unitTools.format_seconds(duration_s))
    if iniTools.as_bool(episode.get("quiet"), True):
        wrapper.add_override("debug-on-errors", "false")
        wrapper.add_override("**.printDebugMessages", "false")
        wrapper.add_override("cmdenv-silent", "true")
    return wrapper


# MARK: Critical Runner Control ---------------------------------------------------------------------------------
def run(control_fd):
    """Own one OMNeT++ simulation and serve commands until it finishes."""
    # Create JSON-lines streams on the inherited control socket.
    sock = socket.socket(fileno=control_fd)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    runner = None
    ini_wrapper = None
    ini_workdir = None
    ini_variant = None
    observation_trace = None
    cleaned = False

    try:
        # Receive the episode request and create the concrete INI variant.
        message = _recv(reader)
        if message.get("type") != "start":
            raise ValueError(f"expected start message, got {message.get('type')!r}")
        episode = message.get("episode") or {}
        section = str(episode.get("section") or episode.get("config_section") or "General")
        observation_fields = episode.get("observation_fields") or episode.get("raw_observation_fields")
        plot_config = _observation_plot_config(episode)
        if plot_config is not None:
            observation_trace = ObservationTrace(plot_config)

        # Quiet runs must not leak simulator stdout/stderr into Olympus logs.
        if iniTools.as_bool(episode.get("quiet"), True):
            _silence_process_output()

        # Import OmnetBind inside the process that owns the simulator.
        from raynet.omnetBind import OmnetGymApi

        ini_wrapper = _create_ini_wrapper(episode)
        ini_workdir = tempfile.TemporaryDirectory(prefix="raynet_olympus_")
        ini_variant = str(ini_wrapper.materialize_ini(directory=ini_workdir.name))

        # Start the simulator and return its initial observations.
        runner = OmnetGymApi()
        runner.initialise(ini_variant, section)
        observations = runner.reset()
        serialized_observations = _serialize_observations(observations, observation_fields)
        if observation_trace is not None:
            observation_trace.record(runner.sim_time(), serialized_observations)
        _send(writer, {
            "type": "reset",
            "observations": serialized_observations,
            "info": _serialize_info({
                "simDone": False,
                "time_s": runner.sim_time(),
                "step_id": 0,
                "group_step": 0,
            }),
            "ini_path": ini_variant,
            "section": section,
        })

        # Process commands until Olympus closes the episode or RayNet finishes it.
        while True:
            message = _recv(reader)
            command = message.get("type")

            if command == "step":
                result = _step_message(
                    runner,
                    message.get("actions") or {},
                    observation_fields,
                )
                if observation_trace is not None:
                    observation_trace.record(
                        result["info"].get("time_s", runner.sim_time()),
                        result["observations"],
                    )
                terminateds = result["terminateds"]
                info = result["info"]
                # Manual shutdown (internal step limit). Shutdown + Cleanup.
                if terminateds.get("__all__", False):
                    path = (
                        observation_trace.write_pdf()
                        if observation_trace is not None else None
                    )
                    if path:
                        info["observation_plot"] = path
                    _send(writer, result)
                    runner.shutdown()
                    runner.cleanup()
                    cleaned = True
                    return 0
                # Simulation automatically shut down. Just cleanup.
                if info.get("simDone", False):
                    path = (
                        observation_trace.write_pdf()
                        if observation_trace is not None else None
                    )
                    if path:
                        info["observation_plot"] = path
                    _send(writer, result)
                    runner.cleanup()
                    cleaned = True
                    return 0
                _send(writer, result)
            # Manual shutdown (Olympus command)
            elif command == "close":
                path = (
                    observation_trace.write_pdf()
                    if observation_trace is not None else None
                )
                runner.shutdown()
                runner.cleanup()
                cleaned = True
                message = {"type": "closed"}
                if path:
                    message["observation_plot"] = path
                _send(writer, message)
                return 0
            else:
                raise ValueError(f"unknown command: {command!r}")

    except BaseException as exc:
        # Return structured errors to Olympus before exiting.
        try:
            _send(writer, {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        except BaseException:
            pass
        return 1

    finally:
        # Clean up unfinished simulations and always close IPC resources.
        if runner is not None and not cleaned:
            try:
                runner.shutdown()
            except BaseException:
                pass
            try:
                runner.cleanup()
            except BaseException:
                pass
        if ini_wrapper is not None:
            ini_wrapper.cleanup()
        if ini_workdir is not None:
            ini_workdir.cleanup()
        try:
            reader.close()
            writer.close()
            sock.close()
        except OSError:
            pass


# MARK: CLI Entry Point -----------------------------------------------------------------------------------------
def main():
    """Parse CLI arguments and run the control loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    args = parser.parse_args()
    return run(args.control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
