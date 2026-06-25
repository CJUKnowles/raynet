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
    return serialized


def _step_message(runner, actions, observation_fields):
    """Advance the simulator and format one step response."""
    observations, rewards, terminateds, info = runner.step(actions or {})
    return {
        "type": "step",
        "observations": _serialize_observations(observations, observation_fields),
        "rewards": obsTools.float_dict(rewards),
        "terminateds": obsTools.bool_dict(terminateds),
        "info": _serialize_info(info),
    }


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
    cleaned = False

    try:
        # Receive the episode request and create the concrete INI variant.
        message = _recv(reader)
        if message.get("type") != "start":
            raise ValueError(f"expected start message, got {message.get('type')!r}")
        episode = message.get("episode") or {}
        section = str(episode.get("section") or episode.get("config_section") or "General")
        observation_fields = episode.get("observation_fields") or episode.get("raw_observation_fields")

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
        _send(writer, {
            "type": "reset",
            "observations": _serialize_observations(observations, observation_fields),
            "info": _serialize_info({
                "simDone": False,
                "time_s": runner.sim_time(),
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
                _send(writer, result)
                terminateds = result["terminateds"]
                info = result["info"]
                # Manual shutdown (internal step limit). Shutdown + Cleanup.
                if terminateds.get("__all__", False):
                    runner.shutdown()
                    runner.cleanup()
                    cleaned = True
                    return 0
                # Simulation automatically shut down. Just cleanup.
                if info.get("simDone", False):
                    runner.cleanup()
                    cleaned = True
                    return 0
            # Manual shutdown (Olympus command)
            elif command == "close":
                runner.shutdown()
                runner.cleanup()
                cleaned = True
                _send(writer, {"type": "closed"})
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
