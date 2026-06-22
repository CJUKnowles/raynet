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
import sys
import tempfile
import traceback
from pathlib import Path

# Make OmnetBind importable before the runner receives any commands.
RAYNET_PATH = Path(os.environ.get("RAYNET_PATH", "/home/james/raynet")).expanduser()
BUILD_PATH = RAYNET_PATH / "build"
if str(BUILD_PATH) not in sys.path:
    sys.path.insert(0, str(BUILD_PATH))

# MARK: Simulator Step Handling ---------------------------------------------------------------------------------
def _step_message(runner, actions):
    """Advance the simulator and format one step response."""
    observations, rewards, terminateds, info = runner.step(actions or {})
    return {
        "type": "step",
        "observations": _serialize_observations(observations),
        "rewards": _float_dict(rewards),
        "terminateds": _bool_dict(terminateds),
        "info": _bool_dict(info),
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


def _serialize_observations(observations):
    """Convert OmnetBind observations into serializable Python lists."""
    out = {}

    # Convert every returned agent observation independently.
    for agent_id, observation in (observations or {}).items():
        if hasattr(observation, "to_list"):
            values = observation.to_list()
        else:
            values = list(observation)
        out[str(agent_id)] = [float(value) for value in values]

    return out


def _float_dict(values):
    return {str(key): float(value) for key, value in (values or {}).items()}


def _bool_dict(values):
    return {str(key): bool(value) for key, value in (values or {}).items()}


# MARK: INI Materialization -------------------------------------------------------------------------------------
def _materialize_ini(episode):
    """Create a concrete INI variant for one Olympus-requested episode."""
    # Validate the requested protocol and source INI.
    ini_path = Path(str(episode["ini_path"])).expanduser()
    section = str(episode.get("section") or episode.get("config_section") or "General")
    protocol = str(episode.get("protocol", "orca")).lower()
    if protocol != "orca":
        raise ValueError(f"RayNet Olympus runner v1 supports protocol=orca, got {protocol!r}")
    if not ini_path.exists():
        raise FileNotFoundError(f"ini_path not found: {ini_path}")

    # Read the scenario parameters used by RayNet's Orca INI template.
    bw_mbps = float(episode.get("bw", episode.get("bandwidth", 100.0)))
    rtt_ms = float(episode.get("delay", episode.get("base_rtt_ms", 20.0)))
    interval_s = float(episode.get("interval_s", episode.get("fixed_interval_s", 0.02)))
    duration_s = episode.get("duration")
    buffer_bits = float(episode.get(
        "buffer_bits",
        episode.get("buffer_size_bits", _static_buffer_bits(bw_mbps, rtt_ms)),
    ))

    # Replace the common template tokens used in existing RayNet configs.
    text = ini_path.read_text()
    replacements = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "RAYNET_PATH": str(RAYNET_PATH),
        "ORCA_BOTTLENECK_BW": _format_mbps(bw_mbps),
        # RayNet bottleneckDelay is one-way delay; Olympus delay is base RTT.
        "ORCA_BASE_RTT": _format_ms(rtt_ms / 2.0),
        "ORCA_BOTTLENECK_BUFFER_SIZE": _format_bits(buffer_bits),
        "MAX_RL_STEPS": _max_steps(duration_s, interval_s),
    }
    for key, value in (episode.get("replacements") or {}).items():
        replacements[str(key)] = str(value)
    for key, value in replacements.items():
        if value:
            text = text.replace(key, value)

    # Apply bridge-owned overrides after template replacement.
    text = _apply_duration_limit(text, duration_s)
    text = _apply_quiet_mode(text, _as_bool(episode.get("quiet"), True))

    # Store the generated INI in RayNet's output directory when available.
    tmp_dir = RAYNET_PATH / "out"
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"olympus_{protocol}_",
        suffix=".ini",
        dir=str(tmp_dir) if tmp_dir.exists() else None,
        delete=False,
    )
    try:
        tmp.write(text)
        return tmp.name, section
    finally:
        tmp.close()

def _apply_duration_limit(text, duration_s):
    """Insert or replace the OMNeT++ sim-time-limit in generated INI text."""
    if duration_s is None:
        return text

    line = f"sim-time-limit = {float(duration_s):.12g}s"
    if "sim-time-limit" in text:
        lines = []
        for item in text.splitlines():
            if item.strip().startswith("sim-time-limit"):
                lines.append(line)
            else:
                lines.append(item)
        return "\n".join(lines) + "\n"

    return text.rstrip() + "\n\n" + line + "\n"

def _apply_quiet_mode(text, quiet):
    """Disable RayNet Orca debug prints in generated INI text when requested."""
    if not quiet:
        return text

    # Replace the known debug toggles used by the Orca training template.
    replacements = {
        "**.printDebugMessages = true": "**.printDebugMessages = false",
        "**.printDebugMessages=true": "**.printDebugMessages=false",
        "debug-on-errors = true": "debug-on-errors = false",
        "debug-on-errors=true": "debug-on-errors=false",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)

    return text

def _static_buffer_bits(bw_mbps, rtt_ms):
    """Return one BDP worth of buffering in bits."""
    return max(1.0, float(bw_mbps) * float(rtt_ms) * 1000.0)

def _max_steps(duration_s, interval_s):
    """Convert an episode duration into an approximate RL step count."""
    if duration_s is None:
        return ""
    interval_s = max(float(interval_s or 0.02), 1e-9)
    return str(max(1, int(round(float(duration_s) / interval_s))))

def _format_ms(value):
    return f"{float(value):.12g}ms"

def _format_mbps(value):
    return f"{float(value):.12g}Mbps"

def _format_bits(value):
    return f"{float(value):.12g}b"

def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


# MARK: Critical Runner Control ---------------------------------------------------------------------------------
def run(control_fd):
    """Own one OMNeT++ simulation and serve commands until it finishes."""
    # Import OmnetBind inside the process that owns the simulator.
    from omnetbind import OmnetGymApi

    # Create JSON-lines streams on the inherited control socket.
    sock = socket.socket(fileno=control_fd)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    runner = None
    ini_variant = None
    cleaned = False

    try:
        # Receive the episode request and create the concrete INI variant.
        message = _recv(reader)
        if message.get("type") != "start":
            raise ValueError(f"expected start message, got {message.get('type')!r}")
        episode = message.get("episode") or {}
        ini_variant, section = _materialize_ini(episode)

        # Start the simulator and return its initial observations.
        runner = OmnetGymApi()
        runner.initialise(ini_variant, section)
        observations = runner.reset()
        _send(writer, {
            "type": "reset",
            "observations": _serialize_observations(observations),
            "ini_path": ini_variant,
            "section": section,
        })

        # Process commands until Olympus closes the episode or RayNet finishes it.
        while True:
            message = _recv(reader)
            command = message.get("type")

            if command == "step":
                result = _step_message(runner, message.get("actions") or {})
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
        if ini_variant:
            try:
                os.unlink(ini_variant)
            except OSError:
                pass
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
