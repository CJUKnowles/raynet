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


RAYNET_PATH = Path(os.environ.get("RAYNET_PATH", "/home/james/raynet")).expanduser()
BUILD_PATH = RAYNET_PATH / "build"
if str(BUILD_PATH) not in sys.path:
    sys.path.insert(0, str(BUILD_PATH))


def _send(writer, message):
    writer.write(json.dumps(message, separators=(",", ":")) + "\n")
    writer.flush()


def _recv(reader):
    line = reader.readline()
    if not line:
        raise EOFError("control socket closed")
    return json.loads(line)


def _serialize_observations(observations):
    out = {}
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


def _static_buffer_bits(bw_mbps, rtt_ms):
    # BDP: Mbps * ms * 1000 = bits.
    return max(1.0, float(bw_mbps) * float(rtt_ms) * 1000.0)


def _max_steps(duration_s, interval_s):
    if duration_s is None:
        return ""
    interval_s = max(float(interval_s or 0.02), 1e-9)
    return str(max(1, int(round(float(duration_s) / interval_s))))


def _apply_duration_limit(text, duration_s):
    if duration_s is None:
        return text
    line = f"sim-time-limit = {float(duration_s):.12g}s"
    if "sim-time-limit" in text:
        lines = [
            line if item.strip().startswith("sim-time-limit") else item
            for item in text.splitlines()
        ]
        return "\n".join(lines) + "\n"
    return text.rstrip() + "\n\n" + line + "\n"


def _apply_quiet_mode(text, quiet):
    if not quiet:
        return text
    replacements = {
        "**.printDebugMessages = true": "**.printDebugMessages = false",
        "**.printDebugMessages=true": "**.printDebugMessages=false",
        "debug-on-errors = true": "debug-on-errors = false",
        "debug-on-errors=true": "debug-on-errors=false",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def _materialize_ini(episode):
    ini_path = Path(str(episode["ini_path"])).expanduser()
    section = str(episode.get("section") or episode.get("config_section") or "General")
    protocol = str(episode.get("protocol", "orca")).lower()
    if protocol != "orca":
        raise ValueError(f"RayNet Olympus runner v1 supports protocol=orca, got {protocol!r}")
    if not ini_path.exists():
        raise FileNotFoundError(f"ini_path not found: {ini_path}")

    bw_mbps = float(episode.get("bw", episode.get("bandwidth", 100.0)))
    rtt_ms = float(episode.get("delay", episode.get("base_rtt_ms", 20.0)))
    interval_s = float(episode.get("interval_s", episode.get("fixed_interval_s", 0.02)))
    duration_s = episode.get("duration")
    buffer_bits = float(episode.get(
        "buffer_bits",
        episode.get("buffer_size_bits", _static_buffer_bits(bw_mbps, rtt_ms)),
    ))

    text = ini_path.read_text()
    replacements = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "RAYNET_PATH": str(RAYNET_PATH),
        "ORCA_BOTTLENECK_BW": _format_mbps(bw_mbps),
        # RayNet dumbbell bottleneckDelay is one-way delay; Olympus scenario
        # delay is the base RTT used by the training config.
        "ORCA_BASE_RTT": _format_ms(rtt_ms / 2.0),
        "ORCA_BOTTLENECK_BUFFER_SIZE": _format_bits(buffer_bits),
        "MAX_RL_STEPS": _max_steps(duration_s, interval_s),
    }
    for key, value in (episode.get("replacements") or {}).items():
        replacements[str(key)] = str(value)
    for key, value in replacements.items():
        if value:
            text = text.replace(key, value)
    text = _apply_duration_limit(text, duration_s)
    text = _apply_quiet_mode(text, _as_bool(episode.get("quiet"), True))

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


def _step_message(runner, actions):
    observations, rewards, terminateds, info = runner.step(actions or {})
    return {
        "type": "step",
        "observations": _serialize_observations(observations),
        "rewards": _float_dict(rewards),
        "terminateds": _bool_dict(terminateds),
        "info": _bool_dict(info),
    }


def run(control_fd):
    from omnetbind import OmnetGymApi

    sock = socket.socket(fileno=control_fd)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    runner = None
    ini_variant = None
    cleaned = False

    try:
        message = _recv(reader)
        if message.get("type") != "start":
            raise ValueError(f"expected start message, got {message.get('type')!r}")
        episode = message.get("episode") or {}
        ini_variant, section = _materialize_ini(episode)

        runner = OmnetGymApi()
        runner.initialise(ini_variant, section)
        observations = runner.reset()
        _send(writer, {
            "type": "reset",
            "observations": _serialize_observations(observations),
            "ini_path": ini_variant,
            "section": section,
        })

        while True:
            message = _recv(reader)
            command = message.get("type")
            if command == "step":
                result = _step_message(runner, message.get("actions") or {})
                _send(writer, result)
                terminateds = result["terminateds"]
                info = result["info"]
                if terminateds.get("__all__", False):
                    runner.shutdown()
                    runner.cleanup()
                    cleaned = True
                    return 0
                if info.get("simDone", False):
                    runner.cleanup()
                    cleaned = True
                    return 0
            elif command == "close":
                runner.shutdown()
                runner.cleanup()
                cleaned = True
                _send(writer, {"type": "closed"})
                return 0
            else:
                raise ValueError(f"unknown command: {command!r}")
    except BaseException as exc:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    args = parser.parse_args()
    return run(args.control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
