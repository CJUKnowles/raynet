"""Experiment generation helpers for RayNet environments."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree

from raynet import iniTools, unitTools


# MARK: Config Loading -----------------------------------------------------------------------------------------
def load_config(path):
    """Load one YAML or JSON experiment config file."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML experiment configs") from exc
    return yaml.safe_load(text) or {}


def load_episode_config(path=None, *, inline_json=None):
    """Load a concrete per-run config from a file or an inline JSON string."""
    if inline_json:
        return json.loads(inline_json)
    if path:
        return load_config(path)
    return {}


# MARK: Unit Normalization -------------------------------------------------------------------------------------
def as_mbps(value):
    """Return a bandwidth value in Mbps."""
    return unitTools.to_mbps(value)


def as_ms(value):
    """Return a time value in milliseconds."""
    return unitTools.to_ms(value)


def as_seconds(value):
    """Return a time value in seconds."""
    return unitTools.to_seconds(value)


def format_mbps(value):
    return unitTools.format_mbps(as_mbps(value))


def format_rtt_as_link_delay(value):
    """Convert an RTT value into the one-way bottleneck delay used by Dumbbell."""
    return unitTools.format_ms(as_ms(value) / 2.0)


def format_ms(value):
    return unitTools.format_ms(as_ms(value))


def format_seconds(value):
    return unitTools.format_seconds(as_seconds(value))


def qsize_bits(bw, delay, bdp_mult=1.0):
    """Return a bottleneck queue size in bits from Mbps, RTT ms, and BDP multiplier."""
    return max(1.0, as_mbps(bw) * as_ms(delay) * 1000.0 * float(bdp_mult))


# MARK: Link Schedule Handling ---------------------------------------------------------------------------------
def resolve_link_schedule(schedule, *, bw, delay):
    """Convert Olympus-style link updates into absolute bw/delay events."""
    resolved = []
    current_bw = as_mbps(bw)
    current_delay = as_ms(delay)
    base_bw = current_bw
    base_delay = current_delay

    for raw_event in schedule or []:
        event = dict(raw_event or {})
        event_time = as_seconds(event["t"])

        if "bw" in event:
            current_bw = as_mbps(event["bw"])
        elif "bw_frac" in event:
            current_bw = base_bw * float(event["bw_frac"])

        if "delay" in event:
            current_delay = as_ms(event["delay"])
        elif "delay_frac" in event:
            current_delay = base_delay * float(event["delay_frac"])

        resolved.append({
            "t": event_time,
            "bw": current_bw,
            "delay": current_delay,
        })

    return resolved


# MARK: Dumbbell Scenario Generation ---------------------------------------------------------------------------
def _set_channel(at_elem, *, module, gate, param, value):
    ElementTree.SubElement(
        at_elem,
        "set-channel-param",
        {
            "src-module": module,
            "src-gate": gate,
            "par": param,
            "value": value,
        },
    )


def _add_dumbbell_link_event(root, *, event_time, bw, delay, per_flow_delays=None):
    at_elem = ElementTree.SubElement(root, "at", {"t": format_seconds(event_time)})

    # Forward data traffic is shaped on the router1 -> router2 bottleneck.
    _set_channel(
        at_elem,
        module="router1",
        gate="pppg$o[0]",
        param="datarate",
        value=format_mbps(bw),
    )

    # Standard envs split RTT across the bottleneck. Inter-RTT envs keep the
    # bottleneck and sender access links at 0 delay, then put each flow's
    # one-way forward-path delay on router2 -> server[i]. In Dumbbell.ned,
    # router2.pppg[0] is the bottleneck and server links are allocated after it.
    link_delay = "0ms" if per_flow_delays else format_rtt_as_link_delay(delay)
    for module in ("router1", "router2"):
        _set_channel(
            at_elem,
            module=module,
            gate="pppg$o[0]",
            param="delay",
            value=link_delay,
        )

    if per_flow_delays:
        for flow_id, flow_delay in enumerate(per_flow_delays):
            _set_channel(
                at_elem,
                module="router2",
                gate=f"pppg$o[{flow_id + 1}]",
                param="delay",
                value=format_ms(flow_delay),
            )


def build_dumbbell_scenario(*, bw, delay, link_schedule=None, include_initial=True,
                            per_flow_delays=None):
    """Build a scenario tree for immediate Dumbbell bottleneck reconfiguration."""
    root = ElementTree.Element("scenario")

    if include_initial:
        _add_dumbbell_link_event(
            root,
            event_time=0,
            bw=bw,
            delay=delay,
            per_flow_delays=per_flow_delays,
        )

    for event in resolve_link_schedule(link_schedule, bw=bw, delay=delay):
        _add_dumbbell_link_event(
            root,
            event_time=event["t"],
            bw=event["bw"],
            delay=event["delay"],
            per_flow_delays=per_flow_delays,
        )

    ElementTree.indent(root, space="    ")
    return ElementTree.ElementTree(root)


def write_dumbbell_scenario(path, *, bw, delay, link_schedule=None, include_initial=True,
                            per_flow_delays=None):
    """Write a Dumbbell scenario XML file and return the path."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_dumbbell_scenario(
        bw=bw,
        delay=delay,
        link_schedule=link_schedule,
        include_initial=include_initial,
        per_flow_delays=per_flow_delays,
    )
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return path


# MARK: Flow Timing --------------------------------------------------------------------------------------------
def _list_or_none(values):
    if values is None:
        return None
    return list(values)


def flow_timing_overrides(*, flows, start_delays=None, flow_durations=None):
    """Create per-client start/stop overrides for staggered RayNet flows."""
    overrides = {}
    start_delays = _list_or_none(start_delays) or [0.0] * int(flows)
    flow_durations = _list_or_none(flow_durations)

    for flow_id in range(int(flows)):
        if flow_id < len(start_delays):
            start = as_seconds(start_delays[flow_id])
        else:
            start = 0.0
        prefix = f"**.client[{flow_id}].app[*]"
        overrides[f"{prefix}.tOpen"] = format_seconds(start)
        overrides[f"{prefix}.tSend"] = format_seconds(start)

        if flow_durations and flow_id < len(flow_durations):
            close_time = start + as_seconds(flow_durations[flow_id])
            overrides[f"{prefix}.tClose"] = format_seconds(close_time)

    return overrides


# MARK: INI Materialization ------------------------------------------------------------------------------------
class ExperimentWrapper:
    """Own generated scenario files around an IniWrapper."""

    def __init__(self, ini_wrapper, episode):
        self.ini_wrapper = ini_wrapper
        self.episode = dict(episode or {})
        self.generated_paths = []

    def add_replacements(self, replacements=None, **kwargs):
        self.ini_wrapper.add_replacements(replacements, **kwargs)
        return self

    def add_overrides(self, overrides=None, **kwargs):
        self.ini_wrapper.add_overrides(overrides, **kwargs)
        return self

    def add_override(self, key, value):
        self.ini_wrapper.add_override(key, value)
        return self

    def add_replacement(self, key, value):
        self.ini_wrapper.add_replacement(key, value)
        return self

    def materialize_ini(self, *, output_path=None, directory=None, prefix=None, suffix=None):
        """Generate scenario files first, then render the INI."""
        target_dir = self._materialization_dir(output_path, directory)
        self._apply_standard_episode_values(target_dir)
        return self.ini_wrapper.materialize_ini(
            output_path=output_path,
            directory=directory,
            prefix=prefix,
            suffix=suffix,
        )

    def cleanup(self):
        """Delete generated INI and scenario files."""
        self.ini_wrapper.cleanup()
        for path in self.generated_paths:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
        self.generated_paths = []

    def _materialization_dir(self, output_path, directory):
        if output_path is not None:
            return Path(output_path).expanduser().parent
        if directory is not None:
            return Path(directory).expanduser()
        if self.ini_wrapper.directory is not None:
            return self.ini_wrapper.directory
        return self.ini_wrapper.template_path.parent / "ini_variants"

    def _apply_standard_episode_values(self, target_dir):
        episode = self.episode
        bw = episode.get("bw")
        delay = episode.get("delay")
        flows = int(episode.get("flows", episode.get("n", 1)))
        bdp_mult = episode.get("bdp_mult", 1.0)
        per_flow_delays = episode.get("per_flow_delays")

        replacements = {
            "home": episode.get("home"),
            "raynet_path": episode.get("raynet_path"),
            "protocol": episode.get("protocol"),
            "flows": flows,
        }
        if bw is not None:
            replacements["bw"] = format_mbps(bw)
        if delay is not None:
            if per_flow_delays:
                replacements["delay"] = "0ms"
            else:
                replacements["delay"] = format_rtt_as_link_delay(delay)
        if episode.get("qsize") is not None:
            replacements["qsize"] = str(episode["qsize"])
        elif bw is not None and delay is not None:
            replacements["qsize"] = unitTools.format_bits(
                qsize_bits(bw, delay, bdp_mult)
            )
        if episode.get("duration") is not None:
            replacements["duration"] = as_seconds(episode["duration"])

        scenario_path = self._scenario_path(target_dir)
        if scenario_path:
            replacements["scenario"] = scenario_path

        self.ini_wrapper.add_replacements(replacements)
        self.ini_wrapper.overrides.setdefault("**.numberOfFlows", str(flows))
        if episode.get("duration") is not None:
            self.ini_wrapper.overrides.setdefault(
                "sim-time-limit",
                format_seconds(episode["duration"]),
            )
        self.ini_wrapper.add_overrides(flow_timing_overrides(
            flows=flows,
            start_delays=episode.get("start_delays"),
            flow_durations=episode.get("flow_durations"),
        ))

    def _scenario_path(self, target_dir):
        episode = self.episode
        per_flow_delays = episode.get("per_flow_delays")
        has_schedule = "link_schedule" in episode or "link_schedules" in episode
        has_per_flow_delays = bool(per_flow_delays)
        schedule = episode.get("link_schedule")
        if schedule is None:
            schedule = episode.get("link_schedules")
        existing = episode.get("scenario_path") or episode.get("scenario")
        if existing and not has_schedule and not has_per_flow_delays:
            return str(Path(existing).expanduser())
        if not has_schedule and not has_per_flow_delays:
            return ""

        if episode.get("bw") is None or episode.get("delay") is None:
            raise ValueError("bw and delay are required to generate a RayNet scenario")

        target_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix="scenario_",
            suffix=".xml",
            dir=str(target_dir),
            delete=False,
        )
        handle.close()
        path = Path(handle.name)
        write_dumbbell_scenario(
            path,
            bw=episode["bw"],
            delay=episode["delay"],
            link_schedule=schedule,
            per_flow_delays=per_flow_delays,
        )
        self.generated_paths.append(path)
        return str(path)


def create_experiment_wrapper(template_path, episode, *, prefix=None):
    """Create an INI wrapper that can also generate RayNet-owned scenario files."""
    ini_wrapper = iniTools.IniWrapper(template_path=template_path, prefix=prefix)
    return ExperimentWrapper(ini_wrapper, episode)


def materialize_experiment(template_path, output_path, episode, *, prefix=None):
    """Generate one concrete INI from a RayNet episode config."""
    wrapper = create_experiment_wrapper(template_path, episode, prefix=prefix)
    return wrapper.materialize_ini(output_path=output_path)


# MARK: CLI Helpers --------------------------------------------------------------------------------------------
def add_episode_cli_args(parser):
    """Add shared concrete-episode CLI arguments."""
    parser.add_argument("--config", help="YAML or JSON concrete episode config")
    parser.add_argument("--episode-json", help="Inline JSON concrete episode config")
    parser.add_argument("--bw", help="Base bottleneck bandwidth, default unit Mbps")
    parser.add_argument("--delay", help="Base bottleneck RTT, default unit ms")
    parser.add_argument("--duration", help="Episode duration, default unit seconds")
    parser.add_argument("--flows", type=int, help="Number of client/server flows")
    parser.add_argument("--protocol", help="RayNet TCP protocol name")
    parser.add_argument("--bdp-mult", type=float, help="Queue size in BDPs")
    parser.add_argument("--link-schedule-json", help="Inline JSON link schedule")
    parser.add_argument("--per-flow-delays-json", help="Inline JSON per-flow delays")


def episode_from_args(args):
    """Merge file/JSON CLI data with explicit command-line fields."""
    episode = load_episode_config(args.config, inline_json=args.episode_json)
    updates = {
        "bw": args.bw,
        "delay": args.delay,
        "duration": args.duration,
        "flows": args.flows,
        "protocol": args.protocol,
        "bdp_mult": args.bdp_mult,
    }
    episode.update({key: value for key, value in updates.items() if value is not None})
    if args.link_schedule_json:
        episode["link_schedule"] = json.loads(args.link_schedule_json)
    if args.per_flow_delays_json:
        episode["per_flow_delays"] = json.loads(args.per_flow_delays_json)
    return episode


def scenario_cli(default_output):
    """Run a standard Dumbbell scenario generator CLI."""
    parser = argparse.ArgumentParser()
    add_episode_cli_args(parser)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()

    episode = episode_from_args(args)
    write_dumbbell_scenario(
        args.output,
        bw=episode["bw"],
        delay=episode["delay"],
        link_schedule=episode.get("link_schedule"),
        per_flow_delays=episode.get("per_flow_delays"),
    )
    print(Path(args.output).expanduser())


def experiment_cli(*, default_template, default_output):
    """Run a standard RayNet INI/scenario materializer CLI."""
    parser = argparse.ArgumentParser()
    add_episode_cli_args(parser)
    parser.add_argument("--template", default=default_template)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()

    episode = episode_from_args(args)
    path = materialize_experiment(args.template, args.output, episode)
    print(path)
