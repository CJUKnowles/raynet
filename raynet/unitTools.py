"""Unit parsing and formatting helpers for RayNet scripts."""

from __future__ import annotations

import re


_QUANTITY_RE = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z/]+)?\s*$")


def parse_quantity(value):
    """Return ``(number, unit)`` for strings such as ``100Mbps`` or ``500b``."""
    if isinstance(value, (int, float)):
        return float(value), ""
    match = _QUANTITY_RE.match(str(value))
    if not match:
        raise ValueError(f"Invalid unit quantity: {value!r}")
    return float(match.group(1)), match.group(2) or ""


def format_quantity(value, unit=""):
    """Format a numeric value with a unit suffix."""
    return f"{float(value):.12g}{unit}"


def format_seconds(value):
    return format_quantity(value, "s")


def format_ms(value):
    return format_quantity(value, "ms")


def format_bits(value):
    return format_quantity(value, "b")


def format_bytes(value):
    return format_quantity(value, "B")


def format_mbps(value):
    return format_quantity(value, "Mbps")


def to_bits(value):
    """Convert a bit/byte quantity into bits."""
    number, unit = parse_quantity(value)
    if unit in {"B", "KB", "MB", "GB"}:
        return number * {"B": 8.0, "KB": 8e3, "MB": 8e6, "GB": 8e9}[unit]
    normalized = unit.lower()
    multipliers = {
        "": 1.0,
        "b": 1.0,
        "bit": 1.0,
        "bits": 1.0,
        "kb": 1e3,
        "kbit": 1e3,
        "kbits": 1e3,
        "mb": 1e6,
        "mbit": 1e6,
        "mbits": 1e6,
        "gb": 1e9,
        "gbit": 1e9,
        "gbits": 1e9,
        "byte": 8.0,
        "bytes": 8.0,
    }
    if normalized not in multipliers:
        raise ValueError(f"Unsupported bit quantity unit: {unit!r}")
    return number * multipliers[normalized]


def to_mbps(value):
    """Convert a data-rate quantity into Mbps."""
    number, unit = parse_quantity(value)
    normalized = unit.lower()
    multipliers = {
        "": 1.0,
        "mbps": 1.0,
        "mbit/s": 1.0,
        "kbps": 1e-3,
        "kbit/s": 1e-3,
        "bps": 1e-6,
        "bit/s": 1e-6,
        "gbps": 1e3,
        "gbit/s": 1e3,
    }
    if normalized not in multipliers:
        raise ValueError(f"Unsupported data-rate unit: {unit!r}")
    return number * multipliers[normalized]


def to_ms(value):
    """Convert a duration quantity into milliseconds."""
    number, unit = parse_quantity(value)
    normalized = unit.lower()
    multipliers = {
        "": 1.0,
        "ms": 1.0,
        "s": 1e3,
        "sec": 1e3,
        "secs": 1e3,
        "second": 1e3,
        "seconds": 1e3,
        "us": 1e-3,
    }
    if normalized not in multipliers:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return number * multipliers[normalized]


def to_seconds(value):
    """Convert a duration quantity into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    number, unit = parse_quantity(value)
    normalized = unit.lower()
    multipliers = {
        "": 1.0,
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
    }
    if normalized not in multipliers:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return number * multipliers[normalized]


def bdp_bits(bw_mbps, rtt_ms):
    """Return one bandwidth-delay product worth of buffering in bits."""
    return max(1.0, float(bw_mbps) * float(rtt_ms) * 1000.0)
