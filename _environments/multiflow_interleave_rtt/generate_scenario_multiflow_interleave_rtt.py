#!/usr/bin/env python3
"""Generate a Dumbbell link-change scenario for multiflow_interleave_rtt."""

import sys
from pathlib import Path

RAYNET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAYNET_ROOT))

from raynet import experimentTools


def main():
    """Parse a concrete episode config and write its scenario XML."""
    default_output = Path(__file__).with_name("scenario.xml")
    experimentTools.scenario_cli(default_output)


if __name__ == "__main__":
    main()
