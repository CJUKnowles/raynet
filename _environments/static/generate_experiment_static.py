#!/usr/bin/env python3
"""Generate a concrete INI for the static environment."""

import sys
from pathlib import Path

RAYNET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAYNET_ROOT))

from raynet import experimentTools


def main():
    """Parse a concrete episode config and write a materialized INI."""
    base = Path(__file__).resolve().parent
    experimentTools.experiment_cli(
        default_template=base / "config_static.ini",
        default_output=base / "generated_static.ini",
    )


if __name__ == "__main__":
    main()
