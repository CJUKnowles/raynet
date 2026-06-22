"""Import access for RayNet's native OMNeT++ Python binding.

The compiled pybind11 module is produced by CMake in ``$RAYNET_PATH/build``.
This wrapper centralizes the path setup so scripts can import it through the
installed ``raynet`` package instead of editing ``sys.path`` locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def raynet_path() -> Path:
    """Return the configured RayNet repository path."""
    return Path(os.environ.get("RAYNET_PATH", Path(__file__).resolve().parents[1])).expanduser()


def build_path() -> Path:
    """Return the directory containing the compiled native binding."""
    return raynet_path() / "build"


def _ensure_native_binding_path() -> None:
    native_build_path = str(build_path())
    if native_build_path not in sys.path:
        sys.path.insert(0, native_build_path)


_ensure_native_binding_path()

try:
    from omnetbind import OmnetGymApi
except ImportError as exc:
    raise ImportError(
        "Could not import RayNet's native omnetbind module. Run './build.sh' "
        "from the RayNet repository and make sure '.venv' is activated."
    ) from exc


__all__ = ["OmnetGymApi", "build_path", "raynet_path"]
