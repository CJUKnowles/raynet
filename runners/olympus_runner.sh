#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYNET_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
export RAYNET_PATH

source "${RAYNET_PATH}/.venv/bin/activate"

if [[ -n "${OMNET_PATH:-}" && -f "${OMNET_PATH}/setenv" ]]; then
    source "${OMNET_PATH}/setenv"
fi

exec "${RAYNET_PATH}/.venv/bin/python" "${RAYNET_PATH}/runners/olympus_runner.py" "$@"
