"""
A universal RayNet runner script that works on any provided protocol, config, and section.
- Bootstraps RayNet environment from build_paths.sh
- Resolves RayNet's favored venv dynamically
- Runs selected RL protocol using its registered runner script
"""

import os
import sys
import subprocess
from pathlib import Path

# ------------------------------------------------------------
# Locate project root (relative to this script, as we don't have access to environment variables if called externally)
# ------------------------------------------------------------
RAYNET_PATH = Path(__file__).resolve().parents[2]
BUILD_PATHS = RAYNET_PATH / "build_paths.sh"

# ------------------------------------------------------------
# Protocol registry
# ------------------------------------------------------------
runner_paths = {
    "orca": str(RAYNET_PATH / "simlibs/Orca/src/OrcaEval.py"),
    "cubic": str(RAYNET_PATH / "simlibs/Orca/src/CubicEval.py"),
    "astrea": str(RAYNET_PATH / "simlibs/Astrea/src/AstreaEval.py"),
    "cleanslate": str(RAYNET_PATH / "simlibs/CleanSlate/src/CleanSlateEval.py"),
}

# ------------------------------------------------------------
# Load and return a RayNet environment from build_paths.sh
# ------------------------------------------------------------
def load_raynet_env():
    # Source from build_paths.sh and output environment variables into result
    cmd = f'''
    set -a
    source "{BUILD_PATHS}"
    env
    '''
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        check=True
    )

    # Convert env printout into a dict
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    
    return env

# Merge with current environment
env = os.environ.copy()
env.update(load_raynet_env())


# ------------------------------------------------------------
# Resolve venv + python
# ------------------------------------------------------------
VENV_PATH = Path(env["RAYNET_VENV_PATH"])
PYTHON_BIN = str(VENV_PATH / "bin" / "python")

def get_registered_protocols():
    return list(runner_paths.keys())

def is_protocol_registered(protocol: str) -> bool:
    return protocol in runner_paths


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def run_simulation(protocol: str, ini_path: str, section: str = "General"):
    if not is_protocol_registered(protocol):
        print(
            f"Error: Unknown protocol '{protocol}'. "
            f"Available: {get_registered_protocols()}"
        )
        sys.exit(1)
    runner = runner_paths[protocol]
    
    print(f"RayNet: Running protocol {protocol}...")
    print("-------------------------------------------------------------------")
    print(f"\tPython: \t {PYTHON_BIN}")
    print(f"\tRunner: \t {runner}")
    print(f"\tConfig: \t {ini_path}")
    print(f"\tSection:\t {section}")
    print("-------------------------------------------------------------------")

    # Run the final combined simulation command
    subprocess.run(
        [PYTHON_BIN, runner, ini_path, section],
        env=env,
        check=True
    )


# ------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python raynet_runner.py <protocol> <ini_path> <section>")
        sys.exit(1)
        
    protocol = sys.argv[1]
    ini_path = sys.argv[2]
    section = sys.argv[3] if len(sys.argv) > 3 else "General"
    
    run_simulation(protocol, ini_path, section)