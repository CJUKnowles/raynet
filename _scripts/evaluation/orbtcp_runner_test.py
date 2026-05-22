"""
This script is a demonstration of how to run RayNet on one of the experiments from orbtcpExperiments.
- There is a specific python version provided, but it shouldn't matter so long as the environment used has ray/rllib and other critical libraries
- The runner script is the one associated with the protocol being evaluated (e.g. OrcaEval.py for Orca)
- An ini_path and section name must be provided to specify what to run
- Each runner script points to a specific checkpoint (like in raynet/_models) that is used for inference. 
- Cubic is only included here becuase it was easier to just run Orca with actions disabled for my experiment setup.
"""

import os

runner_paths = {
    "Orca": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/OrcaEval.py",
    "Cubic": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/CubicEval.py",
    "Astrea": f"{os.getenv('HOME')}/raynet/simlibs/Astrea/src/AstreaEval.py",
    "CleanSlate": f"{os.getenv('HOME')}/raynet/simlibs/CleanSlate/src/CleanSlateEval.py",
}

python = f"{os.getenv('HOME')}/raynet/.venv/bin/python" # RayNet's .venv (you can use your own if you have Ray/RLlib and other critical RL libraries)
runner = runner_paths['Orca']                           # The python script that uses Ray/RLlib to facilitate training/inference of the given protocol
ini_path = f"{os.getenv('HOME')}/omnetpp/samples/orbtcpExperiments/simulations/paperExperiments/experiment1/experiment1_orca.ini"
section = "Orca_Run1" # optional argument that specifies which config section to use. Defaults to the name of the protocol.
# Run the experiment using the protocol's associated runner script (and a valid python virtual environment)
os.system(f"{python} {runner} {ini_path} {section}")