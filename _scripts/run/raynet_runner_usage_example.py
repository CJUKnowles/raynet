import os
from pathlib import Path


raynet_runner = f"{Path(__file__).resolve().parents[0]}/raynet_runner.py"

# Your protocol, .ini path, and section name here:
protocol = "orca"
ini_path = f"{os.getenv('HOME')}/omnetpp/samples/orbtcpExperiments/simulations/paperExperiments/experiment1/experiment1_orca.ini"
section = "Orca_Run1"

# The final command! Uses the provided raynet_runner.py to run your desired simulation. Copy this structure to automate your experiments.
os.system(f"python3 {raynet_runner} {protocol} {ini_path} {section}")