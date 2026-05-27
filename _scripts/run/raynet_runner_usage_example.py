import os
from pathlib import Path


raynet_runner = f"{Path(__file__).resolve().parents[0]}/raynet_runner.py"

# Define your protocol, .ini path, and section name
protocol = "orca"
ini_path = f"home/omnetpp/samples/orbtcpExperiments/simulations/paperExperiments/experiment1/experiment1_orca.ini"
section = "Orca_Run1"

# The final command! Uses the provided raynet_runner.py to run your desired simulation.
os.system(f"python3 {raynet_runner} {protocol} {ini_path} {section}")

"""
...and you're done!
To automate your experiments externally, just perform a similar call in a loop pointing to whatever protocols, 
configs, and sections you want to run. As long as the target simulation contains a Broker and the proper NED sources,
the runner will take care of everything for you! Just treat it like an RL alternative to opp_run. 

Some tips:
- Vector output to a folder named after the config sections. 
    If you're doing multiple runs, give them different section names 
    to prevent results from overwriting each other.
- Most protocol runners use a single-threaded version of Ray, but they are
    configured to not conflict with one another. So feel free to perform
    whatever multithreading you desire within your automation script, RayNet
    should be able to handle it.
- NED sources MUST be included at the beginning of your config.ini to
    function. This is because RayNet does not make use of opp_run, which many users 
    use to inject NED sources. If you copy the sources you usually pass to that command 
    into an ned-path line, it should work identically.
- You don't need to source any particular .venv before calling the runner script.
    It automatically locates RayNet's preferred .venv and uses it to start the simulation.
"""