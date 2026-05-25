Located at: https://github.com/CJUKnowles/raynet

# RayNet - RL Training Platform for Network Protocols
RayNet (Raynet was originally created by Luca Giacomoni in 2023) is a platform that enables the development of RL-driven congestion control protocols via the OMNeT++ discrete event simulator. This repo is an extension of RayNet designed to improve the user experience and expand RayNet's capabilities as part of a final year project at the University of Sussex.

## Requirements
This repository contains RayNet's source code and some scripts to train and evauate RL models. 

The system integrates the core Omnet++ discrete event simulator with its linked simulation libraries and Ray/RLlib through pybindings11. The figure below depicts the different packages/libraries in RayNet:

<img src="/docs/images/libraries.png" width="600">

Raynet requires (at least) the following third party (open-source) software:
- **Omnet++**: Provides the underlying simulation framework. This fork adds partial support for version 6.3 that is unfinished. In particular, this version causes crashes on environment reset that are usually caught and handled by Ray/RLlib without issue. Alternatively, version 6.0 is more stable but may not work with newer versions of external dependencies like INET4.5.
- **INET**: Provides useful simulation components relating to computer networks and congestion control. A custom version of INET4.5 was used for this project (https://github.com/Avian688/inet4.5) and is required for Orca, Astrea, CleanSlate, TcpPaced, and Cubic. Aiden Valentine is the author of this custom INET version as well as the TcpPaced and Cubic implementations in this project.
- **Ray/RLlib**: RayNet supports all traditional RL workflows through `OmnetBindApi`, but Ray/RLlib is the most trivially supported and well-tested option for RayNet. Plus, the repo contains many Ray/RLlib examples to work off of.
- **Python Modules:** Critical python modules like TensorFlow and PyTorch support training and evaluation scripts. A `requirements.txt` is provided that lists the essential modules, and `requirementsExtras.txt` provides **all** modules used in production of the final year project.

OMNeT++ and INET are assumed to be installed the HOME directory. If this is an issue, feel free to alter `build.sh` and `cmakelists.txt` to support your needs or create symbolic links to your existing OMNeT++/INET directories.

## RayNet Components
- **src**: contains the binding API and a environment interface inspired by OMNeT++'s `cmdenv`. The contents of this directory collectively make up the simulation wrapper and will be compiled in to the `build` directory.

- **RLComponents**: contains critical simulation components like the `Broker`. In addition to some helper classes, it also crucially contains `typedefs.h`, which is where observation types are defined on a per-protocol basis. If you wish to make your own protocol, you'll need to add a few lines to this header file.

- **simlibs**: contains various user-provided simulation libraries. This includes RL-driven CC schemes written for RayNet like `Orca` and `Astraea` as well as generally useful OMNeT++ components like `cubic` and `TcpPaced`. Users wishing to add functionality to their simulations are encouraged to put any relevant code here.

## Evaluation Directories
A collection of directories intended for evaluation scripts, topologies, results, plots, etc. are provided for your convenience. These directories include:
- **_experiments**: Contains configuartion and scenario files to support experimentation.
- **_plots**: Evaluation scripts will automatically output aggregate plots here.
- **_results**: The experiment runner will parse simulation vector outputs, compile them into `.csv` files, and save them to this directory.
- **_scripts**: Used for various python scripts. By default, this contains an experiment runner and plotting script.
- **_topologies**: Intended to contain generally useful topologies to be shared among many experiments and training environments. Currently only contains a dumbbell topology.

## Building instrutions

Once the required exteernal dependencies listed above are installed, RayNet is ready to be installed.

### Step 1 - Clone this repo
Clone this repository and its submodules
```
git clone --recurse-submodules -j8 
```

### Step 2 - Run the build script

Navigate to the RayNet directory and run the build script. 
```
cd ~/raynet
./build.sh
```
This process should be repeated any time you make changes to C++ code within the project or its simulation libraries. A few optional flags were provided for convenience, users are encouraged to explore `./build.sh`.

This build script (and project as a whole) assumes you have OMNeT++ installed at `~/omnetpp`. If yours is installed elsewhere, you can either move it, create a symblic link at the correct location, or modify RayNet yourself.

### Step 3 - Python environmnet

A python environment with the modules specified in `requirements.txt` must be created prior to running RayNet. This may be the same environment used by your OMNeT++ installation if you wish, but a dedicated environment is recommended to avoid version conflicts:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
`requirements-extra.txt` contains extra optional modules and specific versions if needed.

## Usage
The easiest way to use RayNet is with the provided runner script `raynet/_scripts/run/raynet_runner.py`, which can use a trained model to perform inference on any RayNet-ready `.ini`.

Runner Usage:
```
python raynet_runner.py <protocol> <ini_path> <section>
```

Any simulation you wish to run must contain a Broker and a list of NED sources. More details provided within the runner script.

## Creating your own protocols
If you wish to implement and train your own schemes, it is recommended place them in `raynet/simlibs/` to be automatically compiled into the project via `./build.sh`. Just make sure it contains a valid `Makefile` similar to the provided simlib examples.

Orca and CleanSlate 

Refer to the Orca simlib for general usage examples. This contains examples of almost everything you will need, including a RayNet agent `Orca.cc`, a training script `OrcaTraining.py`, and an evaluation script `OrcaEval.py`.


[![DOI](https://zenodo.org/badge/561974777.svg)](https://zenodo.org/badge/latestdoi/561974777)

