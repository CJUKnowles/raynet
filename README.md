# RayNet - RL Training Platform for Network Protocols
RayNet (Raynet was originally created by Luca Giacomoni in 2023) is a platform that enables simulation-based development of RL-driven congestion control protocols through the OMNeT++ discrete event simulator.  This repo is an extension of RayNet designed to improve the user experience and expand RayNet's capabilities as part of a final year project at the University of Sussex.

## System software components
This repository contains RayNet's source code and some scripts to train and evauate RL models. 

The system integrates the core Omnet++ discrete event simulator with its linked simulation libraries and Ray/RLlib through pybindings11. The figure below depicts the different packages/libraries in RayNet:

<img src="/docs/images/libraries.png" width="600">

Raynet requires (at least) the following third party (open-source) software:
- **Omnet++**: Provides the underlying simulation framework. Support for version 6.3 was added as part of this fork but still need work. In particular, this version causes crashes on environment reset that are usually caught and handled by Ray/RLlib without issue. If this is a problem, version 6.0 support is more stable but may not work with newer versions of external dependencies like INET.
- **INET**: Provides useful simulation components relating to computer networks and congestion control. A custom version of INET4.5 was used for this project (https://github.com/Avian688/inet4.5) and is required for the Orca, Astrea, CleanSlate, TcpPaced, and Cubic simlibs. Aiden Valentine is the author of this custom INET version as well as the TcpPaced and Cubic simlibs.
- **Ray/RLlib**: RayNet supports all traditional RL workflows through `OmnetBindApi`, but Ray/RLlib is the most trivially supported and well-tested option for RayNet. Plus, the repo contains many Ray/RLlib examples to work off of.
- **Python Modules:** Critical python modules like TensorFlow and PyTorch support training and evaluation scripts. A `requirements.txt` is provided that lists the essential modules, and `requirementsExtras.txt` provides **all** modules used in production of the final year project.

OMNeT++ and INET are assumed to be installed the HOME directory. If this is an issue, feel free to alter `build.sh` and `cmakelists.txt` to support your needs or create symbolic links to your existing OMNeT++/INET directories.

**RLComponents** is part of our distribution and include ad-hoc components (_Stepper_, _Broker_, _RLInterface_) that allow simulations to run agents that make decisions in a time discrete fashion. 

_Custom Components_ refers to any simulation component that the user may need when modelling simulations. For our RL-driven congestion control protocol, the following libraries are required (_ecmp_, _rdp_), both included as submodules in this repository. 

## Dependencies

The project has been tested on Ubuntu 20.04, with Omnet++ v5.6.2, pybind11 v2.7.1. Required dependencies are:
- Omnet++ 6.3.0
- Ray 1.13.0

To be able to reproduce congestion control results, you will also need to install:
- INET 4.2.5

## Building instrutions

Building RayNet consists in several compilation and linkink steps:

- Clone this repo and its submodules
- Compile Omnet++ libraries
- Compile the required simulation libraries (e.g. INET) and **RLComponents**; 
- Compile the python binding module distributed with this repository and link with libraries above
- Set up a python interpreter with required packages (Ray/RLlib, tensorflow, etc)

The new python module can now be imported and used to implement environments following the OpenAI Gym API.


### Step 1 - Clone this repo
Clone this repository and its submodules

```
git clone --recurse-submodules -j8 
```


### Step 2 - Build Omnet++ libraries 

Download [Omnet++](https://omnetpp.org/download/) (version 5.6.2) and install in HOME directory, following [these](https://doc.omnetpp.org/omnetpp/InstallGuide.pdf) instructions. 

### Step 3 - Build simulation libraries.

Libraries should be compiled using the **opp_makemake** utility provided by Omnet++. This utility generates the Makefile to compile library components to be used with Omnet++.

```
cd <custom_model>
opp_makemake -o <custom_library_name> --make-so -M release 
make
```

The script _build.sh_ contains instruction to build the simulation components.

P.S. The command line instructions to build INET are not included in the _build.sh_ script. Follow the library's instruction and build the library in your HOME directory.

### Step 4 - Compile binding module

The binding module is built using **cmake**:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=[Release | Debug] ../
make
```

The output is a python module that can be imported with the standard _import_ clause in a python script. The name of the module is defined in the _CMakeLists.txt_.

### Step 5

RayNet was developped on Python 3.10. 

The last step consists in setting up the python environment, with required packages. 

### Set up training environment

Check the _Manual.pdf_ in the documentation for details on how to set up the training for your learning agents. 



[![DOI](https://zenodo.org/badge/561974777.svg)](https://zenodo.org/badge/latestdoi/561974777)

