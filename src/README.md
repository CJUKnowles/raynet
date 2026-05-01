# RayNet/src
This directory contains several critical classes that can be collectively referred to as RayNet's simulation wrapper. This is where most of the simulation control logic lives - The main event loops, observation collection, and simulation termination (through SIGINT or time limits) are all performed here.

- *GymApiBind.cc* Is the binding module that allows users to communicate with OMNeT++ simulations via a python module. It just maps python function calls to C++ function calls.
- *GymApi.cc* translates these functions into useful simulation control code. This is technically the simulation wrapper.
- *Cmdrlenv.cc* is the simulation interface class, but it is useful to think of as the simulation itself. The binding module, simulation wrapper, and simulation class all work together to provide OMNeT++ with RL functionality.