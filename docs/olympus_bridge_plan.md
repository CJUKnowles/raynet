# RayNet-Olympus Orca Bridge V1

## Summary

V1 targets Olympus's existing Orca learner/model stack with RayNet
simulation-only episodes. RayNet owns OMNeT++, omnetbind, INI materialization,
and simulation lifecycle. Olympus owns algorithm selection, actor inference,
reward/state transforms, replay insertion, learner checkpoints, and run
orchestration.

The boundary between the projects is an IPC runner process in RayNet, not the
omnetbind Python API. External projects should not import omnetbind directly.

## Architecture

- RayNet provides `runners/olympus_runner.sh` as the external entrypoint.
- The launcher activates RayNet's virtualenv, sources OMNeT environment setup
  when available, and execs `runners/olympus_runner.py`.
- `olympus_runner.py` imports `omnetbind`, starts one OMNeT++ simulation, and
  exchanges JSON-lines messages over a private control file descriptor.
- Olympus starts this runner as a subprocess and communicates with it via a
  socketpair. OMNeT/RayNet stdout is not part of the protocol.
- Messages are agent-dictionary shaped even for single-agent Orca so that
  future multi-agent work does not require a protocol rewrite.

## IPC Protocol

Olympus sends:

```json
{"type":"start","episode":{"protocol":"orca","ini_path":"/path/OrcaTraining.ini","section":"General","duration":120,"replacements":{"!BW!":"100Mbps","!DELAY!":"10ms","!QSIZE!":"2000000b","!MAX_RL_STEPS!":"6000"}}}
```

RayNet replies:

```json
{"type":"reset","observations":{"Orca":[...]}}
```

Olympus sends one action dict per step:

```json
{"type":"step","actions":{"Orca":0.12}}
```

RayNet replies:

```json
{"type":"step","observations":{"Orca":[...]},"rewards":{"Orca":0.0},"terminateds":{"__all__":false},"info":{"simDone":false}}
```

For clean early shutdown, Olympus can send:

```json
{"type":"close"}
```

## Responsibilities

RayNet runner:

- Imports `omnetbind` from RayNet's build tree.
- Materializes INI templates by blindly applying the episode `replacements`
  dictionary.
- Applies episode `duration` as an OMNeT++ `sim-time-limit` in the generated
  INI variant.
- Supports `quiet: true` to disable verbose RayNet Orca debug prints in the
  generated INI variant.
- Calls `runner.initialise(...)`, `runner.reset()`, and `runner.step(actions)`.
- Calls `cleanup()` on normal `simDone`.
- Calls `shutdown()` plus `cleanup()` on explicit close or agent termination.

Olympus RayNet adapter:

- Spawns the RayNet runner.
- Converts RayNet Orca's 15-value observation into the raw dict expected by
  Olympus Orca state/reward code.
- Runs the Olympus Orca actor locally for inference.
- Pushes Olympus Orca `Experience` objects into the existing learner manager.
- Pulls fresh actor params from the learner on the existing cadence.
- Never imports `omnetbind`.

## V1 Scope

- Orca only.
- Simulation only.
- Explicit INI/section episodes only.
- No Mininet, iperf, listener binaries, custom kernel modules, or Olympus
  algorithm changes.
- Scenario generation and multi-agent training are out of scope, but the IPC
  shape should remain compatible with multi-agent observations/actions.

## Test Plan

- Olympus unit tests use a fake RayNet IPC client to verify dispatch, action
  dictionaries, terminal behavior, and Orca observation mapping.
- RayNet runner should be smoke-tested with a real built RayNet checkout using
  `source .venv/bin/activate` and `./build.sh`.
- A one-episode Olympus smoke config should use:

```yaml
runtime:
  algorithm: orca
  reward: sage
  state: default_orca
  action: cwnd_multiplier

environment:
  type: raynet
  environment_setup: orca_static

paths:
  py: ./venv_training/bin/python
  raynet: /home/james/raynet
  raynet_runner: /home/james/raynet/runners/olympus_runner.sh
```
