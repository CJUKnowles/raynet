# RayNet-Olympus Bridge Plan

## Summary

V1 targets Olympus's existing Orca learner/model stack with RayNet
simulation-only episodes. RayNet owns OMNeT++, omnetbind, INI materialization,
and simulation lifecycle. Olympus owns algorithm selection, actor inference,
reward/state transforms, replay insertion, learner checkpoints, and run
orchestration.

The boundary between the projects is an IPC runner process in RayNet, not the
omnetbind Python API. External projects should not import omnetbind directly.

V2 extends the same boundary to RayNet Astraea simulations and Olympus's
existing `ma_dreamer` learner. The IPC protocol stays agent-dictionary shaped:
RayNet only runs simulations and exchanges raw observations/actions, while
Olympus owns MA-Dreamer actor inference, Tempest state normalization, reward
calculation, replay insertion, and checkpoint/parameter pulls.

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
- Converts RayNet Astraea's 10-value observations into Olympus Tempest raw
  fields for `ma_dreamer`.
- Maintains per-agent Tempest/Kalman state, recurrent MA-Dreamer RSSM state,
  reward calculators, and trajectory/group metadata inside the Olympus adapter.
- Runs MA-Dreamer local world model + actor locally and sends scalar Astraea
  actions back to RayNet by agent id.
- Pushes MA-Dreamer `Experience` objects through the existing learner service.
- Never imports `omnetbind`.

## V1 Scope

- Orca only.
- Simulation only.
- Explicit INI/section episodes only.
- No Mininet, iperf, listener binaries, custom kernel modules, or Olympus
  algorithm changes.
- Scenario generation and multi-agent training are out of scope, but the IPC
  shape should remain compatible with multi-agent observations/actions.

## V2 Scope

- Astraea multi-agent simulations through the same RayNet IPC runner.
- Olympus algorithm target: `ma_dreamer`.
- No new Olympus algorithms.
- Simulation only: no Mininet, iperf, listener binaries, custom kernels, or
  emulation flow file descriptors.
- Explicit INI/section episodes remain the starting point.
- Link schedules and dynamic flow joins are not yet represented in sim-time;
  the first Astraea smoke path assumes simultaneous flows for the full episode.

## V3 Architecture Correction

The V2 Olympus-side RayNet adapter proved the simulation can train, but it is
not the correct long-term shape. It reimplements too much of Olympus's episode
and worker behavior inside `olympus/environments/raynet/runner.py`. That means
plotting, return summaries, checkpoint behavior, new worker features, and
future learner changes must be copied into the RayNet path by hand. V3 should
remove that duplication.

V3 goal: RayNet is a swappable environment backend. Olympus still owns
learners, workers, state/reward/action plugins, state logs, plots, returns, and
checkpoints.

### Desired Boundary

- RayNet owns OMNeT++, INI materialization, simulation lifecycle, and raw
  simulation stepping.
- RayNet exports observations in the same raw dictionary shape Olympus workers
  already consume from `tcp_sockopt.get_tcp_deepcc_info`.
- Olympus workers remain the place where raw metrics become state vectors,
  rewards, actions, replay entries, state logs, and learner parameter pulls.
- `run_episode` and `run_episode_marl` remain the episode owners. They may
  branch around Mininet-only operations, but they should not be replaced by a
  RayNet-specific copy.

### Proposed V3 Pieces

1. Add a generic Olympus flow backend module.

   The workers currently call:

   - `tcp_sockopt.get_tcp_deepcc_info(flow_fd)`
   - `tcp_sockopt.set_cwnd(flow_fd, new_cwnd)`

   V3 should introduce a tiny backend facade with the same operations:

   - Mininet backend delegates to `tcp_sockopt`.
   - RayNet backend communicates with a per-episode RayNet coordinator.

   Then each worker changes once from importing `tcp_sockopt` directly to
   importing the facade. The learner/state/reward/action code remains shared.

2. Make RayNet provide virtual flow endpoints.

   A RayNet coordinator process should:

   - Spawn the RayNet runner.
   - Receive grouped RayNet observations by agent id.
   - Expose one virtual endpoint per flow to Olympus workers.
   - Deliver each worker its next raw metric dict.
   - Collect each worker's requested action.
   - Step OMNeT++ with an action dictionary keyed by RayNet agent id.

   This replaces Linux file descriptors for simulation only. No listener
   binary, `oc_bridge`, iperf, Mininet namespace, or kernel CC module should be
   involved.

3. Keep `run_episode` / `run_episode_marl` as the common lifecycle.

   Backend-specific branches should be limited to places that are truly
   emulation-specific:

   - Skip `_ensure_tcp_cc` for RayNet.
   - Skip listener binary validation for RayNet.
   - Skip `oc_bridge` startup for RayNet.
   - Replace Mininet `env.run_iperf(...)` with a RayNet episode drive/blocking
     call on the RayNet environment object.
   - Keep the existing worker env construction, worker script resolution, state
     log paths, plotting, episode return parsing, checkpoint copying, result
     CSV writing, and watchdog behavior.

4. Move observation adaptation to RayNet or a protocol adapter boundary.

   Olympus should not need an algorithm-specific conversion from RayNet lists
   into raw dicts. If Orca or Astraea simulations expose lists today, V3 should
   make the RayNet runner or RayNet protocol component emit named raw fields
   matching the Olympus worker raw-info contract.

5. Make protocol support declarative.

   RayNet environment YAML should describe:

   - `protocol`
   - `ini_path`
   - `section`
   - flow count and link parameters
   - optional mapping from RayNet agent ids to Olympus flow indices

   It should not select an Olympus algorithm-specific RayNet rollout function.

### V3 Migration Steps

1. Keep V2 only as a smoke-test reference while building V3.
2. Add the flow backend facade and make Orca / MA-Dreamer workers use it.
3. Implement RayNet virtual flow endpoints and coordinator IPC.
4. Refactor `run_episode` and `run_episode_marl` to branch only around
   Mininet/listener/iperf operations.
5. Remove the duplicated Orca and MA-Dreamer rollout loops from the RayNet
   environment adapter.
6. Verify that standard episode plots and returns work without any RayNet-only
   plotting code.

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

- Astraea smoke config:

```yaml
runtime:
  algorithm: ma_dreamer
  reward: tempest_fairness_ma
  state: tempest
  action: astraea

environment:
  type: raynet
  environment_setup: astraea_smoke
```
