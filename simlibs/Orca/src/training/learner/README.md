# Original Orca TD3 Learner

This directory vendors the two Python files used from the original Orca
`rl-module`:

- `agent.py`: TensorFlow TD3 actor, critics, and training operations.
- `utils.py`: replay buffer and exploration-noise implementations.

The files retain their original MIT license notices. `agent.py` has one
packaging-only change: it imports `utils.py` relatively so the learner can be
used as a self-contained package within RayNet.

The original repository's `d5.py`, `envwrapper.py`, Mahimahi executables, and
parameter files are not used by `OrcaTraining_TD3.py` and are intentionally not
included.

`OrcaTraining_TD3.py` reproduces the relevant orchestration behavior from
`d5.py`, including zero-padded recurrent history, invalid-observation handling,
batched replay ingestion, replay persistence, deterministic evaluation, and the
original learner warmup/update-delay defaults.
