# Incremental differentiable-simulation core

`diffsim` is the compatibility boundary between training code and robot-specific
simulation. It is intentionally introduced beside the existing drone modules:

- legacy drone numerical code remains unchanged and is constructed by an
  adapter;
- new robots implement the tensor-first contracts in `api.py`;
- environment and loss selection happens through explicit registries;
- hot-loop dynamics should be functional, batched over environments, GPU
  resident, and free of `.item()`/`.cpu()` synchronization;
- asset parsing, mesh conversion, configuration, and logging may remain on CPU.

The first migration stage does not claim that the legacy `DroneSimulator` is
already a clean `VectorEnv`; it preserves behavior while providing the seam used
to replace robot-specific assumptions one at a time.
