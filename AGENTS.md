# Codex project context

Before substantial work, read `codex_memory/README.md`, then open only the topic
memory relevant to the task. For live campaign or remote-compute work, always
read `codex_memory/current-state.md` first.

## Working rules

- Treat repository code, current result files, process state, and Git history as
  stronger evidence than imported conversational memory.
- Imported remote status is a dated snapshot. Verify it before reporting or
  mutating remote jobs.
- Preserve the user's dirty worktree. Do not delete results, checkpoints,
  papers, untracked research assets, or unrelated edits.
- Keep changes surgical and verify them proportionally. For long experiments,
  run a smoke test first and make outputs resumable and auditable.
- Never write passwords, tokens, or private keys into this repository or memory.
- Do not use `--reset_lr` for ordinary checkpoint continuation; in this project
  it means fine-tuning with a reset optimizer/schedule.

## Evidence and research conventions

- The central scientific standard is gradient fidelity, not forward fit alone.
  Prefer analytic-gradient versus finite-difference checks where feasible.
- Test a proposed capability on a task where the relevant axis is binding;
  otherwise a negative or neutral result may be uninformative.
- For sim-to-sim quadruped claims, use MuJoCo as an external gate, not as proof
  of real-hardware transfer.
- Keep training seeds, evaluation seeds, checkpoint-selection rules, and output
  directories explicit. Never overwrite the frozen paper truth data.

## Environment notes

- Local historical environment: conda env `pytorch`, Python 3.9, PyTorch
  2.4.1, PyTorch3D 0.7.8. Confirm the actual machine before using GPU indices.
- DGX Spark environment and the last known campaign state are recorded in
  `codex_memory/current-state.md`.
- Canonical quadruped joint order is `FL, FR, RL, RR`; Unitree low-level SDK
  order is `FR, FL, RR, RL`, so hardware bridges must remap explicitly.
