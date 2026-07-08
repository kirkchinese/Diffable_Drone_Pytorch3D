# Current project state

Snapshot provenance: Claude sessions through 2026-07-07, then live repository
and DGX verification on 2026-07-08 CST. Treat process state as time-sensitive.

## Local repository

- Active development branch: `refactor/diffsim-go2` at `6082af3`, pushed and
  clean after the incremental simulation-boundary refactor.
- Quadruped research branch: `chore/track-quadruped-go2-research` at `0612e3b`,
  pushed with the previously untracked Go2 sources, evidence, small checkpoints,
  and a pinned external `mujoco_menagerie` dependency.
- Platform-audit/test branch: `chore/platform-audit-and-tests` at `658e616`,
  pushed. Its CPU-oriented suite passed 14 tests with 6 expected skips and no
  failures.
- Resume/campaign baseline: `141fabc`, preserved by the annotated tag
  `dgx-campaign-20260707`; this is the code actually running on DGX.
- `DGX_MIGRATION.md` is the canonical 2x2x3 campaign guide.

Commit `141fabc` fixed ordinary continuation so that:

- `start_iter` continues from the saved iteration instead of rerunning a full
  new `num_iters` loop;
- optimizer, scheduler, Python/NumPy/Torch/CUDA RNG state, and monitor step are
  restored;
- checkpoint creation flushes `metrics.csv` first, preventing up to 24 missing
  metric rows after interruption;
- the campaign does not pass `--reset_lr` on resume;
- `multi_seed_eval.py` accepts separate checkpoint and output bases, preventing
  accidental overwrite of paper data.

The 60-step interruption test passed: learning-rate sequences were exactly
equal and metric steps 1..60 were continuous. Small loss differences matched
the CUDA nondeterminism baseline.

## Conference-paper 2x2x3 campaign

Experiment design: four cells (`baseline`, `goal`, `clip`, `gcgl`) x training
seeds 1001/1002/1003 = 12 runs, 5000 iterations each. Final selection is each
run's `best_ar.pth`; evaluation is fixed to scene seeds 0/42/123/456/789 x 32
episodes. Results must go to `viz_results/multiseed_eval`, never the frozen
paper truth directory.

DGX Spark setup was completed in the second imported session:

- ARM64/Grace-Blackwell host;
- venv `/home/xl/venvs/diffdrone-cu130`;
- Torch `2.12.1+cu130`;
- PyTorch3D `0.7.9` built successfully from upstream `main` (there is no
  upstream Git tag `v0.7.8`);
- repository 2-iteration smoke test passed at roughly 10.7 s/iteration while
  the machine was not guaranteed idle.

### MPS measurements

Approximate observed aggregate throughput:

| Mode | Throughput | Equivalent serial time |
| --- | ---: | ---: |
| Serial | 0.109 iter/s | 9.2 s/it |
| Non-MPS, 3 processes | 0.085 iter/s | worse than serial |
| MPS, 3 processes | 0.173 iter/s | 5.78 s/it |
| MPS, 4 processes | 0.188-0.213 iter/s | 4.7-5.3 s/it |
| MPS, 5 processes | 0.215-0.219 iter/s in short samples | 4.6 s/it |
| MPS, 6 processes | about 0.216-0.219 iter/s | plateau; slower runs |

MPS clearly reduced multi-process CUDA context overhead. The workload still
appeared launch/synchronization bound: many small kernels, Python/timestep
control, and PyTorch3D rasterization, with low power despite high utilization.

MPS 5/6 exploration later collided with launcher auto-refill races and a CUDA
`illegal memory access`, after which some resumes saw the device busy or
unavailable. The safe recovery was to stop only project training, restart the
MPS server, fix monitoring to derive RUN state from real processes rather than
old logs, and return to MPS=4.

Live state verified at 2026-07-08 10:02 CST:

- all four seed-1001 runs are DONE at 5000/5000;
- MPS is healthy and four seed-1002 jobs are active: baseline step 1825, goal
  650, clip 850, and gcgl 825 according to flushed metrics;
- recent aggregate throughput is about 0.1565 iter/s (effective serial 6.39
  s/it), with GPU utilization around 96%;
- approximately 35,850 training iterations remain including seed 1003, giving
  a rough 64-hour campaign ETA if current throughput persists;
- an unrelated root-owned GPU process is also present, so throughput and ETA
  may change. Do not stop or modify it.

Remote access was routed through an SSH alias named `spark-cpolar` with a
reverse proxy. The public tunnel is transient; inspect local SSH configuration
for the current endpoint. Key authentication is installed and the local user
service `spark-reverse-proxy.service` is enabled with linger. It forwards DGX
`127.0.0.1:1080` to the local HTTP/SOCKS proxy at `127.0.0.1:7897`. The proxy was
verified against GitHub with HTTP 200. Never store its password here.

## Immediate continuation checklist

1. Inspect actual PIDs, MPS server health, DONE markers, latest checkpoints,
   per-run steps, recent errors, and aggregate throughput.
2. Reconcile logs with processes; do not trust stale RUN labels.
3. If the campaign is healthy, leave it alone and report ETA. If not, resume
   only from validated checkpoints and quarantine post-checkpoint best files.
4. Choose a quadruped DGX experiment only after accounting for capacity still
   needed by the paper campaign.
