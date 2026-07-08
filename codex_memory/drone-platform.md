# Main drone platform

The current engineering direction is to move the root DiffDrone/PyTorch3D
project toward a commercially credible differentiable drone simulation and
training platform. The user's preferred working relationship is technical lead
plus implementation collaborator: audit first, then make evidence-backed
changes. README claims are not acceptance evidence.

Primary audit documents:

- `research/platform_audit/capability_audit.md`
- `research/platform_audit/progress_log.md`

Last imported status:

- P0 gradient coverage was added in `testscript/test_gradients.py`,
  `test_render_gradients.py`, and `test_dynamics_axes.py`.
- Reproducibility/evaluation work added `train.py --seed`, a private generator
  for scene generation, structured metrics JSON, and canonical aggregation.
- A 19-experiment ablation table exists; `exp07_cmaes_lossnet` collapsing to
  about 1% success was still an investigation item.
- Multi-drone tests were repaired, but production wiring in `train.py` still
  required a design decision.
- External baselines, SAC dependency closure, full pytest/CI, and a true
  DiffPhysDrone parity reference remained incomplete.
- Historical suite result was 17 PASS / 2 SKIP / 0 FAIL on GPU, with
  `python testscript/run_all.py --cpu-only` available for CPU-only checks.

Important caution: older audit aggregation overwrote
`viz_results/thesis_eval/summary_aggregated.csv` with a 3-seed result. Paper
truth was subsequently reconstructed and frozen under the conference-paper
directory. Never use the mutable audit CSV as the paper's 5-seed source.
