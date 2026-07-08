# Thesis and conference paper

## Chinese thesis

Title: `基于可微物理的无人机自主避障轨迹规划方法`. The core method is
GCGL: global gradient-norm clipping plus a goal-reaching auxiliary loss.

Authoritative LaTeX source:
`docs/论文相关/5.29需要修订论文/thesis/thesis-latex/`.
References are a manually ordered `thebibliography`, not BibTeX. The final
delivery still needs migration to the university Word template. Local preview
uses `testscript/build_preview.sh` with Noto CJK substitutions because the
authoritative source uses Windows Chinese fonts.

Trusted experiment mapping:

- exp01 = Baseline-MSE
- exp17 = GoalLoss-Only
- exp21 = GCGL
- exp22 = GradClip-Only
- exp_clip_0p5/2p0/5p0 = sensitivity runs

Previously verified headline values included stable AR 0.78 -> 0.91 and offline
success 74.4% -> 83.1%. Later raw-data recomputation refined Welch statistics to
`t=2.95`, `d=1.87`; older Chinese text may still contain rounded `2.97/1.88` in
places and should be checked against the current source rather than memory.

Review work already added a qualitative trajectory figure, clarified the
pre-clipping gradient norm, corrected grouped-bar wording, strengthened the
significance statement, and narrowed contribution claims. Remaining historical
items included shortening the abstract, clarifying CNN-only parameter counts,
normalizing method order in figures, and acknowledging absent external
PPO/EGO/real-hardware baselines.

## English conference paper

Location: `docs/论文相关/会议论文/`. It is an IEEEtran two-column English paper,
historically seven pages, prepared for review with 周维钧老师. The scientific
positioning chosen by the user is:

- headline finding: asymmetric interaction between goal loss and gradient
  clipping;
- middle-ground narrative: discuss instability without overclaiming a dramatic
  post-peak collapse;
- primary comparison relies on offline success and stable AR; retention is a
  secondary, raw-single-batch-peak-sensitive metric;
- the 15-configuration table is not a strict factorial design; the 2x2 study is
  the controlled interaction evidence.

All paper numbers were recomputed from original five-seed logs. Frozen copies
and recomputation scripts live under the paper's `data/` directory. The
framework figure was converted to native TikZ; the AR-curve legend was moved so
it no longer covered its annotation. The dynamics command equation was
corrected to match `navigation_utils.py`, and depth is described as PyTorch3D
sensing under `no_grad`, not end-to-end differentiable rendering.

The 2x2x3 training campaign exists to replace the single-run-per-cell caveat.
After it completes, evaluate the preregistered `best_ar.pth` checkpoints and
report uncertainty across training seeds. Do not rewrite the paper's conclusion
before those results are actually verified.

Publication choices still requiring the user/teacher include author order and
corresponding author, verified English affiliation, target conference, page
limit, and acknowledgements.
