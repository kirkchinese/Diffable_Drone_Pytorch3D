# Drone-to-quadruped research

Primary output: `research/quadruped_migration/`. Read the current repository
notes for detail:

- `research/quadruped_migration/research_note.md`
- `research/quadruped_migration/go2_3d/README.md`
- `research/quadruped_migration/go2_3d/reports/go2_3d_research_note.md`

The long-term goal is a credible differentiable-physics path from the drone
work to Unitree Go2 locomotion, suitable as research and portfolio work. Claims
must stay honest: zero sim-to-real gap is not attainable; MuJoCo is a necessary
external proxy gate, not sufficient proof of hardware transfer.

## Scientific through-line

The drone succeeds because its useful dynamics are smooth and its attitude is
handled with an algebraic construction. Quadruped contact removes both
advantages. Early experiments measured roughly 4000x BPTT gradient growth and
about 52% gradient-direction errors. Positive scalar controls such as GDecay or
gradient clipping can reduce magnitude but cannot repair sign errors.

Across the entire study, three principles survived repeated attempts to break
them:

1. Gradient fidelity matters more than one-step forward fit.
2. A residual/intervention must be placed where the real error occurs.
3. Structure is load-bearing: domain randomization or a larger MLP cannot repair
   an axis outside the simulator's representational class.

## Completed 2D line

The planar SLIP/SRBM work covers contact pathology, smooth versus hard contact,
locomotion, and neural residuals R1-R11. Important results:

- smooth contact made standing and locomotion trainable; hard contact produced
  huge or misleading gradients;
- analytic foot velocity fixed a false `1/dt` friction-gradient explosion;
- contact-force residuals outperformed free acceleration residuals in both
  forward accuracy and gradient fidelity when force law was the mismatch;
- physical gating, nonnegative normal force, friction-cone constraints, and
  bounded corrections prevented phantom/off-ground forces and takeover;
- when geometry caused a control-sign flip, a force residual became structurally
  blind; a kinematic/foot-placement residual restored the correct sign and
  oracle-level closed-loop behavior;
- combined kinematic and force heads automatically routed pure force, pure
  geometry, and mixed mismatches to the right channel; light per-head
  regularization sharpened routing.

Thus the full statement is not simply “more physical residuals are better.” The
residual's physical structure must match the mismatch type and location.

## Completed 3D Go2 line

- E3D-0..4 built and checked a URDF-derived Go2 visual/inertial model, floating
  base SRBD, smooth 3D contact, standing, and trot policy training.
- E3D-4b/c showed structured residuals and hardened multi-seed evaluation were
  more credible than free MLP correction.
- E3D-5/5c used MuJoCo as an external gate. Better one-step residual fit could
  worsen policy transfer; even a more appropriate generalized-force channel did
  not beat the nominal robust policy under out-of-class mismatch. Residual
  correction is therefore best treated as diagnosis/sysID unless its closed
  loop is separately proven.
- E3D-8 made gait phase continuous and differentiable. Finite-difference versus
  BPTT agreement stayed good for smooth phase control and failed for hard
  contact. Coupled CPGs could lock, but free coupling found a degenerate 3+1
  gait; trot structure and moderate Tegotae feedback helped. Gradient fidelity
  is the foundation, while inductive structure determines whether the learned
  gait is useful.
- E3D-9 showed `robustness > correction`: dynamics domain randomization improved
  or stabilized MuJoCo transfer on represented axes (especially kp=300/400),
  but not on the unrepresented soft-actuator kp=200 axis.
- E3D-9b/c tried crude lag and a more faithful second-order compliant foot-space
  layer. Neither closed kp=200, and both traded away performance elsewhere.
  Joint-space actuator behavior is outside the current SRBD twin; covering it
  requires joint dynamics/ABA or a different simulator structure.
- E3D-10 connected differentiable heightmap perception to control, but a
  swing-height-only action could not solve support-foot tipping on dense terrain.
  Easy terrain did not require perception; hard terrain failed for both blind
  and perceptive policies. The intervention did not match the failure mode.
- E3D-11 found that on flat constant-speed terrain, low friction was not binding
  and could even improve tracking by softening landing errors.
- E3D-11b made friction binding using slopes. At 16 degrees, mu=0.3 slid
  backward while mu>=0.6 climbed, matching the `mu < tan(theta)` threshold.
  This is the clean counterexample showing that an axis is only testable when
  the task actually stresses it.

## High-value next experiments for DGX

These were not completed in the imported sessions. Choose one only after
checking the live paper campaign and run a small gate before a large sweep.

1. Slope-conditioned low-friction DR: train with a friction range that includes
   the slip regime and with slope/high-traction demands, then compare nominal
   and DR policies across `(slope, mu)` in MuJoCo and multiple seeds. This is the
   most direct continuation of E3D-11b.
2. Matched perceptive terrain task: first create a sparse-obstacle regime where
   swing-foot collision is the dominant failure and height control is genuinely
   causal; then expand actions to foot placement, body attitude, and speed for
   support-instability terrain. Use blind/perceptive and smooth/MuJoCo gates.
3. Symmetry-constrained CPG on variable speed/terrain: parameterize coupling by
   leg symmetry to remove the 3+1 attractor, and require gait adaptation so the
   extra degrees of freedom are binding.
4. Stair/hard-contact external gate: train only on a smooth surrogate, validate
   on MuJoCo stairs, and treat failure as an explicit differentiable-model
   boundary. Do not claim stairs are faithfully differentiable.
5. Joint-space structure upgrade: add articulated dynamics/ABA plus motor PD and
   actuator randomization if the goal is to close the kp=200 blind spot. This is
   the largest and riskiest option, not a quick DGX sweep.

Recommended ordering by scientific value per engineering cost: (1), then (2)
or (3), then (4); reserve (5) for a deliberate new phase.

## Operational cautions

- Full-resolution Go2 visuals are roughly 400k faces and should be decimated for
  batch rendering.
- PyTorch3D does not directly read official COLLADA assets; conversion needs
  `pycollada` and `trimesh`.
- Historical local GPU numbering was confusing and machine-dependent. Always
  inspect the actual device rather than relying on old `cuda:0/cuda:1` notes.
- Evaluate relative CPG phase per sample before circular averaging. Averaging
  absolute phase across a batch can falsely report loss of synchronization.
- Small Go2 workloads can be kernel-launch bound; more GPU memory alone does not
  guarantee speedup. Profile before redesigning around DGX throughput.
