# Go2 model provenance

- Source repo: https://github.com/Unitree-Go2-Robot/go2_description
- Branch: humble
- Pinned commit: 8bd6717ff0c7b5ca388c0e10e426dd9ad873ceaf
- Retrieved: 2026-06-09T04:34:43Z
- Files vendored: urdf/go2_description.urdf, dae/{base,hip,thigh,thigh_mirror,calf,calf_mirror,foot}.dae
- Purpose: authoritative source for Go2 kinematic tree, link inertials, joint limits.
  Meshes converted DAE->OBJ (../obj/) only for PyTorch3D differentiable rendering; physics params come from the URDF.
