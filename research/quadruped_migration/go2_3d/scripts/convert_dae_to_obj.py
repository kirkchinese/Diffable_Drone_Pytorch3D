"""Convert the official Go2 COLLADA (.dae) visual meshes to .obj for PyTorch3D.

The official Unitree URDF (go2_description, humble @ 8bd6717) ships visual meshes
as COLLADA. PyTorch3D's io only loads .obj/.ply, so we bake each DAE (scene-graph
transforms applied, concatenated to a single mesh) into a flat .obj.

Physics parameters (mass, inertia, joint limits) come from the URDF, NOT from these
meshes -- the meshes are used only for differentiable depth/RGB rendering.

Run:
    python convert_dae_to_obj.py
Outputs go2_3d/assets/obj/<name>.obj and prints per-mesh bounds for a units check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
DAE_DIR = ASSETS / "go2_description" / "dae"
OBJ_DIR = ASSETS / "obj"

MESHES = ["base", "hip", "thigh", "thigh_mirror", "calf", "calf_mirror", "foot"]


def convert_one(name: str) -> dict:
    src = DAE_DIR / f"{name}.dae"
    # force='mesh' concatenates all geometries and bakes scene-graph node transforms.
    mesh = trimesh.load(src, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{name}: expected Trimesh, got {type(mesh)}")

    dst = OBJ_DIR / f"{name}.obj"
    # include_texture=False keeps it geometry-only (renderer adds a default white texture).
    mesh.export(dst, include_texture=False)

    lo, hi = mesh.bounds
    return {
        "name": name,
        "verts": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "extent_m": np.round(hi - lo, 4).tolist(),
        "min_m": np.round(lo, 4).tolist(),
        "max_m": np.round(hi, 4).tolist(),
        "watertight": bool(mesh.is_watertight),
    }


def main() -> None:
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DAE source : {DAE_DIR}")
    print(f"OBJ output : {OBJ_DIR}\n")
    print(f"{'mesh':14s} {'verts':>8s} {'faces':>8s}  {'extent (x,y,z) m':<26s} watertight")
    for name in MESHES:
        info = convert_one(name)
        ext = ",".join(f"{v:.3f}" for v in info["extent_m"])
        print(f"{info['name']:14s} {info['verts']:8d} {info['faces']:8d}  ({ext:<24s}) {info['watertight']}")


if __name__ == "__main__":
    main()
