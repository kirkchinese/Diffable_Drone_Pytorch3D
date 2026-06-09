"""E3D-0 proof-of-concept: render the differentiable Go2 digital twin.

Produces a multi-view montage of two poses (q=0 straight legs, and a nominal standing
crouch), reports quantitative foot heights, and verifies the whole pipeline is
differentiable end-to-end (rendered pixels -> gradient w.r.t. joint angles q).

Run:  python render_standing_pose.py --device cuda:0
Out:  ../figures/go2_digital_twin_poses.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from go2_render import Go2Twin  # noqa: E402
from go2_urdf import LEGS  # noqa: E402

FIG_DIR = HERE.parent / "figures"
NOMINAL_STAND = [0.0, 0.9, -1.8]   # hip, thigh, calf (typical Go2 standing crouch)


def q_from_per_leg(per_leg) -> torch.Tensor:
    return torch.tensor(per_leg * 4, dtype=torch.float32).unsqueeze(0)


def foot_height_report(twin, q, label):
    feet = twin.kin.foot_positions(q.to(twin.device, twin.dtype))[0]   # (4,3)
    base_z = 0.0
    fz = feet[:, 2]
    print(f"[{label}] foot z (base frame): "
          + ", ".join(f"{leg}={z:+.3f}" for leg, z in zip(LEGS, fz.tolist()))
          + f"  | base height above feet ~ {base_z - fz.mean().item():.3f} m")
    return fz.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()

    twin = Go2Twin(device=args.device, dtype=torch.float32)
    print(f"Go2Twin on {args.device}: {len(twin.visual_links)} visual links, "
          f"{len(twin._mesh_cache)} unique meshes")

    poses = {"q = 0 (straight legs)": q_from_per_leg([0.0, 0.0, 0.0]),
             "nominal standing": q_from_per_leg(NOMINAL_STAND)}
    azims = [135.0, 90.0, 200.0]

    fig, axes = plt.subplots(len(poses), len(azims), figsize=(4 * len(azims), 4 * len(poses)))
    for r, (label, q) in enumerate(poses.items()):
        foot_height_report(twin, q, label)
        for c, az in enumerate(azims):
            rgb, _ = twin.render(q, image_size=args.image_size, azim=az)
            ax = axes[r, c]
            ax.imshow(np.clip(rgb[0], 0, 1))
            ax.set_title(f"{label}\nazim={az:.0f}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Go2 differentiable digital twin (official URDF + PyTorch3D rigid skinning)",
                 fontsize=14)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "go2_digital_twin_poses.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")

    # ---- end-to-end differentiability: pixels -> q gradient ----
    q = q_from_per_leg(NOMINAL_STAND).to(twin.device, twin.dtype).requires_grad_(True)
    rgb, _ = twin.render(q, image_size=128, return_tensor=True)
    loss = rgb.mean()
    loss.backward()
    gnorm = q.grad.norm().item()
    print(f"[diff-check] d(mean pixel)/dq norm = {gnorm:.3e}  "
          f"finite={torch.isfinite(q.grad).all().item()}  "
          f"nonzero_dims={(q.grad.abs() > 0).sum().item()}/12")


if __name__ == "__main__":
    main()
