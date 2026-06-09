"""Composite single-rigid-body (SRBD) inertia of the Go2, from the URDF.

Aggregates all link inertials into one rigid body: total mass, composite COM, and the
composite inertia tensor about that COM -- all expressed in the **base/body frame** and
evaluated at a given joint configuration ``q`` via differentiable FK + the parallel-axis
theorem. This is the inertia the floating-base SRBD dynamics (E3D-1+) integrates.

Frames (consistent with go2_kinematics / the URDF): base/body frame, +X fwd, +Y left,
+Z up. Each URDF link inertia is given about that link's COM in the link frame; we rotate
it into the base frame and shift to the composite COM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_kinematics import Go2Kinematics  # noqa: E402


@dataclass
class CompositeInertia:
    mass: torch.Tensor      # scalar
    com: torch.Tensor       # (3,) in base frame
    inertia: torch.Tensor   # (3,3) about composite COM, base frame


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)


def composite_inertia(kin: Go2Kinematics, q: torch.Tensor) -> CompositeInertia:
    """Composite mass / COM / inertia (about COM, base frame) at joint config ``q`` (12,)."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    dev, dt = kin.device, kin.dtype
    T = kin.forward(q.to(dev, dt))   # link -> (1,4,4), base at origin/identity

    M = torch.zeros((), device=dev, dtype=dt)
    m_c = torch.zeros(3, device=dev, dtype=dt)     # Sum m_i c_i
    # First pass: mass + first moment -> COM
    contribs = []   # (m_i, c_i base, I_i base-about-own-COM)
    for name, link in kin.model.links.items():
        inr = link.inertial
        if inr is None or inr.mass <= 0:
            continue
        m_i = torch.tensor(inr.mass, device=dev, dtype=dt)
        R_l = T[name][0, :3, :3]
        t_l = T[name][0, :3, 3]
        com_local = torch.tensor(inr.com, device=dev, dtype=dt)
        c_i = R_l @ com_local + t_l                       # link COM in base frame
        I_local = torch.tensor(inr.inertia, device=dev, dtype=dt)
        I_i = R_l @ I_local @ R_l.T                        # rotate inertia into base frame
        contribs.append((m_i, c_i, I_i))
        M = M + m_i
        m_c = m_c + m_i * c_i
    com = m_c / M

    # Second pass: parallel-axis shift each link inertia to composite COM, sum.
    I = torch.zeros(3, 3, device=dev, dtype=dt)
    eye = torch.eye(3, device=dev, dtype=dt)
    for m_i, c_i, I_i in contribs:
        d = c_i - com
        I = I + I_i + m_i * (d.dot(d) * eye - torch.outer(d, d))
    # symmetrize against round-off
    I = 0.5 * (I + I.T)
    return CompositeInertia(mass=M, com=com, inertia=I)


if __name__ == "__main__":
    kin = Go2Kinematics.from_urdf(device="cpu", dtype=torch.float64)
    for label, q in [("q=0 (straight legs)", torch.zeros(12, dtype=torch.float64)),
                     ("nominal standing", torch.tensor([0.0, 0.9, -1.8] * 4, dtype=torch.float64))]:
        ci = composite_inertia(kin, q)
        evals = torch.linalg.eigvalsh(ci.inertia)
        print(f"[{label}]")
        print(f"  mass = {ci.mass.item():.4f} kg")
        print(f"  COM (base) = {ci.com.numpy().round(5).tolist()}")
        print(f"  I diag = {torch.diag(ci.inertia).numpy().round(5).tolist()}")
        print(f"  I eigvals = {evals.numpy().round(5).tolist()}  (all > 0 -> SPD: {(evals > 0).all().item()})")
