"""Differentiable forward kinematics for the Go2 digital twin (PyTorch).

Implements rigid forward kinematics over the URDF kinematic tree. Everything is a
batched 4x4 homogeneous transform built from differentiable torch ops, so gradients
flow from foot positions / link poses back to joint angles ``q`` and the floating-base
pose -- exactly what a differentiable-physics quadruped twin needs.

"Skeletal binding" for a *rigid* robot = one link is one rigid bone; we transform each
link's whole mesh by its world transform (no vertex blend weights). See go2_render.py
for the mesh-skinning side built on top of these transforms.

Conventions (match the URDF / ROS body frame): +X forward, +Y left, +Z up, gravity -Z.
RPY is fixed-axis: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Rotation matrices are Body->World
with columns [X, Y, Z], identical to the drone project's convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_urdf import Go2Model, load_go2, LEGS, JOINTS_PER_LEG  # noqa: E402

ACTUATED_ORDER = [f"{leg}_{j}_joint" for leg in LEGS for j in JOINTS_PER_LEG]


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Fixed-axis roll-pitch-yaw -> rotation matrix. rpy: (..., 3) -> (..., 3, 3)."""
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    z = torch.zeros_like(r)
    o = torch.ones_like(r)
    Rx = torch.stack([o, z, z, z, cr, -sr, z, sr, cr], -1).reshape(*r.shape, 3, 3)
    Ry = torch.stack([cp, z, sp, z, o, z, -sp, z, cp], -1).reshape(*r.shape, 3, 3)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], -1).reshape(*r.shape, 3, 3)
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation about a fixed unit ``axis`` (3,) by batched ``angle`` (...,)."""
    axis = axis / axis.norm().clamp_min(1e-12)
    ax, ay, az = axis.unbind(-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c
    r00 = c + ax * ax * C
    r01 = ax * ay * C - az * s
    r02 = ax * az * C + ay * s
    r10 = ay * ax * C + az * s
    r11 = c + ay * ay * C
    r12 = ay * az * C - ax * s
    r20 = az * ax * C - ay * s
    r21 = az * ay * C + ax * s
    r22 = c + az * az * C
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], -1).reshape(*angle.shape, 3, 3)


def _homog(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(...,3,3), (...,3) -> (...,4,4) homogeneous transform."""
    shape = R.shape[:-2]
    T = torch.zeros(*shape, 4, 4, device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


class Go2Kinematics:
    """Batched differentiable FK over the Go2 tree.

    Usage:
        kin = Go2Kinematics.from_urdf(device="cuda:0")
        T = kin.forward(q)                       # q: (B,12) -> dict link -> (B,4,4)
        feet = kin.foot_positions(q)             # (B,4,3) in base frame (base at origin)
    """

    def __init__(self, model: Go2Model, device="cpu", dtype=torch.float32):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.base_link = model.base_link

        # BFS topological order of joints from the base.
        self._ordered_joints = self._topo_sort(model)
        self.q_index = {name: i for i, name in enumerate(ACTUATED_ORDER)}

        # Pre-bake each joint's constant origin transform + axis.
        self._origin = {}   # joint name -> (4,4)
        self._axis = {}     # joint name -> (3,) or None (fixed)
        for j in model.joints.values():
            R = rpy_to_matrix(torch.tensor(j.origin_rpy, device=self.device, dtype=dtype))
            t = torch.tensor(j.origin_xyz, device=self.device, dtype=dtype)
            self._origin[j.name] = _homog(R, t)
            if j.jtype == "revolute":
                self._axis[j.name] = torch.tensor(j.axis, device=self.device, dtype=dtype)
            else:
                self._axis[j.name] = None

    @classmethod
    def from_urdf(cls, urdf_path: Optional[Path] = None, device="cpu", dtype=torch.float32):
        model = load_go2(urdf_path) if urdf_path else load_go2()
        return cls(model, device=device, dtype=dtype)

    @staticmethod
    def _topo_sort(model: Go2Model) -> list:
        children = {}
        for j in model.joints.values():
            children.setdefault(j.parent, []).append(j)
        order, stack = [], [model.base_link]
        while stack:
            link = stack.pop(0)
            for j in children.get(link, []):
                order.append(j)
                stack.append(j.child)
        return order

    def forward(self, q: torch.Tensor,
                base_pos: Optional[torch.Tensor] = None,
                base_R: Optional[torch.Tensor] = None) -> dict:
        """q: (B,12) actuated angles in ACTUATED_ORDER. Returns {link_name: (B,4,4)}."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        B = q.shape[0]
        dev, dt = self.device, self.dtype
        q = q.to(dev, dt)

        if base_R is None:
            base_R = torch.eye(3, device=dev, dtype=dt).expand(B, 3, 3)
        if base_pos is None:
            base_pos = torch.zeros(B, 3, device=dev, dtype=dt)
        T_world = {self.base_link: _homog(base_R, base_pos)}

        for j in self._ordered_joints:
            T_origin = self._origin[j.name].expand(B, 4, 4)
            axis = self._axis[j.name]
            if axis is not None:
                angle = q[:, self.q_index[j.name]]
                T_motion = _homog(axis_angle_to_matrix(axis, angle),
                                  torch.zeros(B, 3, device=dev, dtype=dt))
                T_joint = T_origin @ T_motion
            else:
                T_joint = T_origin
            T_world[j.child] = T_world[j.parent] @ T_joint
        return T_world

    def foot_positions(self, q, base_pos=None, base_R=None) -> torch.Tensor:
        """(B,4,3) world positions of [FL,FR,RL,RR]_foot."""
        T = self.forward(q, base_pos, base_R)
        return torch.stack([T[f"{leg}_foot"][:, :3, 3] for leg in LEGS], dim=1)


if __name__ == "__main__":
    kin = Go2Kinematics.from_urdf(device="cpu", dtype=torch.float64)
    B = 1
    # q=0: legs straight down -> feet should match the summary's zero-config offsets.
    q0 = torch.zeros(B, 12, dtype=torch.float64)
    feet0 = kin.foot_positions(q0)[0]
    print("feet @ q=0 (base frame):")
    for leg, p in zip(LEGS, feet0):
        print(f"  {leg}: {p.tolist()}")

    # Gradient sanity: d(foot_z)/dq should be non-trivial and finite.
    q = torch.zeros(B, 12, dtype=torch.float64, requires_grad=True)
    feet = kin.foot_positions(q)
    fl_foot_z = feet[0, 0, 2]
    fl_foot_z.backward()
    g = q.grad[0].reshape(4, 3)[0]  # FL leg [hip, thigh, calf] grads
    print(f"\nd(FL_foot_z)/d[FL_hip,thigh,calf] = {g.tolist()}")
    print("FK is differentiable:", q.grad is not None and torch.isfinite(q.grad).all().item())
