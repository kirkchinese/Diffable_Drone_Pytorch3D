"""3D SRBD standing environment (E3D-3): floating base + 4 legs + smooth contact.

Action space (deliberately minimal & differentiable, see report E3D-3):
  a = leg_ext in R^4 = per-leg VERTICAL extension [FL,FR,RL,RR]. The foot's body-frame
  height is (nominal - leg_ext); extending a leg pushes its foot down -> more penetration
  -> more normal force. **Forces are produced by the contact model, never commanded
  directly**, so the contact-gradient question stays central and the policy's contribution
  is measurable. Horizontal foot positions are fixed at nominal (standing needs no stepping).

Frames: identical to floating_base_srbd. p/v = COM (world); q body->world (w,x,y,z); w body.
Foot contact velocity uses ONLY the rigid-body part v_com + omega_world x r -- NOT a
position-difference (that 1/dt term blew up friction gradients in 2D-F4). The commanded
leg extension is treated as quasi-static geometry (sets penetration, not foot velocity).

Attitude in the LOSS is the singularity-free "up-vector" error 1 - R[2,2] (body z vs world
z); Euler angles are used ONLY for human-readable diagnostics, never integrated/differentiated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from pytorch3d.transforms import quaternion_to_matrix

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "models"))
from floating_base_srbd import FloatingBaseState, srbd_step, contact_to_body_wrench  # noqa: E402
from contact_3d import ContactParams, foot_contact_force_world  # noqa: E402
from go2_kinematics import Go2Kinematics  # noqa: E402
from go2_inertia import composite_inertia  # noqa: E402

NOMINAL_Q = [0.0, 0.9, -1.8] * 4


@dataclass
class StandingConfig:
    mass: torch.Tensor          # scalar
    I_body: torch.Tensor        # (3,3)
    I_body_inv: torch.Tensor    # (3,3)
    c_body: torch.Tensor        # (3,) composite COM offset in base frame
    foot_rel_com: torch.Tensor  # (4,3) nominal foot positions relative to COM (body frame)
    contact: ContactParams
    dt: float
    device: torch.device
    dtype: torch.dtype

    @property
    def rest_height(self) -> float:
        """Analytic rest COM height from geometry: -foot_z_rel_com - equilibrium penetration."""
        pen_eq = (self.mass.item() * 9.81) / (4.0 * self.contact.k_n)
        return float(-self.foot_rel_com[:, 2].mean().item() - pen_eq)


def build_standing_config(device="cuda:0", dtype=torch.float32,
                          nominal_q=None, contact: Optional[ContactParams] = None,
                          dt: float = 1e-3) -> StandingConfig:
    kin = Go2Kinematics.from_urdf(device=device, dtype=dtype)
    q = torch.tensor(NOMINAL_Q if nominal_q is None else nominal_q, dtype=dtype, device=device)
    ci = composite_inertia(kin, q)
    feet = kin.foot_positions(q)[0]                  # (4,3) base frame
    foot_rel_com = (feet - ci.com).detach()
    I = ci.inertia.detach()
    return StandingConfig(
        mass=ci.mass.detach(), I_body=I, I_body_inv=torch.linalg.inv(I),
        c_body=ci.com.detach(), foot_rel_com=foot_rel_com,
        contact=contact or ContactParams(), dt=dt,
        device=torch.device(device), dtype=dtype)


# --------------------------------------------------------------------------- #
def level_error(q: torch.Tensor) -> torch.Tensor:
    """Singularity-free tilt error: 1 - (body z . world z) = 1 - R[2,2]. (B,). 0 = level."""
    R = quaternion_to_matrix(q)
    return 1.0 - R[:, 2, 2]


def roll_pitch_diag(q: torch.Tensor):
    """Roll/pitch for HUMAN diagnostics only (atan2 from R). Never used in the loss/integration."""
    R = quaternion_to_matrix(q)
    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 2, 1] ** 2 + R[:, 2, 2] ** 2))
    return roll, pitch


def foot_world(state: FloatingBaseState, leg_ext: torch.Tensor, cfg: StandingConfig):
    """Foot world positions (B,4,3) and rigid-body foot velocities (B,4,3)."""
    R = quaternion_to_matrix(state.q)                              # (B,3,3)
    fr = cfg.foot_rel_com.unsqueeze(0).expand(state.p.shape[0], 4, 3).clone()  # (B,4,3)
    fr[:, :, 2] = fr[:, :, 2] - leg_ext                            # extend leg -> foot lower
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, fr)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = state.v[:, None, :] + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
    return foot_w, foot_v


def standing_step(state: FloatingBaseState, leg_ext: torch.Tensor, cfg: StandingConfig,
                  mode: str = "smooth", grad_decay: float = 1.0):
    """One env step. Returns (next_state, info). The 4-foot wrench is computed vectorized --
    math-identical to summing contact_to_body_wrench over feet (equivalence unit-tested):
        f_world  = sum_k f_k_world
        tau_body = R^T sum_k (r_k_world x f_k_world)."""
    foot_w, foot_v = foot_world(state, leg_ext, cfg)                       # (B,4,3)
    out = foot_contact_force_world(foot_w, foot_v, cfg.contact, mode=mode)  # batched over feet
    f_each = out["f_world"]                                                # (B,4,3) world
    R = quaternion_to_matrix(state.q)                                      # (B,3,3) body->world
    r_world = foot_w - state.p[:, None, :]                                 # (B,4,3) lever from COM
    tau_world = torch.cross(r_world, f_each, dim=-1).sum(1)                # (B,3)
    f_tot = f_each.sum(1)                                                  # (B,3)
    tau_body = torch.einsum("bji,bj->bi", R, tau_world)                    # R^T tau_world
    nxt = srbd_step(state, cfg.mass, cfg.I_body, cfg.I_body_inv, cfg.dt,
                    f_world=f_tot, tau_body=tau_body, grad_decay=grad_decay)
    cone = torch.linalg.norm(out["f_t"], dim=-1) / out["mu_fn"].squeeze(-1).clamp_min(1e-9)  # (B,4)
    info = dict(net_force_world=f_tot, net_tau_body=tau_body,
                f_n=out["f_n"].squeeze(-1), cone=cone, foot_world=foot_w)
    return nxt, info


def initial_state(cfg: StandingConfig, B=1, height=None, roll=0.0, pitch=0.0,
                  device=None, dtype=None) -> FloatingBaseState:
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
    dev = device or cfg.device
    dt = dtype or cfg.dtype
    if height is None:
        height = cfg.rest_height
    p = torch.zeros(B, 3, device=dev, dtype=dt); p[:, 2] = height
    ang = torch.zeros(B, 3, device=dev, dtype=dt)
    ang[:, 0] = roll if torch.is_tensor(roll) else roll
    ang[:, 1] = pitch if torch.is_tensor(pitch) else pitch
    Rm = euler_angles_to_matrix(ang, "XYZ")
    q = matrix_to_quaternion(Rm)
    return FloatingBaseState(p, q, torch.zeros(B, 3, device=dev, dtype=dt),
                             torch.zeros(B, 3, device=dev, dtype=dt))
