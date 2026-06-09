"""3D foot contact: smooth normal penalty + smooth Coulomb friction cone (E3D-2).

Ports the 2D contact-gradient findings (E1 normal, F1 friction) to a 3D foot on a flat
ground. Design goals, in priority order (per the project's hard-won lessons):
  1. FRAMES ARE EXPLICIT. Everything here is WORLD frame: foot position/velocity in, contact
     force out, ground normal n = world +Z. Do NOT mix a body-frame foot velocity with a
     world-frame normal -- that silently flips the friction direction. The SRBD bridge
     (floating_base_srbd.contact_to_body_wrench) is the ONLY place world<->body happens.
  2. DIFFERENTIABLE APPROXIMATION, not hard projection. Normal force uses a softplus penalty;
     friction uses a tanh-saturated cone so ||f_t|| <= mu*f_n holds *by construction* and the
     gradient is bounded everywhere (no sign(), no hard clamp).
  3. Friction OPPOSES the tangential sliding velocity (dissipative): f_t . v_t <= 0 always.

Contact force model (flat ground at z = ground_z, normal n = +Z):
    gap   = foot_z - ground_z                      # signed; <0 means penetrating
    pen   = eps_pen * softplus(-gap/eps_pen)       # smooth penetration depth >= 0
    gate  = sigmoid(-gap/eps_pen)                  # smooth contact indicator in (0,1)
    vn    = v_foot . n                             # normal velocity (world); >0 = moving up
    srelu = lambda x: x * sigmoid(x/v_d)           # smooth relu, EXACTLY 0 at x=0 (no rest-force bias)
    f_n   = k_n * pen + k_d * gate * srelu(-vn)    # stiffness + approach-only damping; >=0, 0 at vn=0
    v_t   = v_foot - vn n                           # tangential velocity (world)
    f_t   = -mu f_n * tanh(|v_t|/v_eps) * v_t/|v_t| # smooth Coulomb; ||f_t|| <= mu f_n
    f_contact_world = f_n n + f_t

A hard-contact variant (relu penalty + sign() friction) is provided ONLY for the E1-style
smooth-vs-hard gradient contrast; it is not meant for training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContactParams:
    k_n: float = 1.0e4       # normal stiffness (N/m)
    k_d: float = 400.0       # contact-gated normal damping (N·s/m); ~O(critical) to settle bounces
    mu: float = 0.8          # friction coefficient
    eps_pen: float = 2.0e-3  # penetration smoothing length (m)
    v_eps: float = 0.05      # friction velocity scale (m/s)
    v_d: float = 0.05        # damping smooth-relu velocity scale (m/s)
    ground_z: float = 0.0


def _safe_tanh_over_norm(x: torch.Tensor, v_eps: float, dim: int = -1):
    """Return (|x|, tanh(|x|/v_eps)/|x|) with the second term safe & smooth at |x|->0 (-> 1/v_eps)."""
    n = torch.linalg.norm(x, dim=dim, keepdim=True)
    small = n < 1e-9
    ratio = torch.where(small,
                        torch.full_like(n, 1.0 / v_eps),
                        torch.tanh(n / v_eps) / n.clamp_min(1e-12))
    return n, ratio


def foot_contact_force_world(p_foot_world: torch.Tensor, v_foot_world: torch.Tensor,
                             params: ContactParams = ContactParams(),
                             normal_world: Optional[torch.Tensor] = None,
                             mode: str = "smooth") -> dict:
    """World-frame contact force for feet on a flat ground. All tensors (...,3).

    Returns dict with f_world (...,3) and diagnostics (f_n, f_t, pen, v_t, cone_ratio).
    mode='smooth' (differentiable, for training) or 'hard' (relu+sign, for gradient contrast).
    """
    p = params
    if normal_world is None:
        normal_world = p_foot_world.new_tensor([0.0, 0.0, 1.0])
    n = normal_world / torch.linalg.norm(normal_world, dim=-1, keepdim=True)

    gap = (p_foot_world * n).sum(-1, keepdim=True) - p.ground_z      # (...,1)
    vn = (v_foot_world * n).sum(-1, keepdim=True)                    # normal velocity (world)
    v_t = v_foot_world - vn * n                                      # tangential velocity (world)

    if mode == "smooth":
        pen = p.eps_pen * F.softplus(-gap / p.eps_pen)              # smooth depth >= 0
        gate = torch.sigmoid(-gap / p.eps_pen)                      # smooth contact indicator
        srelu_vn = (-vn) * torch.sigmoid(-vn / p.v_d)              # smooth relu, =0 at vn=0
        f_n = p.k_n * pen + p.k_d * gate * srelu_vn                 # >= 0; damping only on approach, 0 at rest
        vt_mag, ratio = _safe_tanh_over_norm(v_t, p.v_eps)
        f_t = -p.mu * f_n * ratio * v_t                            # ||f_t|| = mu f_n tanh(|vt|/v_eps)
        cone_mag = p.mu * f_n * torch.tanh(vt_mag / p.v_eps)
    elif mode == "hard":
        pen = torch.relu(-gap)                                      # non-smooth at gap=0
        f_n = p.k_n * pen + p.k_d * pen * torch.relu(-vn)
        vt_mag = torch.linalg.norm(v_t, dim=-1, keepdim=True)
        dir_t = v_t / vt_mag.clamp_min(1e-12)
        f_t = -p.mu * f_n * dir_t                                   # hard Coulomb (sign-like)
        cone_mag = torch.linalg.norm(f_t, dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    f_world = f_n * n + f_t
    return dict(f_world=f_world, f_n=f_n, f_t=f_t, pen=pen, gap=gap, v_t=v_t,
                vt_mag=vt_mag if mode == "hard" else torch.linalg.norm(v_t, -1, keepdim=True),
                cone_mag=cone_mag, mu_fn=p.mu * f_n)


if __name__ == "__main__":
    for dev in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
        params = ContactParams()
        # foot penetrating 3mm, sliding in +x at 0.2 m/s, descending at 0.1 m/s
        pf = torch.tensor([[0.1, 0.0, -0.003]], device=dev, requires_grad=True)
        vf = torch.tensor([[0.2, 0.0, -0.1]], device=dev, requires_grad=True)
        out = foot_contact_force_world(pf, vf, params)
        fw = out["f_world"]
        fn = out["f_n"].item(); ft = out["f_t"][0]; mufn = out["mu_fn"].item()
        cone_ok = (torch.linalg.norm(ft).item() <= mufn + 1e-6)
        dissip = (out["f_t"][0] * out["v_t"][0]).sum().item()       # should be <= 0
        fw.sum().backward()
        print(f"[{dev}] f_n={fn:.2f}N  |f_t|={torch.linalg.norm(ft).item():.2f}N  "
              f"mu*f_n={mufn:.2f}N  cone_ok={cone_ok}  f_t·v_t={dissip:+.3e}(<=0)  "
              f"grad_finite={torch.isfinite(pf.grad).all().item() and torch.isfinite(vf.grad).all().item()}")
    print("contact_3d smoke test OK: friction within cone, dissipative, differentiable, GPU-clean.")
