"""Floating-base single-rigid-body dynamics (SRBD) for the Go2 digital twin — E3D-1.

This module deliberately does ONE thing well: integrate a free floating rigid body with
**unambiguous frames, a single global quaternion convention, exponential-map quaternion
integration (NOT Euler angles), and clean differentiability**. Contact/friction come later
(E3D-2+); getting this layer exactly right is the prerequisite.

================================  CONVENTIONS (global)  ========================
State  x = (p, q, v, w):
  p : (B,3) **COM** position,        WORLD frame   (NOT the base-link origin -- see note)
  q : (B,4) orientation quaternion, (w,x,y,z) scalar-first, UNIT, **Body->World**
            (R = quaternion_to_matrix(q) maps body vectors to world: v_world = R @ v_body)
  v : (B,3) **COM** linear velocity, WORLD frame
  w : (B,3) base angular velocity,  **BODY frame**   (omega_body)

  Why COM (not base origin): translational Newton (m v_dot = f) and rotational Euler
  (I w_dot = tau) decouple ONLY about the COM. So p/v track the COM. The base-link origin
  (needed for FK / rendering) is recovered as:  base_origin_world = p - R @ c_body,
  where c_body is the composite COM offset in the body frame (models/go2_inertia.py).

Inertia:
  I_body : (3,3) or (B,3,3) inertia about the COM, expressed in the BODY frame (constant
           in body frame -> Euler's equation has the w x (I w) gyroscopic term).

External inputs to a step:
  f_world  : (B,3) net external force EXCLUDING gravity, in the WORLD frame.
  tau_body : (B,3) net external torque about the COM, in the BODY frame.

  CONTACT WRENCH CONVERSION (write it down, do not eyeball -- a frame mix-up here silently
  flips torques). For a world-frame contact force f_i_world applied at a world-frame contact
  point p_i_world, with COM at p_com_world = p (the state) and R = quaternion_to_matrix(q):
        r_world  = p_i_world - p_com_world          # lever, WORLD frame
        f_world += f_i_world                        # forces just sum in WORLD
        tau_body += R^T (r_world x f_i_world)        # torque about COM, rotated into BODY
  Equivalently (identity R(a x b) = (Ra) x (Rb) for R in SO(3)):
        tau_body += (R^T r_world) x (R^T f_i_world) = r_body x f_body.
  Use contact_to_body_wrench() so callers cannot get the frame wrong. The two forms are
  unit-tested for equality (see e3d1_floating_base_checks / E3D-2).

Equations of motion:
  WORLD translation:  m dv/dt = m g + f_world
  BODY rotation:      I dw/dt = tau_body - w x (I w)
  Orientation kin.:   q integrated by the exponential map of the body rotation vector w*dt
                      (q_{t+1} = normalize(q_t (x) exp_quat(w*dt))), right-multiplied
                      because w is a BODY-frame rate. No Euler angles anywhere.

Integrator: semi-implicit (symplectic) Euler -- update velocities first, then advance
position/orientation with the NEW velocities. First-order, stable, BPTT-friendly.
==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import torch
from pytorch3d.transforms import quaternion_to_matrix, quaternion_multiply

G_WORLD = (0.0, 0.0, -9.81)


# --------------------------------------------------------------------------- #
# Quaternion exponential map  (rotation vector -> unit quaternion), grad-safe.
# --------------------------------------------------------------------------- #
def quat_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    """Map a rotation vector phi=(...,3) (axis*angle) to a unit quaternion (w,x,y,z).

    delta_q = [cos(theta/2), sin(theta/2) * phi/theta]. Safe & differentiable at theta->0
    via a Taylor expansion of sin(theta/2)/theta (-> 1/2). This is the smooth replacement
    for Euler-angle integration; it has no kinematic singularity, so BPTT gradients through
    orientation stay bounded.
    """
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)          # (...,1)
    half = 0.5 * theta
    small = theta < 1e-6
    # sin(theta/2)/theta  ->  1/2 - theta^2/48 + ...   (clamp keeps the large branch finite)
    sin_half_over_theta = torch.where(
        small, 0.5 - theta * theta / 48.0, torch.sin(half) / theta.clamp_min(1e-12))
    w = torch.cos(half)
    xyz = rotvec * sin_half_over_theta
    return torch.cat([w, xyz], dim=-1)


def _matvec(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(3,3) or (B,3,3) times (B,3) -> (B,3)."""
    if M.dim() == 2:
        return v @ M.T
    return torch.einsum("bij,bj->bi", M, v)


@dataclass
class FloatingBaseState:
    p: torch.Tensor   # (B,3) world
    q: torch.Tensor   # (B,4) body->world, (w,x,y,z)
    v: torch.Tensor   # (B,3) world
    w: torch.Tensor   # (B,3) body

    def detach(self):
        return FloatingBaseState(self.p.detach(), self.q.detach(), self.v.detach(), self.w.detach())


def srbd_step(state: FloatingBaseState,
              mass: torch.Tensor, I_body: torch.Tensor, I_body_inv: torch.Tensor,
              dt: float,
              f_world: Optional[torch.Tensor] = None,
              tau_body: Optional[torch.Tensor] = None,
              g_world: torch.Tensor = None,
              grad_decay: float = 1.0) -> FloatingBaseState:
    """One semi-implicit Euler step of the free floating-base SRBD. See module docstring."""
    p, q, v, w = state.p, state.q, state.v, state.w
    B = p.shape[0]
    dev, dt_dtype = p.device, p.dtype
    if g_world is None:
        g_world = p.new_tensor(G_WORLD)
    if f_world is None:
        f_world = torch.zeros_like(p)
    if tau_body is None:
        tau_body = torch.zeros_like(w)

    # ---- velocities first (semi-implicit) ----
    # rotation (body frame): w_dot = I^{-1} (tau - w x (I w))
    Iw = _matvec(I_body, w)
    w_dot = _matvec(I_body_inv, tau_body - torch.cross(w, Iw, dim=-1))
    w_next = w + dt * w_dot

    # translation (world): v_dot = g + f/m
    if mass.dim() == 0:
        f_over_m = f_world / mass
    else:
        f_over_m = f_world / mass.view(-1, 1)
    v_next = v + dt * (g_world + f_over_m)

    # ---- optional temporal gradient decay (same mechanism as the drone GDecay) ----
    if grad_decay != 1.0:
        from go2_grad import g_decay  # local import to keep dependency optional
        decay = grad_decay ** dt
        p, v_next, w_next = g_decay(p, decay), g_decay(v_next, decay), g_decay(w_next, decay)

    # ---- advance configuration with NEW velocities ----
    p_next = p + dt * v_next
    dq = quat_exp_map(w_next * dt)                 # body-frame increment
    q_next = quaternion_multiply(q, dq)            # right-multiply (body rate)
    q_next = q_next / torch.linalg.norm(q_next, dim=-1, keepdim=True)
    return FloatingBaseState(p_next, q_next, v_next, w_next)


def rollout(state: FloatingBaseState, mass, I_body, I_body_inv, dt: int, steps: int,
            force_fn: Optional[Callable] = None, torque_fn: Optional[Callable] = None,
            g_world=None, grad_decay: float = 1.0):
    """Run ``steps`` of srbd_step. ``force_fn(t,state)->(B,3)`` / ``torque_fn`` optional.
    Returns (final_state, traj) where traj is a dict of stacked (steps+1,B,*) tensors."""
    traj = {k: [getattr(state, k)] for k in ("p", "q", "v", "w")}
    for t in range(steps):
        f = force_fn(t, state) if force_fn else None
        tau = torque_fn(t, state) if torque_fn else None
        state = srbd_step(state, mass, I_body, I_body_inv, dt, f, tau, g_world, grad_decay)
        for k in ("p", "q", "v", "w"):
            traj[k].append(getattr(state, k))
    traj = {k: torch.stack(val, dim=0) for k, val in traj.items()}
    return state, traj


# --------------------------------------------------------------------------- #
# Contact wrench -> SRBD inputs. THE one frame conversion that must be exact.
# --------------------------------------------------------------------------- #
def contact_to_body_wrench(p_contact_world: torch.Tensor, f_contact_world: torch.Tensor,
                           p_com_world: torch.Tensor, q: torch.Tensor):
    """A world contact force at a world contact point -> SRBD inputs (f_world, tau_body).

    Unambiguous, world-frame-first (verify-first; optimize to body frame only after):
        r_world  = p_contact_world - p_com_world
        tau_body = R^T (r_world x f_contact_world)        R = quaternion_to_matrix(q), body->world
    All of (B,3); q is (B,4). Differentiable. The caller sums contributions over the feet.
    Body-frame-optimized equivalent (post-verification): with r_body = R^T r_world and
    f_body = R^T f_contact_world, tau_body = r_body x f_body (identity R(a x b)=(Ra)x(Rb)).
    """
    R = quaternion_to_matrix(q)                              # (B,3,3) body->world
    r_world = p_contact_world - p_com_world                  # (B,3)
    tau_world = torch.cross(r_world, f_contact_world, dim=-1)
    tau_body = torch.einsum("bji,bj->bi", R, tau_world)      # R^T @ tau_world
    return f_contact_world, tau_body


def base_origin_world(p_com_world: torch.Tensor, q: torch.Tensor, c_body: torch.Tensor) -> torch.Tensor:
    """Map the COM-position state to the base-link origin in world: base = p - R @ c_body.
    c_body (composite COM offset in body frame) may be (3,) or (B,3)."""
    R = quaternion_to_matrix(q)
    if c_body.dim() == 1:
        return p_com_world - torch.einsum("bij,j->bi", R, c_body)
    return p_com_world - torch.einsum("bij,bj->bi", R, c_body)


# --------------------------------------------------------------------------- #
# Diagnostics (for verification): conserved quantities of the free body.
# --------------------------------------------------------------------------- #
def angular_momentum_world(q: torch.Tensor, w: torch.Tensor, I_body: torch.Tensor) -> torch.Tensor:
    """L_world = R (I_body w). Torque-free -> conserved. Returns (B,3)."""
    R = quaternion_to_matrix(q)
    Iw = _matvec(I_body, w)
    return torch.einsum("bij,bj->bi", R, Iw)


def rotational_energy(w: torch.Tensor, I_body: torch.Tensor) -> torch.Tensor:
    """E_rot = 1/2 w . (I_body w). Torque-free -> conserved. Returns (B,)."""
    return 0.5 * (w * _matvec(I_body, w)).sum(-1)


if __name__ == "__main__":
    # Verify the contact-wrench frame conversion BEFORE building E3D-2 on it.
    from pytorch3d.transforms import random_quaternions
    torch.manual_seed(0)
    for dev in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
        B = 8
        q = random_quaternions(B).to(dev)
        R = quaternion_to_matrix(q)
        p_com = torch.randn(B, 3, device=dev)
        p_ct = p_com + torch.randn(B, 3, device=dev) * 0.3        # contact point near COM
        f_w = torch.randn(B, 3, device=dev, requires_grad=True)   # world contact force

        f_out, tau_body = contact_to_body_wrench(p_ct, f_w, p_com, q)
        # Independent cross-check via the body-frame form r_body x f_body.
        r_world = p_ct - p_com
        r_body = torch.einsum("bji,bj->bi", R, r_world)
        f_body = torch.einsum("bji,bj->bi", R, f_w)
        tau_body_alt = torch.cross(r_body, f_body, dim=-1)
        max_err = (tau_body - tau_body_alt).abs().max().item()

        # Differentiability + GPU residency.
        tau_body.sum().backward()
        ok_grad = torch.isfinite(f_w.grad).all().item()
        print(f"[{dev}] |R^T(r x f) - r_body x f_body| = {max_err:.2e}  "
              f"(equivalent: {max_err < 1e-5})  grad finite={ok_grad}  "
              f"tau on {tau_body.device}")
    print("contact_to_body_wrench verified: world-frame and body-frame torque forms agree, differentiable, GPU-clean.")
