"""E3D-1 verification: pin down floating-base SRBD frames / quaternion / integration / grad.

Five checks (print metrics + a 4-panel figure). Everything in float64 for clean numerics.

  C1 Free-fall vs analytic        -> translation + gravity sign correct; first-order convergence.
  C2 Quaternion unit-norm         -> orientation stays a valid unit quaternion over long rollout.
  C3 Torque-free conservation     -> WORLD angular momentum & rotational energy bounded-drift
                                     (sharp test: a frame/sign bug here blows the drift up).
  C4 Differentiability vs horizon -> BPTT grads finite & bounded through the smooth dynamics.
  C5 exp-map vs Euler-angle       -> Euler-angle orientation integration explodes near the
                                     pitch=+-90 singularity; exp-map (ours) stays bounded.
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
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE.parent / "models"))
from floating_base_srbd import (  # noqa: E402
    FloatingBaseState, srbd_step, rollout, quat_exp_map,
    angular_momentum_world, rotational_energy, G_WORLD)
from go2_kinematics import Go2Kinematics  # noqa: E402
from go2_inertia import composite_inertia  # noqa: E402
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
DT64 = torch.float64


def go2_inertia(device="cpu"):
    kin = Go2Kinematics.from_urdf(device=device, dtype=DT64)
    ci = composite_inertia(kin, torch.tensor([0.0, 0.9, -1.8] * 4, dtype=DT64))
    return ci.mass.detach(), ci.inertia.detach(), torch.linalg.inv(ci.inertia).detach()


def identity_state(B=1):
    q = torch.zeros(B, 4, dtype=DT64); q[:, 0] = 1.0
    return FloatingBaseState(torch.zeros(B, 3, dtype=DT64), q,
                             torch.zeros(B, 3, dtype=DT64), torch.zeros(B, 3, dtype=DT64))


# --------------------------------------------------------------------------- #
def c1_freefall(mass, I, Iinv):
    g = G_WORLD[2]
    results = {}
    for dt in (2e-3, 1e-3):
        steps = int(1.0 / dt)
        s = identity_state(1)
        s.v[:] = torch.tensor([0.3, -0.2, 1.0], dtype=DT64)  # nonzero initial velocity
        _, traj = rollout(s, mass, I, Iinv, dt, steps)
        t = torch.arange(steps + 1, dtype=DT64) * dt
        pz_num = traj["p"][:, 0, 2]
        pz_ana = 0.0 + 1.0 * t + 0.5 * g * t ** 2
        vz_num = traj["v"][:, 0, 2]
        vz_ana = 1.0 + g * t
        results[dt] = dict(t=t.numpy(), pz_num=pz_num.numpy(), pz_ana=pz_ana.numpy(),
                           pz_err=(pz_num - pz_ana).abs().max().item(),
                           vz_err=(vz_num - vz_ana).abs().max().item())
    order = results[2e-3]["pz_err"] / results[1e-3]["pz_err"]
    print(f"[C1 free-fall] pz max|err| dt=2e-3:{results[2e-3]['pz_err']:.3e}  "
          f"dt=1e-3:{results[1e-3]['pz_err']:.3e}  ratio={order:.2f} (≈2 => 1st-order) | "
          f"vz max|err|={results[1e-3]['vz_err']:.2e} (≈0: velocity exact)")
    return results


def c2_quat_norm(mass, I, Iinv):
    dt, steps = 1e-3, 8000
    s = identity_state(1)
    s.w[:] = torch.tensor([0.9, -1.3, 0.7], dtype=DT64)  # tumbling
    _, traj = rollout(s, mass, I, Iinv, dt, steps)
    norms = torch.linalg.norm(traj["q"][:, 0, :], dim=-1)
    dev = (norms - 1.0).abs().max().item()
    print(f"[C2 quat-norm] max|‖q‖-1| over {steps} steps = {dev:.3e} (≈machine eps)")
    return traj["q"][:, 0, :].numpy(), norms.numpy(), dt


def c3_conservation(mass, I, Iinv):
    dt, steps = 1e-3, 6000
    s = identity_state(1)
    s.w[:] = torch.tensor([1.5, 0.3, 2.4], dtype=DT64)  # generic spin, asymmetric I
    _, traj = rollout(s, mass, I, Iinv, dt, steps)
    L = angular_momentum_world(traj["q"].reshape(-1, 4), traj["w"].reshape(-1, 3), I).reshape(steps + 1, 3)
    E = rotational_energy(traj["w"].reshape(-1, 3), I).reshape(steps + 1)
    L0 = torch.linalg.norm(L[0])
    L_drift = (torch.linalg.norm(L - L[0:1], dim=-1) / L0)
    E_drift = ((E - E[0]).abs() / E[0])
    print(f"[C3 torque-free] world |L| rel-drift max={L_drift.max():.3e}  "
          f"E_rot rel-drift max={E_drift.max():.3e}  (bounded & small => frames/signs OK)")
    t = np.arange(steps + 1) * dt
    return t, L_drift.numpy(), E_drift.numpy()


def c4_diff_vs_horizon(mass, I, Iinv):
    dt = 1e-3
    horizons = [50, 100, 200, 400, 800, 1600]
    gnorms = []
    for H in horizons:
        s = identity_state(1)
        w0 = torch.tensor([[0.8, 1.1, -0.5]], dtype=DT64, requires_grad=True)
        v0 = torch.tensor([[0.2, 0.0, 1.5]], dtype=DT64, requires_grad=True)
        s = FloatingBaseState(s.p, s.q, v0, w0)
        st, _ = rollout(s, mass, I, Iinv, dt, H)
        # loss: final position-z (transl.) + final body-x mapped to world (orient.)
        R = quaternion_to_matrix(st.q)
        loss = st.p[0, 2] + R[0, :, 0].sum()
        loss.backward()
        g = torch.cat([w0.grad.flatten(), v0.grad.flatten()])
        gnorms.append(g.norm().item())
    print(f"[C4 diff] grad-norm vs horizon {horizons} = "
          + ", ".join(f"{g:.2f}" for g in gnorms) + "  (bounded, no explosion)")
    return horizons, gnorms


# ---- Euler-angle orientation integrator (the pitfall) for contrast ----------
def _euler_rate_zyx(euler, w):
    """body omega -> ZYX euler rates; the inverse kinematic matrix has 1/cos(pitch)."""
    roll, pitch = euler[..., 0], euler[..., 1]
    sr, cr = torch.sin(roll), torch.cos(roll)
    cp, tp = torch.cos(pitch), torch.tan(pitch)
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    roll_dot = wx + sr * tp * wy + cr * tp * wz
    pitch_dot = cr * wy - sr * wz
    yaw_dot = (sr / cp) * wy + (cr / cp) * wz          # <-- singular at pitch=±90°
    return torch.stack([roll_dot, pitch_dot, yaw_dot], -1)


def c5_expmap_vs_euler():
    """A constant body-y driver (w_y=1) sweeps pitch toward 90°; we differentiate the final
    orientation w.r.t. the body-**z** rate w0. Under exp-map this sensitivity is bounded; the
    ZYX Euler-angle integrator represents w0 via yaw_dot = w0/cos(pitch), which blows up at the
    pitch=90° gimbal singularity -> the exact gradient-explosion pitfall to avoid."""
    dt = 1e-3
    # pitch ≈ w_y*t; 90°≈1.5708s. March right up to the gimbal singularity.
    horizons = [200, 600, 1000, 1300, 1500, 1560, 1568, 1570]

    def omega(w0):  # body rate: y-driver = 1.0, z-rate = w0 (the differentiated input)
        return torch.stack([torch.zeros_like(w0), torch.ones_like(w0), w0], -1)

    W0 = 0.05  # small z-rate: keeps the Euler forward pass clean so the gradient growth is monotonic
    g_exp, g_eul = [], []
    for H in horizons:
        # exp-map (ours)
        w0 = torch.tensor([W0], dtype=DT64, requires_grad=True)
        q = torch.tensor([[1.0, 0, 0, 0]], dtype=DT64)
        for _ in range(H):
            q = quaternion_multiply(q, quat_exp_map(omega(w0) * dt))
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        quaternion_to_matrix(q)[0, 0, 2].backward()
        g_exp.append(w0.grad.abs().item())
        # euler-angle (pitfall)
        w0 = torch.tensor([W0], dtype=DT64, requires_grad=True)
        eul = torch.zeros(1, 3, dtype=DT64)
        for _ in range(H):
            eul = eul + _euler_rate_zyx(eul, omega(w0)) * dt
        eul[0, 2].backward()   # final yaw: accumulates (cos roll / cos pitch)*w0 -> 1/cos(pitch)
        g_eul.append(w0.grad.abs().item())
    pitch_deg = [h * dt * 180 / np.pi for h in horizons]
    print("[C5 exp vs euler] pitch(deg): " + ", ".join(f"{p:.0f}" for p in pitch_deg))
    print("   exp-map |grad|: " + ", ".join(f"{g:.2e}" for g in g_exp))
    print("   euler   |grad|: " + ", ".join(f"{g:.2e}" for g in g_eul) + "  (explodes near 90°)")
    return pitch_deg, g_exp, g_eul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")  # float64 verification -> CPU is fine & exact
    args = ap.parse_args()
    mass, I, Iinv = go2_inertia(args.device)
    print(f"SRBD inertia (nominal stand): mass={mass:.3f} kg, I diag={torch.diag(I).numpy().round(4).tolist()}\n")

    c1 = c1_freefall(mass, I, Iinv)
    qtraj, qn, dt2 = c2_quat_norm(mass, I, Iinv)
    t3, Ld, Ed = c3_conservation(mass, I, Iinv)
    hz, gn = c4_diff_vs_horizon(mass, I, Iinv)
    pd, ge, gu = c5_expmap_vs_euler()

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    r = c1[1e-3]
    a.plot(r["t"], r["pz_num"], label="numeric pz"); a.plot(r["t"], r["pz_ana"], "--", label="analytic")
    a.set_title(f"C1 free-fall (dt=1e-3, max|err|={r['pz_err']:.1e} m)"); a.set_xlabel("t (s)"); a.set_ylabel("z (m)"); a.legend()

    a = ax[0, 1]
    a.plot(np.arange(len(qn)) * dt2, qn - 1.0)
    a.set_title(f"C2 quaternion norm dev (max={np.abs(qn-1).max():.1e})"); a.set_xlabel("t (s)"); a.set_ylabel("‖q‖ - 1")

    a = ax[1, 0]
    a.plot(t3, Ld, label="world |L| rel-drift"); a.plot(t3, Ed, label="E_rot rel-drift")
    a.set_title("C3 torque-free conservation"); a.set_xlabel("t (s)"); a.set_yscale("log"); a.legend()

    a = ax[1, 1]
    a.semilogy(pd, gu, "o-", label="Euler-angle (pitfall)"); a.semilogy(pd, ge, "s-", label="exp-map (ours)")
    a.axvline(90, color="k", ls=":", lw=1, label="pitch=90° (gimbal)")
    a.set_title("C5 orientation grad: exp-map vs Euler-angle"); a.set_xlabel("pitch reached (deg)")
    a.set_ylabel("|d(orient)/d(w0)|"); a.legend()
    fig.suptitle("E3D-1 floating-base SRBD verification (Go2 composite inertia)", fontsize=14)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "e3d1_floating_base_checks.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
