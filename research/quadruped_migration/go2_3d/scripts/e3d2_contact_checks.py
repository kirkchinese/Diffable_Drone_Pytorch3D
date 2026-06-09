"""E3D-2 verification: 3D foot contact (smooth normal + friction cone) + SRBD integration.

Ports the 2D E1/F1 contact-gradient analysis to 3D and adds the frame/direction checks the
project flagged as easy to get wrong. Six checks (+ a 4-panel figure):

  K1 normal force vs penetration   -> smooth bounded gradient; hard contact = step/spike (E1).
  K2 friction vs tangential speed  -> smooth Coulomb bounded grad; hard = ~inf grad at 0 (F1).
  K3 friction-cone satisfaction    -> ||f_t|| <= mu f_n by construction (sweep, 0 violations).
  K4 friction direction            -> f_t . v_t <= 0 for all directions (dissipative; frame OK).
  K5 smooth vs hard gradient       -> |d f_contact / d depth| bounded (smooth) vs spiky (hard).
  K6 drop-and-settle integration   -> Go2 SRBD + 4 feet onto ground settles; rest sum f_n ~ m g.
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
from contact_3d import ContactParams, foot_contact_force_world  # noqa: E402
from floating_base_srbd import (  # noqa: E402
    FloatingBaseState, srbd_step, contact_to_body_wrench)
from go2_kinematics import Go2Kinematics  # noqa: E402
from go2_inertia import composite_inertia  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
DT = torch.float64
PARAMS = ContactParams()


# --------------------------------------------------------------------------- #
def _grad_curve(fn, xs):
    """d fn(x)/dx at each scalar x via autograd; returns (vals, grads)."""
    vals, grads = [], []
    for x in xs:
        t = torch.tensor([x], dtype=DT, requires_grad=True)
        y = fn(t)
        g, = torch.autograd.grad(y, t)
        vals.append(y.item()); grads.append(g.item())
    return np.array(vals), np.array(grads)


def k1_normal(depths):
    def f_smooth(d):  # d = penetration depth (>0 below ground): foot_z = -d
        p = torch.cat([torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT), -d])
        return foot_contact_force_world(p[None], torch.zeros(1, 3, dtype=DT), PARAMS, mode="smooth")["f_n"].squeeze()
    def f_hard(d):
        p = torch.cat([torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT), -d])
        return foot_contact_force_world(p[None], torch.zeros(1, 3, dtype=DT), PARAMS, mode="hard")["f_n"].squeeze()
    vs, gs = _grad_curve(f_smooth, depths)
    vh, gh = _grad_curve(f_hard, depths)
    print(f"[K1 normal] smooth max|df_n/dd|={np.abs(gs).max():.1f}  hard max|df_n/dd|={np.abs(gh).max():.1f} "
          f"(hard has the step at d=0)")
    return vs, gs, vh, gh


def k2_friction(speeds):
    # foot penetrating 3mm (fixed f_n), sliding in +x at varying speed.
    def make(mode):
        def f(vx):
            p = torch.tensor([[0.0, 0.0, -0.003]], dtype=DT)
            v = torch.cat([vx, torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT)])[None]
            return foot_contact_force_world(p, v, PARAMS, mode=mode)["f_t"][0, 0].abs()
        return f
    vs, gs = _grad_curve(make("smooth"), speeds)
    vh, gh = _grad_curve(make("hard"), speeds)
    print(f"[K2 friction] smooth max|d|f_t|/dv|={np.abs(gs).max():.1f}  "
          f"hard max|d|f_t|/dv|={np.abs(gh).max():.1f} (hard ~inf at v=0)")
    return vs, gs, vh, gh


def k3_k4_cone_direction():
    torch.manual_seed(0)
    N = 4000
    p = torch.zeros(N, 3, dtype=DT); p[:, 2] = -torch.rand(N, dtype=DT) * 0.01   # 0..10mm pen
    v = torch.randn(N, 3, dtype=DT) * 0.5                                        # random 3D foot vel
    out = foot_contact_force_world(p, v, PARAMS, mode="smooth")
    ft = out["f_t"]; mufn = out["mu_fn"]; vt = out["v_t"]
    cone_ratio = torch.linalg.norm(ft, dim=-1, keepdim=True) / mufn.clamp_min(1e-9)
    cone_viol = (cone_ratio > 1.0 + 1e-6).float().mean().item()
    dissip = (ft * vt).sum(-1)
    dissip_viol = (dissip > 1e-9).float().mean().item()
    print(f"[K3 cone] max ||f_t||/(mu f_n)={cone_ratio.max().item():.4f}  violations={cone_viol*100:.2f}%")
    print(f"[K4 direction] f_t·v_t max={dissip.max().item():.2e}  non-dissipative={dissip_viol*100:.2f}%")
    return cone_ratio.squeeze(-1).numpy()


def k6_drop_settle(device="cpu"):
    dev = device
    kin = Go2Kinematics.from_urdf(device=dev, dtype=torch.float32)
    qj = torch.tensor([0.0, 0.9, -1.8] * 4, dtype=torch.float32)
    ci = composite_inertia(kin, qj)
    mass = ci.mass.detach().to(dev); I = ci.inertia.detach().to(dev); Iinv = torch.linalg.inv(I)
    c_body = ci.com.detach().to(dev)
    feet = kin.foot_positions(qj)[0].to(dev)               # (4,3) base frame
    foot_rel_com = (feet - c_body)                         # (4,3) relative to COM, body frame
    params = ContactParams()

    B = 1
    q = torch.zeros(B, 4, dtype=torch.float32, device=dev); q[:, 0] = 1.0
    p = torch.zeros(B, 3, dtype=torch.float32, device=dev); p[:, 2] = 0.40   # drop from 0.40 m
    s = FloatingBaseState(p, q, torch.zeros(B, 3, device=dev), torch.zeros(B, 3, device=dev))
    dt, steps = 1e-3, 2500
    zs, fns = [], []
    mg = (mass * 9.81).item()
    for _ in range(steps):
        R = quaternion_to_matrix(s.q)                                  # (B,3,3)
        fr = foot_rel_com.unsqueeze(0).expand(B, 4, 3)                 # (B,4,3) body
        foot_world = s.p[:, None, :] + torch.einsum("bij,bkj->bki", R, fr)         # (B,4,3)
        w_world = torch.einsum("bij,bj->bi", R, s.w)                   # body->world omega
        foot_vel = s.v[:, None, :] + torch.cross(w_world[:, None, :].expand(B, 4, 3),
                                                 foot_world - s.p[:, None, :], dim=-1)
        f_tot = torch.zeros(B, 3, device=dev); tau_tot = torch.zeros(B, 3, device=dev)
        sum_fn = 0.0
        for k in range(4):
            out = foot_contact_force_world(foot_world[:, k, :], foot_vel[:, k, :], params, mode="smooth")
            fw, tau_b = contact_to_body_wrench(foot_world[:, k, :], out["f_world"], s.p, s.q)
            f_tot = f_tot + fw; tau_tot = tau_tot + tau_b
            sum_fn += out["f_n"].sum().item()
        s = srbd_step(s, mass, I, Iinv, dt, f_world=f_tot, tau_body=tau_tot)
        zs.append(s.p[0, 2].item()); fns.append(sum_fn)
    rest_fn = np.mean(fns[-200:])
    print(f"[K6 drop-settle] rest COM z={zs[-1]:.4f} m  sum f_n(rest)={rest_fn:.1f} N  "
          f"m g={mg:.1f} N  balance err={abs(rest_fn-mg)/mg*100:.1f}%  (device {dev})")
    return np.arange(steps) * dt, np.array(zs), np.array(fns), mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    depths = np.linspace(-0.004, 0.012, 200)
    speeds = np.linspace(-0.3, 0.3, 200)
    vs, gs, vh, gh = k1_normal(depths)
    fvs, fgs, fvh, fgh = k2_friction(speeds)
    cone = k3_k4_cone_direction()
    t, zs, fns, mg = k6_drop_settle(args.device)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(depths * 1e3, vs, label="smooth f_n"); a.plot(depths * 1e3, vh, "--", label="hard f_n")
    a2 = a.twinx(); a2.plot(depths * 1e3, gs, "C2", alpha=.6, label="smooth df_n/dd")
    a2.plot(depths * 1e3, gh, "C3--", alpha=.6, label="hard df_n/dd")
    a.set_title("K1 normal force vs penetration (E1)"); a.set_xlabel("penetration (mm)"); a.set_ylabel("f_n (N)")
    a.legend(loc="upper left"); a2.legend(loc="lower right")

    a = ax[0, 1]
    a.plot(speeds, fvs, label="smooth |f_t|"); a.plot(speeds, fvh, "--", label="hard |f_t|")
    a2 = a.twinx(); a2.plot(speeds, fgs, "C2", alpha=.6, label="smooth grad"); a2.plot(speeds, fgh, "C3--", alpha=.6, label="hard grad")
    a.set_title("K2 friction vs tangential speed (F1)"); a.set_xlabel("v_t (m/s)"); a.set_ylabel("|f_t| (N)")
    a.legend(loc="upper left"); a2.legend(loc="lower right")

    a = ax[1, 0]
    a.hist(cone, bins=60, color="C0"); a.axvline(1.0, color="k", ls=":")
    a.set_title(f"K3 friction cone ||f_t||/(mu f_n) (max={cone.max():.3f} <= 1)")
    a.set_xlabel("cone ratio"); a.set_ylabel("count")

    a = ax[1, 1]
    a.plot(t, zs, label="COM height z"); a.set_xlabel("t (s)"); a.set_ylabel("z (m)")
    a2 = a.twinx(); a2.plot(t, fns, "C1", alpha=.7, label="Σ f_n"); a2.axhline(mg, color="C3", ls=":", label="m g")
    a.set_title("K6 drop-and-settle (SRBD + 4 feet)"); a.legend(loc="upper right"); a2.legend(loc="right")

    fig.suptitle("E3D-2 3D foot contact: smooth normal + friction cone (Go2 SRBD)", fontsize=14)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "e3d2_contact_checks.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
