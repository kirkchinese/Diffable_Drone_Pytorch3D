"""E3D-3 Stage B: closed-loop minimal differentiable training of 3D SRBD standing.

The 3D analog of 2D-F3: train a small MLP via BPTT through the differentiable SRBD + smooth
contact to regulate the floating base to a TARGET height (above the passive equilibrium, so
the policy is provably necessary) and LEVEL attitude from random perturbations.

Addresses the pre-registered pitfalls:
  #4 param scale     : gradient_sanity() checks BPTT grad-norm is bounded before training.
  #5 action space    : a = 4 leg extensions (tanh-bounded); forces emerge from contact.
  #6 policy necessity: target z* = rest + 0.04 (passive can't reach it); passive baseline + the
                       height gap the policy closes are reported; action magnitude logged.
  #3 cone saturation : cone occupancy ||f_t||/(mu f_n) monitored during eval.
  #8 GDecay drift    : eval rolls out 5x the train horizon to expose slow attitude/height drift.
  #9 fair compare    : hard contact uses the SAME net/recipe/clip; its grad norms are reported,
                       so any failure is shown to be gradient-borne, not a rigged setup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
from srbd_standing import (  # noqa: E402
    build_standing_config, standing_step, initial_state, level_error, roll_pitch_diag)
from contact_3d import ContactParams  # noqa: E402
from floating_base_srbd import FloatingBaseState  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
ACT_SCALE = 0.10           # max leg extension (m)
DELTA_Z = 0.04             # target height above passive rest (m) -> requires the policy


class Policy(nn.Module):
    def __init__(self, obs=9, hid=32, act=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act))

    def forward(self, obs):
        return ACT_SCALE * torch.tanh(self.net(obs))


def observe(state: FloatingBaseState, z_star: float):
    R = quaternion_to_matrix(state.q)
    tilt = R[:, 2, :2]                      # world-z components in body (x,y): smooth tilt, no Euler
    return torch.cat([(state.p[:, 2:3] - z_star), tilt, state.v, state.w], dim=-1)


def sample_init(cfg, B, gen):
    """Batched random init: roll/pitch ±0.08 rad, height rest ±0.03 m, zero velocity."""
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
    dev, dt = cfg.device, cfg.dtype
    roll = (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.16
    pitch = (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.16
    h = cfg.rest_height + (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.06
    ang = torch.zeros(B, 3, device=dev, dtype=dt); ang[:, 0] = roll; ang[:, 1] = pitch
    q = matrix_to_quaternion(euler_angles_to_matrix(ang, "XYZ"))
    p = torch.zeros(B, 3, device=dev, dtype=dt); p[:, 2] = h
    return FloatingBaseState(p, q, torch.zeros(B, 3, device=dev, dtype=dt), torch.zeros(B, 3, device=dev, dtype=dt))


def rollout_loss(policy, cfg, state, z_star, horizon, mode, grad_decay, collect=False):
    B = state.p.shape[0]
    loss = state.p.new_zeros(())
    traj = []
    for t in range(horizon):
        a = policy(observe(state, z_star))
        state, info = standing_step(state, a, cfg, mode=mode, grad_decay=grad_decay)
        h_err = (state.p[:, 2] - z_star) ** 2
        att = level_error(state.q)
        vel = (state.v ** 2).sum(-1) + 0.1 * (state.w ** 2).sum(-1)
        horiz = (state.p[:, :2] ** 2).sum(-1)
        loss = loss + (4.0 * h_err + 2.0 * att + 0.05 * vel + 0.2 * horiz + 1e-3 * (a ** 2).sum(-1)).mean()
        if collect:
            traj.append(dict(z=state.p[:, 2].mean().item(),
                             att_deg=float(np.degrees(level_to_angle(att.mean().item()))),
                             cone=info["cone"].max().item(), act=a.abs().mean().item()))
    return loss / horizon, state, traj


def level_to_angle(le):  # le = 1 - cos(tilt) -> tilt angle
    return float(np.arccos(np.clip(1.0 - le, -1, 1)))


def gradient_sanity(cfg, horizon=200):
    """Pitfall #4: BPTT grad-norm through the rollout, smooth vs hard (untrained policy)."""
    out = {}
    for mode in ("smooth", "hard"):
        torch.manual_seed(0)
        pol = Policy().to(cfg.device, cfg.dtype)
        gen = torch.Generator(device=cfg.device).manual_seed(1)
        s = sample_init(cfg, 16, gen)
        loss, _, _ = rollout_loss(pol, cfg, s, cfg.rest_height + DELTA_Z, horizon, mode, 1.0)
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in pol.parameters())).item()
        out[mode] = gnorm
    print(f"[grad sanity] BPTT grad-norm (H={horizon}, untrained): smooth={out['smooth']:.2e}  hard={out['hard']:.2e}")
    return out


def train(cfg, mode="smooth", grad_decay=0.9, horizon=300, iters=80, B=64, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = Policy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    z_star = cfg.rest_height + DELTA_Z
    hist = {"loss": [], "gnorm": []}
    for it in range(iters):
        s = sample_init(cfg, B, gen)
        loss, _, _ = rollout_loss(pol, cfg, s, z_star, horizon, mode, grad_decay)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(pol.parameters(), clip).item()
        opt.step()
        hist["loss"].append(loss.item()); hist["gnorm"].append(gnorm)
    return pol, hist


@torch.no_grad()
def evaluate(policy, cfg, horizon, label, z_star=None, seed=7):
    z_star = z_star if z_star is not None else cfg.rest_height + DELTA_Z
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, 32, gen)
    zs, atts, cones = [], [], []
    for t in range(horizon):
        a = policy(observe(s, z_star)) if policy is not None else torch.zeros(32, 4, device=cfg.device, dtype=cfg.dtype)
        s, info = standing_step(s, a, cfg, mode="smooth", grad_decay=1.0)
        zs.append(s.p[:, 2].mean().item())
        atts.append(np.degrees(level_to_angle(level_error(s.q).mean().item())))
        cones.append(info["cone"].max().item())
    h_err = abs(zs[-1] - z_star)
    print(f"   [{label}] final z={zs[-1]:.4f} (target {z_star:.4f}, err={h_err*1e3:.1f}mm)  "
          f"final tilt={atts[-1]:.3f}°  max cone={max(cones):.3f}")
    return dict(zs=np.array(zs), atts=np.array(atts), cones=np.array(cones), z_star=z_star, h_err=h_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32, contact=ContactParams(), dt=1e-3)
    H = 300
    print(f"E3D-3 standing on {args.device}: rest={cfg.rest_height:.4f} m, target z*={cfg.rest_height+DELTA_Z:.4f} m "
          f"(+{DELTA_Z*1e3:.0f}mm, passive cannot reach)\n")

    gsan = gradient_sanity(cfg, horizon=H)

    print("\nTraining runs:")
    runs = {}
    runs["smooth+GDecay"] = train(cfg, "smooth", grad_decay=0.9, horizon=H, iters=args.iters)
    runs["smooth+noGDecay"] = train(cfg, "smooth", grad_decay=1.0, horizon=H, iters=args.iters)
    runs["smooth+shortH"] = train(cfg, "smooth", grad_decay=0.9, horizon=80, iters=args.iters)
    runs["hard+GDecay"] = train(cfg, "hard", grad_decay=0.9, horizon=H, iters=args.iters)
    for k, (_, h) in runs.items():
        print(f"   {k:16s} loss {h['loss'][0]:.3f}->{h['loss'][-1]:.3f}  "
              f"median grad-norm(preclip)={np.median(h['gnorm']):.2e}")

    print("\nEvaluation (long-horizon 5x = 1500 steps, exposes drift):")
    LONG = 1500
    evals = {}
    evals["passive"] = evaluate(None, cfg, LONG, "passive (zero policy)")
    for k, (pol, _) in runs.items():
        evals[k] = evaluate(pol, cfg, LONG, k)

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    for k, (_, h) in runs.items():
        a.plot(h["loss"], label=k)
    a.set_title("training loss (BPTT)"); a.set_yscale("log"); a.set_xlabel("iter"); a.legend(fontsize=8)
    a = ax[0, 1]
    for k, (_, h) in runs.items():
        a.plot(h["gnorm"], label=k, alpha=.8)
    a.set_title("grad-norm (pre-clip): smooth bounded vs hard"); a.set_yscale("log"); a.set_xlabel("iter"); a.legend(fontsize=8)
    a = ax[1, 0]
    t = np.arange(LONG) * cfg.dt
    for k in ("passive", "smooth+GDecay", "hard+GDecay"):
        a.plot(t, evals[k]["zs"], label=k)
    a.axhline(cfg.rest_height + DELTA_Z, color="k", ls=":", label="target z*")
    a.axhline(cfg.rest_height, color="gray", ls="--", label="passive rest")
    a.set_title("long rollout: COM height (policy necessity)"); a.set_xlabel("t (s)"); a.set_ylabel("z (m)"); a.legend(fontsize=8)
    a = ax[1, 1]
    for k in ("passive", "smooth+GDecay", "smooth+shortH", "hard+GDecay"):
        a.plot(t, evals[k]["atts"], label=k)
    a.set_title("long rollout: tilt angle (GDecay/horizon drift check)"); a.set_xlabel("t (s)"); a.set_ylabel("tilt (deg)")
    a.set_yscale("log"); a.legend(fontsize=8)
    fig.suptitle("E3D-3 3D SRBD standing: differentiable training through smooth contact", fontsize=14)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    out = FIG / "e3d3_standing_train.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    import json
    summary = {"grad_sanity": gsan, "rest_height": cfg.rest_height, "z_star": cfg.rest_height + DELTA_Z,
               "runs": {k: {"loss_final": h["loss"][-1], "grad_median": float(np.median(h["gnorm"]))}
                        for k, (_, h) in runs.items()},
               "eval": {k: {"final_z": float(e["zs"][-1]), "h_err_mm": float(e["h_err"] * 1e3),
                            "final_tilt_deg": float(e["atts"][-1]), "max_cone": float(e["cones"].max())}
                        for k, e in evals.items()}}
    (RESULTS / "e3d3_standing.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}\nsaved {RESULTS/'e3d3_standing.json'}")


if __name__ == "__main__":
    main()
