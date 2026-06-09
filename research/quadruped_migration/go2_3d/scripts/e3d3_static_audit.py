"""E3D-3 Stage A: static-equilibrium audit (run BEFORE training a standing policy).

Confirms the things that Sigma f_n = m g does NOT: net torque, attitude drift, rest height
provenance, per-foot load split, and foot-naming/lever-sign correctness. Addresses the
pre-training pitfalls:
  A1 foot naming/signs    -> body-frame foot positions match FL:+x+y, FR:+x-y, RL:-x+y, RR:-x-y.
  A2 settle to equilibrium -> from a drop, COM z, roll, pitch settle; net_force->[0,0,mg],
                              net_tau_body->0 (NOT just vertical balance).
  A3 rest height geometry -> settled z matches -foot_z_rel_com - m g/(4 k_n).
  A4 load redistribution  -> a PITCH perturbation makes a FRONT/BACK load split (not L/R);
                              a ROLL perturbation makes a LEFT/RIGHT split (not F/B).
                              (If naming/levers were wrong, pitch would leak into L/R.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
from srbd_standing import (  # noqa: E402
    build_standing_config, standing_step, initial_state, level_error, roll_pitch_diag)
from contact_3d import ContactParams  # noqa: E402

FIG = HERE.parent / "figures"
LEGS = ["FL", "FR", "RL", "RR"]


def a1_foot_naming(cfg):
    fr = cfg.foot_rel_com.cpu().numpy()
    print("[A1 foot naming] body-frame foot positions relative to COM:")
    ok = True
    expect = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}
    for i, leg in enumerate(LEGS):
        x, y, z = fr[i]
        sx, sy = np.sign(x), np.sign(y)
        good = (sx == expect[leg][0] and sy == expect[leg][1])
        ok = ok and good
        print(f"   {leg}: x={x:+.4f} y={y:+.4f} z={z:+.4f}  sign({'+x' if sx>0 else '-x'},{'+y' if sy>0 else '-y'}) {'OK' if good else 'WRONG'}")
    print(f"   naming/sign pattern correct: {ok}")
    return ok


def settle(cfg, height, roll=0.0, pitch=0.0, steps=2000):
    s = initial_state(cfg, B=1, height=height, roll=pitch * 0 + roll, pitch=pitch)
    rec = {k: [] for k in ("z", "roll", "pitch", "nf", "ntau", "fn")}
    for _ in range(steps):
        s, info = standing_step(s, torch.zeros(1, 4, device=cfg.device, dtype=cfg.dtype), cfg)
        r, p = roll_pitch_diag(s.q)
        rec["z"].append(s.p[0, 2].item()); rec["roll"].append(r.item()); rec["pitch"].append(p.item())
        rec["nf"].append(info["net_force_world"][0].detach().cpu().numpy())
        rec["ntau"].append(info["net_tau_body"][0].detach().cpu().numpy())
        rec["fn"].append(info["f_n"][0].detach().cpu().numpy())
    return {k: np.array(v) for k, v in rec.items()}


def main():
    cfg = build_standing_config(device="cpu", dtype=torch.float64,
                                contact=ContactParams(), dt=1e-3)
    mg = cfg.mass.item() * 9.81
    print(f"SRBD standing: mass={cfg.mass.item():.3f} kg, m g={mg:.2f} N, "
          f"analytic rest height={cfg.rest_height:.4f} m\n")

    a1_foot_naming(cfg)

    # A2/A3: drop from above rest, settle, check equilibrium.
    rec = settle(cfg, height=cfg.rest_height + 0.03, steps=2500)
    z_rest = rec["z"][-200:].mean()
    nf_rest = rec["nf"][-200:].mean(0)
    ntau_rest = rec["ntau"][-200:].mean(0)
    fn_rest = rec["fn"][-200:].mean(0)
    print(f"\n[A2 equilibrium] settled: z={z_rest:.4f} m  roll={rec['roll'][-1]:+.2e}  pitch={rec['pitch'][-1]:+.2e}")
    print(f"   net force world = [{nf_rest[0]:+.3f}, {nf_rest[1]:+.3f}, {nf_rest[2]:+.2f}] N (expect ~[0,0,{mg:.1f}])")
    print(f"   net tau body    = [{ntau_rest[0]:+.4f}, {ntau_rest[1]:+.4f}, {ntau_rest[2]:+.4f}] N·m (expect ~0)")
    print(f"[A3 rest height] settled z={z_rest:.4f} vs analytic {cfg.rest_height:.4f}  "
          f"(err {abs(z_rest-cfg.rest_height)*1e3:.2f} mm)")
    print(f"   per-foot f_n: " + "  ".join(f"{l}={v:.1f}N" for l, v in zip(LEGS, fn_rest))
          + f"  (sum={fn_rest.sum():.1f}, front={fn_rest[0]+fn_rest[1]:.1f}, back={fn_rest[2]+fn_rest[3]:.1f})")

    # A4: load-redistribution / naming validation via pitch and roll perturbations.
    print("\n[A4 load redistribution] early-time per-foot load under attitude perturbation:")
    for label, kw in [("pitch +0.10", dict(pitch=0.10)), ("roll +0.10", dict(roll=0.10))]:
        s = initial_state(cfg, B=1, height=cfg.rest_height, **kw)
        s, info = standing_step(s, torch.zeros(1, 4, device=cfg.device, dtype=cfg.dtype), cfg)
        fn = info["f_n"][0].detach().cpu().numpy()
        fb = (fn[0] + fn[1]) - (fn[2] + fn[3])      # front - back
        lr = (fn[0] + fn[2]) - (fn[1] + fn[3])      # left - right
        print(f"   {label}: f_n={np.round(fn,1).tolist()}  front-back={fb:+.1f}N  left-right={lr:+.1f}N")
    print("   -> pitch should drive front-back (not left-right); roll should drive left-right (not front-back).")

    # figure
    t = np.arange(len(rec["z"])) * cfg.dt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t, rec["z"]); ax[0].axhline(cfg.rest_height, color="C3", ls=":", label="analytic rest")
    ax[0].set_title("COM height settle"); ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("z (m)"); ax[0].legend()
    ax[1].plot(t, np.degrees(rec["roll"]), label="roll"); ax[1].plot(t, np.degrees(rec["pitch"]), label="pitch")
    ax[1].set_title("attitude drift -> 0"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("deg"); ax[1].legend()
    ax[2].plot(t, rec["ntau"][:, 0], label="tau_x (roll)"); ax[2].plot(t, rec["ntau"][:, 1], label="tau_y (pitch)")
    ax[2].plot(t, rec["ntau"][:, 2], label="tau_z (yaw)")
    ax[2].set_title("net torque about COM -> 0"); ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("N·m"); ax[2].legend()
    fig.suptitle("E3D-3 Stage A: static-equilibrium audit (Go2 SRBD standing)", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "e3d3_static_audit.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
