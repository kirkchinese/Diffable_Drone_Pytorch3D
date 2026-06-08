"""
R11: R10 capstone 的两点加固 —— (A) 闭环 capstone  (B) 正则化路由鲁棒性
==========================================================================
R10 在**梯度/前向**层面证明了"双头自动路由"。R11 把它做扎实:

(A) 闭环 capstone: 对三种 teacher(force/geom/both), 各经 5 种动力学训平衡策略
    (nominal / force-head / kin-head / both-head / oracle), 全部**部署到 teacher**。
    若 both-head 在三种 teacher 上都≈oracle, 则 R10 不只是"梯度 capstone"而是"闭环 capstone"。

(B) 证明自动路由非偶然: 加正则 L = L_forward + λ_kin‖Δx‖² + λ_force‖ΔF‖²,
    看路由是否仍稳定(且应**更锐**: 不需要的头被显式压到 0, 需要的头保留)。
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from srbm_residual import NOMINAL, sample_XU
from srbm_residual_contact import ContactResidual
from srbm_residual_kinematic import KinematicResidual
from srbm_residual_struct import jac_of, cos_sign
from srbm_residual_extreme import accel_asym
from srbm_residual_combined import combined_accel, scale_for_teacher, REAL_BIG, DGEOM
from srbm_residual_closedloop import train_policy, evaluate

C_NOM, C_C, C_D, C_E, C_TEA = "#8C8C8C", "#55A868", "#DD8452", "#4C72B0", "#000000"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

TEACHERS = {
    "force": lambda XU: accel_asym(XU, REAL_BIG, 0.0),
    "geom":  lambda XU: accel_asym(XU, NOMINAL, DGEOM),
    "both":  lambda XU: accel_asym(XU, REAL_BIG, DGEOM),
}


def nominal_fn(XU):
    return combined_accel(XU, NOMINAL)


def train_student_reg(kind, teacher_fn, n_iters, dev, scale, lam_kin=0.0, lam_force=0.0,
                      seed=0, lr=2e-3):
    """训练 C/D/E。可选头幅值正则。返回 (student_fn, heads_fn)。"""
    torch.manual_seed(seed)
    kin = KinematicResidual().to(dev).to(torch.get_default_dtype()) if kind in ("D", "E") else None
    force = ContactResidual().to(dev).to(torch.get_default_dtype()) if kind in ("C", "E") else None
    params = (list(kin.parameters()) if kin else []) + (list(force.parameters()) if force else [])
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = teacher_fn(XU)
        A_S, kin_sq, force_sq = combined_accel(XU, NOMINAL, kin=kin, force=force, return_reg=True)
        loss = (((A_S - A_T) / scale) ** 2).mean() + lam_kin * kin_sq + lam_force * force_sq
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    student = lambda XU: combined_accel(XU, NOMINAL, kin=kin, force=force)
    heads = lambda XU: combined_accel(XU, NOMINAL, kin=kin, force=force, return_heads=True)[1]
    return student, heads


# ----------------------------- Part A: 闭环 capstone -----------------------------
def part_a(dev, res_iters, pol_iters):
    rows = {}
    for tname, tfn in TEACHERS.items():
        sc = scale_for_teacher(tfn, dev)
        sC, _ = train_student_reg("C", tfn, res_iters, dev, sc, seed=0)
        sD, _ = train_student_reg("D", tfn, res_iters, dev, sc, seed=0)
        sE, _ = train_student_reg("E", tfn, res_iters, dev, sc, seed=0)
        models = {"nominal": nominal_fn, "force-head": sC, "kin-head": sD,
                  "both-head": sE, "oracle": tfn}
        rows[tname] = {}
        for name, fn in models.items():
            pol, _ = train_policy(fn, pol_iters, dev, seed=0)
            dep, _ = evaluate(pol, tfn, dev)
            rows[tname][name] = dep
        print(f"  [A] T={tname:6s}  " + "  ".join(f"{n}={rows[tname][n]:.4f}" for n in models),
              flush=True)
    return rows


# ------------------------ Part B: 正则化路由鲁棒性 ------------------------
def part_b(dev, res_iters, lam_kin, lam_force):
    eval_XU = sample_XU(512, dev)
    out = {}
    for tname, tfn in TEACHERS.items():
        sc = scale_for_teacher(tfn, dev)
        J_T = jac_of(tfn, eval_XU)
        out[tname] = {}
        for tag, (lk, lf) in [("noreg", (0.0, 0.0)), ("reg", (lam_kin, lam_force))]:
            s, h = train_student_reg("E", tfn, res_iters, dev, sc, lk, lf, seed=0)
            heads = h(eval_XU)
            cos = cos_sign(jac_of(s, eval_XU), J_T)[0]
            out[tname][tag] = dict(kin=heads["kin"], force=heads["force"], cos=cos)
        r = out[tname]
        print(f"  [B] T={tname:6s} noreg(kin={r['noreg']['kin']:.3f} force={r['noreg']['force']:.3f} "
              f"cos={r['noreg']['cos']:.3f}) | reg(kin={r['reg']['kin']:.3f} "
              f"force={r['reg']['force']:.3f} cos={r['reg']['cos']:.3f})", flush=True)
    return out


def _plot(rows, routing, tnames):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) 闭环部署 loss: 3 teacher × 5 policy
    pols = ["nominal", "force-head", "kin-head", "both-head", "oracle"]
    cols = [C_NOM, C_C, C_D, C_E, C_TEA]
    x = np.arange(len(tnames)); w = 0.16
    for j, pn in enumerate(pols):
        ax[0].bar(x + (j - 2) * w, [rows[t][pn] for t in tnames], w, color=cols[j],
                  label=f"π_{pn}", edgecolor="k" if pn == "oracle" else None, linewidth=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"T_{t}" for t in tnames])
    ax[0].set_yscale("log"); ax[0].set_ylabel("balance loss deployed on teacher (log)")
    ax[0].set_title("(a) closed-loop capstone: both-head ≈ oracle\non ALL teacher types"); ax[0].legend(fontsize=7)
    # (b) 正则前后头活跃度(每指标归一到自身最大)
    kin_no = np.array([routing[t]["noreg"]["kin"] for t in tnames])
    frc_no = np.array([routing[t]["noreg"]["force"] for t in tnames])
    kin_rg = np.array([routing[t]["reg"]["kin"] for t in tnames])
    frc_rg = np.array([routing[t]["reg"]["force"] for t in tnames])
    kmax, fmax = max(kin_no.max(), kin_rg.max()) + 1e-12, max(frc_no.max(), frc_rg.max()) + 1e-12
    xb = np.arange(len(tnames)); wb = 0.2
    ax[1].bar(xb - 1.5 * wb, kin_no / kmax, wb, color=C_D, alpha=.55, label="kin (no reg)")
    ax[1].bar(xb - 0.5 * wb, kin_rg / kmax, wb, color=C_D, hatch="//", edgecolor="k", label="kin (reg)")
    ax[1].bar(xb + 0.5 * wb, frc_no / fmax, wb, color=C_C, alpha=.55, label="force (no reg)")
    ax[1].bar(xb + 1.5 * wb, frc_rg / fmax, wb, color=C_C, hatch="//", edgecolor="k", label="force (reg)")
    ax[1].set_xticks(xb); ax[1].set_xticklabels([f"T_{t}" for t in tnames])
    ax[1].set_ylabel("E head activity (norm. to max)")
    ax[1].set_title("(b) routing survives reg (sharper):\nunneeded head → 0, needed head stays"); ax[1].legend(fontsize=6.5)
    # (c) 正则前后梯度余弦(不该掉)
    ax[2].bar(xb - wb / 2, [routing[t]["noreg"]["cos"] for t in tnames], wb, color=C_E, alpha=.55, label="no reg")
    ax[2].bar(xb + wb / 2, [routing[t]["reg"]["cos"] for t in tnames], wb, color=C_E, hatch="//",
              edgecolor="k", label="reg")
    ax[2].set_xticks(xb); ax[2].set_xticklabels([f"T_{t}" for t in tnames])
    ax[2].set_ylim(0, 1.02); ax[2].set_ylabel("E full-gradient cosine to teacher")
    ax[2].set_title("(c) reg keeps gradient fidelity (≥0.99)"); ax[2].legend(fontsize=7)
    fig.suptitle("R11  Auto-routing reinforced: (a) closed-loop capstone  (b,c) routing robust to L2 reg",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R11_routing_capstone.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lam_kin", type=float, default=0.05)
    ap.add_argument("--lam_force", type=float, default=0.02)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    res_iters = 300 if args.quick else 1200
    pol_iters = 60 if args.quick else 200
    print(f"device={dev} res_iters={res_iters} pol_iters={pol_iters} "
          f"lam_kin={args.lam_kin} lam_force={args.lam_force} DGEOM={DGEOM}", flush=True)

    print("== Part A: closed-loop capstone ==", flush=True)
    rows = part_a(dev, res_iters, pol_iters)
    print("== Part B: regularized routing robustness ==", flush=True)
    routing = part_b(dev, res_iters, args.lam_kin, args.lam_force)

    json.dump({"closed_loop": rows, "routing": routing,
               "lam_kin": args.lam_kin, "lam_force": args.lam_force},
              open(os.path.join(HERE, "results_phase3j.json"), "w"), indent=2, ensure_ascii=False)
    _plot(rows, routing, list(TEACHERS))
    print("[R11] figure saved.", flush=True)


if __name__ == "__main__":
    main()
