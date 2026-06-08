"""
约束接触力残差: 平滑摩擦锥 + 残差幅值限制(防接管) + 门控 + 法向非负  [R4]
==========================================================================
在 R3(srbm_residual_contact) 基础上加两条**结构性、可微**的物理约束:

  法向(乘性有界 -> 非负 + 限幅 + 自门控):
      F_n = F_n_phys · (1 + α·tanh(ΔF_n))          α=0.6  (法向最多被修正 ±60%)
      => 0 ≤ F_n ≤ (1+α)F_n_phys, 残差**结构上不可能接管/置零/吸地**

  切向(平滑摩擦锥 + 门控):
      F_t = μF_n · tanh( (F_t_phys + gate·ΔF_t) / (μF_n + ε) )
      => 结构上 |F_t| ≤ μF_n; 且 F_n=0 时 F_t=0 (自门控)

监控量(实验 R4):
  ① 摩擦锥占用率 |F_t|/(μF_n)        —— 约束版应 ≤1; 无约束版会越界
  ② 法向相对修正 |F_n-F_n_phys|/F_n_phys —— 限幅版应 ≤α; 不限幅版会 >1(接管)
  ③ 梯度保真(余弦 vs 真 J_T)          —— 加约束是否伤梯度(预期: 几乎不伤)
  ④ 离地法向力                        —— 应 ≈0

teacher/评估协议复用 srbm_residual; 每足 MLP 复用 srbm_residual_contact.ContactResidual.net(取未缩放 logits)。
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import srbm_dynamics as srb
from srbm_dynamics import SRBMParams
from srbm_residual import (NOMINAL, REAL, sample_XU, accel_from_XU, jac_autograd,
                           grad_metrics, fwd_rms)
from srbm_residual_contact import ContactResidual, sample_airborne

# R4 用**更大失配**的 teacher, 把约束"压满"以展示其价值(否则温和失配下门控已够, 约束未启用)。
# mu_real=1.3 > 标称 0.9: 真实切向力超出标称摩擦锥 -> 锥约束被迫"压满"(并暴露"先验过紧"的代价)。
REAL = SRBMParams(m=13.0, I=0.50, k_n=12000.0, k_d=55.0, mu=1.30)

C_NOM, C_R3, C_R4, C_UNC = "#8C8C8C", "#55A868", "#4C72B0", "#C44E52"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
G = srb.G
ALPHA_N = 0.6   # 法向最大相对修正
EPS_T = 1e-6


def contact_accel_c(XU, p, residual=None, gated=True, cone=True, alpha_n=ALPHA_N,
                    return_diag=False):
    """约束接触力残差的 student 加速度。

    alpha_n=None -> 法向用 R3 的加性 + 非负 clamp(无幅值上限);
    alpha_n=值   -> 法向用乘性有界(限幅).  cone=True -> 切向加平滑摩擦锥.
    """
    px, pz, th = XU[:, 0], XU[:, 1], XU[:, 2]
    vx, vz, om = XU[:, 3], XU[:, 4], XU[:, 5]
    base_x = [p.half_len, -p.half_len]
    Fx = torch.zeros_like(px); Fz = torch.zeros_like(px); tau = torch.zeros_like(px)
    relcorr, coneratio, Fns = [], [], []
    for k, bx in enumerate(base_x):
        ext_k = XU[:, 6 + k]
        fx, fz = srb.foot_world(px, pz, th, ext_k, bx, p.leg_nominal)
        rx, rz = fx - px, fz - pz
        vfx = vx - om * rz
        vfz = vz + om * rx
        pen = p.eps * F.softplus(-fz / p.eps)
        gate = torch.sigmoid(-fz / p.eps)
        Fn_phys = torch.clamp(p.k_n * pen - p.k_d * vfz * gate, min=0.0)
        Ft_phys = -p.mu * Fn_phys * torch.tanh(vfx / p.v_eps)
        if residual is not None:
            feat = torch.stack([pz, th, vx, vz, om, ext_k, fz, gate, vfz, vfx,
                                Fn_phys, Ft_phys], dim=-1)
            d = residual.net(feat)   # 取未缩放 logits(绕过 ContactResidual 的 OUT_SCALE)
            dFn, dFt = d[:, 0], d[:, 1]
            # ---- 法向 ----
            if alpha_n is not None:
                Fn = Fn_phys * (1.0 + alpha_n * torch.tanh(dFn))        # 乘性有界(限幅+非负+自门控)
            else:
                dFn_g = dFn * gate if gated else dFn
                Fn = torch.clamp(Fn_phys + dFn_g, min=0.0)             # R3 加性(无上限)
            # ---- 切向 ----
            dFt_g = dFt * gate if gated else dFt
            raw_t = Ft_phys + dFt_g
            if cone:
                muFn = p.mu * Fn
                Ft = muFn * torch.tanh(raw_t / (muFn + EPS_T))         # 平滑摩擦锥
            else:
                Ft = raw_t
            relcorr.append((Fn - Fn_phys) / (Fn_phys + 1e-6))
            coneratio.append(Ft.abs() / (p.mu * Fn + 1e-6))
        else:
            Fn, Ft = Fn_phys, Ft_phys
            relcorr.append(torch.zeros_like(px)); coneratio.append(Ft.abs() / (p.mu * Fn + 1e-6))
        Fns.append(Fn)
        Fx = Fx + Ft; Fz = Fz + Fn; tau = tau + (rx * Fn - rz * Ft)
    A = torch.stack([Fx / p.m, Fz / p.m - G, tau / p.I], dim=-1)
    if return_diag:
        return A, {"relcorr": torch.stack(relcorr, -1), "coneratio": torch.stack(coneratio, -1),
                   "Fn": torch.stack(Fns, -1)}
    return A


def jac_c(XU, p, residual, gated, cone, alpha_n):
    XU = XU.clone().requires_grad_(True)
    A = contact_accel_c(XU, p, residual, gated, cone, alpha_n)
    Js = [torch.autograd.grad(A[:, j].sum(), XU, retain_graph=(j < 2))[0] for j in range(3)]
    return torch.stack(Js, 1).detach()


def train_c(n_iters, dev, scale, gated, cone, alpha_n, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    r = ContactResidual().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = accel_from_XU(XU, REAL, "smooth")
        A_S = contact_accel_c(XU, NOMINAL, r, gated, cone, alpha_n)
        loss = (((A_S - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return r


def monitors(r, dev, eval_XU, XU_air, J_teacher, gated, cone, alpha_n):
    cos, _, _ = grad_metrics(jac_c(eval_XU, NOMINAL, r, gated, cone, alpha_n), J_teacher)
    with torch.no_grad():
        _, dg = contact_accel_c(eval_XU, NOMINAL, r, gated, cone, alpha_n, return_diag=True)
        _, dair = contact_accel_c(XU_air, NOMINAL, r, gated, cone, alpha_n, return_diag=True)
        A_T = accel_from_XU(eval_XU, REAL, "smooth")
        A_S = contact_accel_c(eval_XU, NOMINAL, r, gated, cone, alpha_n)
    return {
        "grad_cos": cos,
        "fwd_rms": float(((A_S - A_T) ** 2).mean().sqrt()),
        "cone_ratio_max": float(dg["coneratio"].max()),
        "cone_violation_frac": float((dg["coneratio"] > 1.001).float().mean()),
        "relcorr_abs_max": float(dg["relcorr"].abs().max()),
        "relcorr_abs_mean": float(dg["relcorr"].abs().mean()),
        "airborne_Fn": float(dair["Fn"].abs().mean()),
        "_relcorr": dg["relcorr"].flatten().cpu().numpy(),
        "_coneratio": dg["coneratio"].flatten().cpu().numpy(),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 250 if args.quick else 1500
    print(f"device={dev} n_iters={n_iters}")

    eval_XU = sample_XU(512, dev); XU_air = sample_airborne(512, dev)
    J_teacher = jac_autograd(eval_XU, REAL, "smooth")
    big = sample_XU(4096, dev)
    with torch.no_grad():
        scale = (accel_from_XU(big, REAL, "smooth")
                 - accel_from_XU(big, NOMINAL, "smooth")).std(dim=0).clamp_min(1e-6)

    # 三配置: 无约束(加性,无锥,无门控) / R3门控(加性+非负+门控,无锥无限幅) / R4全约束(限幅+锥+门控)
    cfgs = {
        "unconstrained": dict(gated=False, cone=False, alpha_n=None),
        "R3 gated":      dict(gated=True,  cone=False, alpha_n=None),
        "R4 constrained": dict(gated=True, cone=True,  alpha_n=ALPHA_N),
    }
    res = {}
    mons = {}
    for name, c in cfgs.items():
        r = train_c(n_iters, dev, scale, seed=0, **c)
        m = monitors(r, dev, eval_XU, XU_air, J_teacher, **c)
        mons[name] = m
        res[name] = {k: v for k, v in m.items() if not k.startswith("_")}
        print(f"[{name}] cos={m['grad_cos']:.4f} fwd={m['fwd_rms']:.2f} "
              f"coneMax={m['cone_ratio_max']:.2f} coneViol={m['cone_violation_frac']:.3f} "
              f"relcorrMax={m['relcorr_abs_max']:.2f} airFn={m['airborne_Fn']:.3f}")
    json.dump(res, open(os.path.join(HERE, "results_phase3c.json"), "w"), indent=2, ensure_ascii=False)

    # ---------- 图 R4 ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    # (a) 摩擦锥占用率分布
    ax[0].hist(np.clip(mons["unconstrained"]["_coneratio"], 0, 3), bins=60, color=C_UNC,
               alpha=.6, label="unconstrained")
    ax[0].hist(np.clip(mons["R4 constrained"]["_coneratio"], 0, 3), bins=60, color=C_R4,
               alpha=.6, label="R4 constrained")
    ax[0].axvline(1.0, color="k", ls="--", lw=1.2, label="friction-cone limit |Ft|=μFn")
    ax[0].set_xlabel(r"cone occupancy $|F_t|/(\mu F_n)$"); ax[0].set_ylabel("count")
    ax[0].set_title("(a) friction cone: constrained never exceeds 1"); ax[0].legend(fontsize=7)
    # (b) 法向相对修正分布
    ax[1].hist(np.clip(mons["unconstrained"]["_relcorr"], -2, 2), bins=60, color=C_UNC, alpha=.6,
               label="unconstrained (takes over)")
    ax[1].hist(np.clip(mons["R4 constrained"]["_relcorr"], -2, 2), bins=60, color=C_R4, alpha=.6,
               label=f"R4 (cap ±{ALPHA_N})")
    for s in (-ALPHA_N, ALPHA_N):
        ax[1].axvline(s, color="k", ls="--", lw=1.0)
    ax[1].set_xlabel(r"normal relative correction $(F_n-F_n^{phys})/F_n^{phys}$")
    ax[1].set_ylabel("count"); ax[1].set_title("(b) magnitude cap: residual can't take over")
    ax[1].legend(fontsize=7)
    # (c) 加约束不伤梯度 + 离地力
    names = list(cfgs.keys()); cosv = [mons[n]["grad_cos"] for n in names]
    ax[2].bar(range(3), cosv, color=[C_UNC, C_R3, C_R4])
    ax[2].set_ylim(0.9, 1.005); ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(names, fontsize=8, rotation=10)
    ax[2].set_ylabel("gradient cosine to TRUE J")
    ax[2].set_title("(c) constraints barely cost gradient fidelity")
    for i, n in enumerate(names):
        ax[2].text(i, cosv[i], f"{cosv[i]:.3f}\nairFn={mons[n]['airborne_Fn']:.2f}",
                   ha="center", va="bottom", fontsize=7)
    fig.suptitle("R4  Smooth friction cone + residual magnitude cap: structural physical legality "
                 "at ~zero gradient cost", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R4_constrained_residual.png"), bbox_inches="tight")
    plt.close(fig)
    print("[R4] figure saved.")


if __name__ == "__main__":
    main()
