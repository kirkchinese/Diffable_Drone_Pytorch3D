"""
接触力残差 (Contact-Force Residual) + 接触门控 + 法向非负约束  [R3]
==========================================================================
最贴近四足误差源的残差形式: 不在加速度上补, 而在**每足接触力**上补:
    F_n = clamp( F_n_phys + gate·ΔF_n , 0 )      (法向: 门控 + 非负)
    F_t =        F_t_phys + gate·ΔF_t            (切向: 门控)
gate = contact_weight = sigmoid(-fz/eps) (离地 -> 0 -> 残差不凭空给地面力)。

每足共享一个 MLP(腿同构归纳偏置), 输入(每足):
  机身: pz, θ, vx, vz, ω
  动作: ext_i
  接触特征: gap_i(=fz_i), contact_weight_i, foot_vn_i, foot_vt_i
  局部物理量: F_n_phys_i, F_t_phys_i
输出(每足): ΔF_n_i, ΔF_t_i

teacher/评估协议复用 srbm_residual (参数失配 teacher -> 真梯度 J_T 解析可得)。

实验:
  R3a 梯度保真: 接触力残差(门控) vs 加速度残差 vs nominal (对照真 J_T)
  R3b 门控消融: 在"明确离地"留出集上, 门控版法向力**结构性为 0**, 无门控版外推凭空给力
  R3c 接管比 / 前向误差
"""
from __future__ import annotations
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import srbm_dynamics as srb
from srbm_residual import (NOMINAL, REAL, sample_XU, accel_from_XU, jac_autograd,
                           grad_metrics, fwd_rms, Residual as AccelResidual, train_residual)

C_NOM, C_ACC, C_CON, C_UNG, C_TEA = "#8C8C8C", "#4C72B0", "#55A868", "#C44E52", "#333333"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
G = srb.G
OUT_SCALE = 50.0  # 残差输出 -> 力单位(N)


class ContactResidual(nn.Module):
    """每足共享 MLP: 12 维特征 -> (ΔF_n, ΔF_t)。零初始化 -> 起点零残差。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return OUT_SCALE * self.net(feat)


def contact_accel(XU, p, residual=None, gated=True, return_diag=False):
    """student 加速度: 每足平滑接触力 + (可选)门控/非负接触力残差。"""
    px, pz, th = XU[:, 0], XU[:, 1], XU[:, 2]
    vx, vz, om = XU[:, 3], XU[:, 4], XU[:, 5]
    base_x = [p.half_len, -p.half_len]
    Fx = torch.zeros_like(px); Fz = torch.zeros_like(px); tau = torch.zeros_like(px)
    Fn_air_terms = []  # 诊断: 各足法向力(用于离地检验)
    for k, bx in enumerate(base_x):
        ext_k = XU[:, 6 + k]
        fx, fz = srb.foot_world(px, pz, th, ext_k, bx, p.leg_nominal)
        rx, rz = fx - px, fz - pz
        vfx = vx - om * rz
        vfz = vz + om * rx
        # 平滑物理接触力
        pen = p.eps * F.softplus(-fz / p.eps)
        gate = torch.sigmoid(-fz / p.eps)                       # contact_weight
        Fn_phys = torch.clamp(p.k_n * pen - p.k_d * vfz * gate, min=0.0)
        Ft_phys = -p.mu * Fn_phys * torch.tanh(vfx / p.v_eps)
        if residual is not None:
            feat = torch.stack([pz, th, vx, vz, om, ext_k,
                                fz, gate, vfz, vfx, Fn_phys, Ft_phys], dim=-1)
            d = residual(feat)
            dFn, dFt = d[:, 0], d[:, 1]
            if gated:
                dFn = dFn * gate                                # 门控: 离地不给力
                dFt = dFt * gate
            Fn = torch.clamp(Fn_phys + dFn, min=0.0)            # 法向非负
            Ft = Ft_phys + dFt
        else:
            Fn, Ft = Fn_phys, Ft_phys
        Fx = Fx + Ft; Fz = Fz + Fn
        tau = tau + (rx * Fn - rz * Ft)
        Fn_air_terms.append(Fn)
    A = torch.stack([Fx / p.m, Fz / p.m - G, tau / p.I], dim=-1)
    if return_diag:
        return A, torch.stack(Fn_air_terms, dim=-1)   # (N,3),(N,2)
    return A


def jac_contact(XU, p, residual, gated=True):
    XU = XU.clone().requires_grad_(True)
    A = contact_accel(XU, p, residual, gated)
    Js = []
    for j in range(3):
        g, = torch.autograd.grad(A[:, j].sum(), XU, retain_graph=(j < 2))
        Js.append(g)
    return torch.stack(Js, dim=1).detach()


def sample_airborne(N, dev):
    """明确离地状态集: 高机身 + 负腿伸长 -> 足端在地面之上(gap>0)。"""
    X = torch.stack([
        torch.zeros(N, device=dev),
        0.36 + 0.04 * torch.rand(N, device=dev),       # pz 高
        0.2 * (torch.rand(N, device=dev) - 0.5),
        -0.1 + 0.6 * torch.rand(N, device=dev),
        0.6 * (torch.rand(N, device=dev) - 0.5),
        2.0 * (torch.rand(N, device=dev) - 0.5),
    ], dim=-1)
    U = -0.06 + 0.03 * torch.rand(N, 2, device=dev)    # ext 负 -> 缩腿
    return torch.cat([X, U], dim=-1)


def train_contact_residual(n_iters, dev, scale, gated=True, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    r = ContactResidual().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    for it in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = accel_from_XU(XU, REAL, "smooth")
        A_S = contact_accel(XU, NOMINAL, residual=r, gated=gated)
        loss = (((A_S - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return r


def airborne_force(residual, gated, XU_air):
    with torch.no_grad():
        _, Fn = contact_accel(XU_air, NOMINAL, residual=residual, gated=gated, return_diag=True)
    return float(Fn.abs().mean())   # 离地时本应为 0


def takeover_contact(XU, residual, gated):
    with torch.no_grad():
        A_N = contact_accel(XU, NOMINAL, residual=None)
        A_S = contact_accel(XU, NOMINAL, residual=residual, gated=gated)
    return float((A_S - A_N).norm(dim=-1).mean() / (A_N.norm(dim=-1).mean() + 1e-9))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 250 if args.quick else 1500
    print(f"device={dev} n_iters={n_iters}")

    eval_XU = sample_XU(512, dev)
    XU_air = sample_airborne(512, dev)
    J_teacher = jac_autograd(eval_XU, REAL, "smooth")
    J_nom = jac_autograd(eval_XU, NOMINAL, "smooth")
    cos_nom, _, _ = grad_metrics(J_nom, J_teacher)
    fwd_nom = fwd_rms(eval_XU, NOMINAL, REAL, "smooth")

    big = sample_XU(4096, dev)
    with torch.no_grad():
        scale = (accel_from_XU(big, REAL, "smooth")
                 - accel_from_XU(big, NOMINAL, "smooth")).std(dim=0).clamp_min(1e-6)

    # 加速度残差(上一支线, 作对照)
    rAcc, _ = train_residual("smooth", n_iters, dev, eval_XU, J_teacher, scale, seed=0)
    Jacc = jac_autograd(eval_XU, NOMINAL, "smooth", residual=rAcc)
    cos_acc, _, _ = grad_metrics(Jacc, J_teacher)

    # 接触力残差: 门控 / 无门控
    rConG = train_contact_residual(n_iters, dev, scale, gated=True, seed=0)
    rConU = train_contact_residual(n_iters, dev, scale, gated=False, seed=0)
    cos_cg, _, _ = grad_metrics(jac_contact(eval_XU, NOMINAL, rConG, True), J_teacher)
    cos_cu, _, _ = grad_metrics(jac_contact(eval_XU, NOMINAL, rConU, False), J_teacher)

    fwd_cg = float(((contact_accel(eval_XU, NOMINAL, rConG, True)
                     - accel_from_XU(eval_XU, REAL, "smooth")) ** 2).mean().sqrt())
    air_phys = airborne_force(None, True, XU_air)          # 物理(应为0)
    air_gated = airborne_force(rConG, True, XU_air)
    air_ungated = airborne_force(rConU, False, XU_air)
    tk_cg = takeover_contact(eval_XU, rConG, True)

    summary = {
        "nominal": {"fwd_rms": fwd_nom, "grad_cos": cos_nom},
        "accel_residual": {"grad_cos": cos_acc},
        "contact_residual_gated": {"fwd_rms": fwd_cg, "grad_cos": cos_cg, "takeover": tk_cg},
        "contact_residual_ungated": {"grad_cos": cos_cu},
        "airborne_Fn_physics": air_phys,
        "airborne_Fn_gated": air_gated,
        "airborne_Fn_ungated": air_ungated,
    }
    json.dump(summary, open(os.path.join(HERE, "results_phase3b.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---------- 图 R3 ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = ["nominal", "accel\nresidual", "contact res.\n(gated)", "contact res.\n(ungated)"]
    cosv = [cos_nom, cos_acc, cos_cg, cos_cu]
    ax[0].bar(range(4), cosv, color=[C_NOM, C_ACC, C_CON, C_UNG])
    ax[0].set_ylim(0, 1.02); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].axhline(1.0, color="k", lw=.6, ls=":"); ax[0].set_ylabel("gradient cosine to TRUE J")
    ax[0].set_title("(a) gradient fidelity\n(contact-force residual is physically aligned)")

    al = ["physics\n(no residual)", "contact res.\n(GATED)", "contact res.\n(UNGATED)"]
    av = [air_phys, air_gated, air_ungated]
    bars = ax[1].bar(range(3), av, color=[C_NOM, C_CON, C_UNG])
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(al, fontsize=8)
    ax[1].set_ylabel("mean |F_n| at AIRBORNE states [N]")
    ax[1].set_title("(b) gating ablation: spurious ground force\nwhen foot is off the ground")
    for b, v in zip(bars, av):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # (c) 残差力 vs gap 散点(门控 vs 无门控), 展示门控把离地力压到 0
    gaps = XU_air[:, 1] - (NOMINAL.leg_nominal + XU_air[:, 6])  # 近似 foot 高度(θ≈0)
    with torch.no_grad():
        _, FnG = contact_accel(XU_air, NOMINAL, rConG, True, return_diag=True)
        _, FnU = contact_accel(XU_air, NOMINAL, rConU, False, return_diag=True)
    ax[2].scatter(gaps.cpu(), FnU[:, 0].cpu(), s=6, color=C_UNG, alpha=.5, label="ungated")
    ax[2].scatter(gaps.cpu(), FnG[:, 0].cpu(), s=6, color=C_CON, alpha=.5, label="gated")
    ax[2].axhline(0, color="k", lw=.6, ls=":")
    ax[2].set_xlabel("foot gap above ground [m] (>0: airborne)")
    ax[2].set_ylabel("residual model F_n [N]")
    ax[2].set_title("(c) gated residual → 0 force when airborne"); ax[2].legend(fontsize=8)
    fig.suptitle("R3  Contact-force residual: physically-aligned + contact gating prevents "
                 "phantom ground force", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R3_contact_residual.png"), bbox_inches="tight")
    plt.close(fig)
    print("[R3] figure saved.")


if __name__ == "__main__":
    main()
