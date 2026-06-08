"""
运动学(落足/接触点)残差 [R9]: 修正**接触点位置** r_x, 治 R8 揭示的几何符号翻转
==========================================================================
R8 结论: 力臂符号翻转是**运动学**失配; 只能缩放力的 C(约束接触力残差)对它**结构失明**
(τ=Σ r_x,nominal·F_n, 标称力臂被焊死)。R9 验证 R8 开出的"药方": 让残差**修正接触点位置**
而非力 —— 输出每足学习到的 x 偏置 Δx_k(state), 用**同一套平滑接触模型**在**修正后的落足点**
算力与力矩, 梯度 ∂α/∂ext 经**修正后的力臂**流动 -> 应能跟上符号翻转。

    D(运动学残差): foot_x = nominal_foot_x + Δx_k(state);  τ = Σ (r_x+Δx_k)·F_n
        (复用 srbm_accel 的 foot_dx 通道, Δx 由小 MLP 给出, 零初始化=起点标称)

对照 R8 的非对称剪刀 teacher(fd=[−δx,+δx], δx>half_len 翻转差动控制方向):
  A nominal / C R4约束接触力残差(力, 失明) / D 运动学残差(几何, 应recover)。
核心检验: D 的差动控制增益 g_diff 是否随 teacher 在 δx=0.25 穿零反号? 姿态梯度余弦是否
在 |δx|>0.25 仍保真(C 已塌到 0.31)? + 闭环: D 的策略能否部署到翻转后的 teacher 站住?
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import srbm_dynamics as srb
from srbm_residual import NOMINAL, sample_XU
from srbm_residual_constrained import contact_accel_c, ContactResidual
from srbm_residual_struct import jac_of, cos_sign
from srbm_residual_extreme import accel_asym, nominal_accel, scale_for, diff_gain, train_resid as train_force
from srbm_residual_closedloop import train_policy, evaluate

C_NOM, C_C, C_D, C_TEA = "#8C8C8C", "#55A868", "#DD8452", "#4C72B0"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
HALF = NOMINAL.half_len
OFFSET_SCALE = 0.6   # 每足可学习 x 偏置上限 ±0.6 (覆盖 δx 至 0.45)


class KinematicResidual(nn.Module):
    """body state(5) -> 每足接触点 x 偏置 Δx_k。零初始化 -> 起点=标称落足。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, XU):
        feat = XU[:, 1:6]                       # pz, θ, vx, vz, ω (落足是几何/运动学量, 不取 ext)
        return OFFSET_SCALE * torch.tanh(self.net(feat))


def kinematic_accel(XU, p, residual):
    """运动学残差 student: 在**修正后的落足点**跑同一套平滑接触模型。"""
    state = (XU[:, 0], XU[:, 1], XU[:, 2], XU[:, 3], XU[:, 4], XU[:, 5])
    ext = [XU[:, 6], XU[:, 7]]
    dfoot = residual(XU)                         # (N,2) 每足 Δx
    ax, az, al, _ = srb.srbm_accel(state, ext, "smooth", p, [p.half_len, -p.half_len],
                                   foot_dx=[dfoot[:, 0], dfoot[:, 1]])
    return torch.stack([ax, az, al], dim=-1)


def train_kinematic(n_iters, dev, scale, dx, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    r = KinematicResidual().to(dev).to(torch.get_default_dtype())
    teacher = lambda XU: accel_asym(XU, NOMINAL, dx)
    student = lambda XU: kinematic_accel(XU, NOMINAL, r)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = teacher(XU)
        loss = (((student(XU) - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return student


def part1_sweep(dev, n_iters, n_pts):
    eval_XU = sample_XU(512, dev)
    deltas = np.linspace(-0.45, 0.45, n_pts)
    gain = {k: [] for k in ["teacher", "A", "C", "D"]}
    cos = {k: [] for k in ["A", "C", "D"]}
    fwd = {k: [] for k in ["A", "C", "D"]}
    for dx in deltas:
        teacher_fn = lambda XU, d=dx: accel_asym(XU, NOMINAL, float(d))
        J_T = jac_of(teacher_fn, eval_XU)
        sc = scale_for(float(dx), dev)
        stC = train_force("C", n_iters, dev, sc, float(dx), seed=1)
        stD = train_kinematic(n_iters, dev, sc, float(dx), seed=1)
        gain["teacher"].append(diff_gain(J_T))
        with torch.no_grad():
            AT = teacher_fn(eval_XU)
        for key, fn in [("A", nominal_accel), ("C", stC), ("D", stD)]:
            J = jac_of(fn, eval_XU)
            gain[key].append(diff_gain(J))
            cos[key].append(cos_sign(J, J_T, rows=[2], cols=[6, 7])[0])
            with torch.no_grad():
                fwd[key].append(float(((fn(eval_XU) - AT) ** 2).mean().sqrt()))
        print(f"  δx={dx:+.3f}  g_diff T={gain['teacher'][-1]:+.0f} C={gain['C'][-1]:+.0f} "
              f"D={gain['D'][-1]:+.0f} | att-cos C={cos['C'][-1]:.2f} D={cos['D'][-1]:.2f} "
              f"| fwd C={fwd['C'][-1]:.2f} D={fwd['D'][-1]:.2f}", flush=True)
    return deltas.tolist(), gain, cos, fwd


def part2_closed_loop(dev, dxs, res_iters, pol_iters):
    rows = {k: [] for k in ["nominal", "force-res", "kin-res", "teacher"]}
    for dx in dxs:
        teacher_fn = lambda XU, d=dx: accel_asym(XU, NOMINAL, float(d))
        sc = scale_for(float(dx), dev)
        stC = train_force("C", res_iters, dev, sc, float(dx), seed=0)
        stD = train_kinematic(res_iters, dev, sc, float(dx), seed=0)
        models = {"nominal": nominal_accel, "force-res": stC, "kin-res": stD, "teacher": teacher_fn}
        for name, fn in models.items():
            pol, _ = train_policy(fn, pol_iters, dev, seed=0)
            dep, _ = evaluate(pol, teacher_fn, dev)
            rows[name].append(dep)
        print(f"  δx={dx:+.2f}  deploy@teacher  " +
              " ".join(f"{k}={rows[k][-1]:.4f}" for k in rows), flush=True)
    return rows


def _plot(deltas, gain, cos, fwd, dxs, cl):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    flip = HALF
    # (a) 差动控制增益 g_diff: D 是否跟上 teacher 的符号翻转?
    gmap = [("teacher", C_TEA, "teacher (true)"), ("A", C_NOM, "A nominal"),
            ("C", C_C, "C force-res (R4)"), ("D", C_D, "D kinematic-res (new)")]
    for k, c, lab in gmap:
        lw = 2.6 if k == "teacher" else 1.8
        ax[0].plot(deltas, gain[k], color=c, lw=lw, marker="o", ms=3, label=lab)
    for xl in (-flip, flip):
        ax[0].axvline(xl, color="k", lw=1, ls="--", alpha=.6)
    ax[0].axhline(0, color="r", lw=.9, ls=":")
    ax[0].set_xlabel("scissor offset δx [m]")
    ax[0].set_ylabel(r"diff. control gain $\partial\alpha/\partial e_{\rm diff}$")
    ax[0].set_title("(a) kinematic residual D tracks the sign flip;\nforce residual C cannot")
    ax[0].legend(fontsize=7)
    # (b) 姿态梯度余弦保真
    for k, c, lab in [("A", C_NOM, "A nominal"), ("C", C_C, "C force-res"), ("D", C_D, "D kinematic-res")]:
        ax[1].plot(deltas, cos[k], color=c, lw=2, marker="s", ms=3, label=lab)
    for xl in (-flip, flip):
        ax[1].axvline(xl, color="k", lw=1, ls="--", alpha=.6)
    ax[1].set_xlabel("scissor offset δx [m]"); ax[1].set_ylabel("attitude ∂α/∂ext cosine to teacher")
    ax[1].set_title("(b) attitude-gradient fidelity\n(D stays high past flip; C collapses)")
    ax[1].legend(fontsize=7)
    # (c) 闭环部署 loss vs δx
    cmap = {"nominal": C_NOM, "force-res": C_C, "kin-res": C_D, "teacher": C_TEA}
    for name, c in cmap.items():
        ax[2].plot(dxs, cl[name], color=c, lw=2, marker="o", ms=5, label=f"π_{name}")
    ax[2].axvline(flip, color="k", lw=1, ls="--", alpha=.6)
    ax[2].set_yscale("log"); ax[2].set_xlabel("scissor offset δx [m]")
    ax[2].set_ylabel("balance loss deployed on teacher (log)")
    ax[2].set_title("(c) closed-loop: kinematic residual recovers\npost-flip transfer (support-restored δx)")
    ax[2].legend(fontsize=7)
    fig.suptitle("R9  Kinematic (foot-placement) residual fixes the geometric sign flip that "
                 "force/accel residuals structurally cannot", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R9_kinematic_residual.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 250 if args.quick else 800
    n_pts = 9 if args.quick else 19
    res_iters = 250 if args.quick else 800
    pol_iters = 60 if args.quick else 200
    # 闭环只取支撑基**已恢复**的翻转后 δx(避开 δx=0.25 奇点)
    dxs = [0.0, 0.40] if args.quick else [0.0, 0.35, 0.40, 0.45]
    print(f"device={dev} OFFSET_SCALE={OFFSET_SCALE} sweep_pts={n_pts} "
          f"res_iters={res_iters} pol_iters={pol_iters}", flush=True)

    print("== Part 1: gradient sweep (does kinematic residual recover the flip?) ==", flush=True)
    deltas, gain, cos, fwd = part1_sweep(dev, n_iters, n_pts)
    print("== Part 2: closed-loop deployment (support-restored post-flip δx) ==", flush=True)
    cl = part2_closed_loop(dev, dxs, res_iters, pol_iters)

    res = {"half_len": HALF, "deltas": deltas, "diff_gain": gain, "grad_cos": cos, "fwd_rms": fwd,
           "closed_loop_dxs": dxs, "closed_loop": cl}
    json.dump(res, open(os.path.join(HERE, "results_phase3h.json"), "w"), indent=2, ensure_ascii=False)
    _plot(deltas, gain, cos, fwd, dxs, cl)
    print("[R9] figure saved.", flush=True)


if __name__ == "__main__":
    main()
