"""
神经残差动力学: 前向精度 vs 梯度保真 (方案 B 参数失配 teacher)
==========================================================================
研究问题(用户核心关切): 前向回归训练出的残差 r_phi, 其**梯度**是否可信?
即 "前向准 != 梯度准", 以及 "残差是否偷偷接管动力学(takeover)"。

设计(利用方案 B 的关键性质):
  teacher = SRBM(真实参数 m,I,k,d,μ 失配)  —— 本身可微, 故"真梯度" J_T=∂a_T/∂(x,u) 解析可得!
  student = SRBM(标称参数) + 加速度残差 r_phi(x,u)
  用前向回归训练: r_phi ≈ a_teacher - a_nominal
评估:
  R1  前向误差 vs 梯度误差 随训练 (预期: 前向快降, 梯度滞后)
  R1  J_student vs J_teacher 余弦/符号一致 (对照未校正 J_nominal)
  R1  接管比 ‖r_phi‖/‖a_nominal‖
  R2  rollout 漂移 (nominal vs student vs teacher)
  R3  危险对照: 硬接触 teacher -> 残差给"自信但错误"的平滑梯度(与有限差分真值符号不一致)

形式选择: 本阶段用**加速度残差**(最干净, 隔离梯度保真问题)。
接触力残差(最贴近四足误差源)/状态残差留作下一步。
"""
from __future__ import annotations
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import srbm_dynamics as srb
from srbm_dynamics import SRBMParams

C_NOM, C_STU, C_TEA, C_HARD, C_PURPLE = "#8C8C8C", "#55A868", "#4C72B0", "#C44E52", "#8172B3"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

NOMINAL = SRBMParams()                                              # 标称模型
REAL = SRBMParams(m=12.0, I=0.45, k_n=9000.0, k_d=45.0, mu=0.65)    # 真实模型(参数失配)


def accel_from_XU(XU, p, model="smooth", residual=None):
    """(N,8)=[px,pz,θ,vx,vz,ω, ext0,ext1] -> (N,3)=[ax,az,α]。可选叠加加速度残差。"""
    X, U = XU[:, :6], XU[:, 6:]
    state = (X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5])
    ext = [U[:, 0], U[:, 1]]
    ax, az, al, _ = srb.srbm_accel(state, ext, model, p, [p.half_len, -p.half_len])
    A = torch.stack([ax, az, al], dim=-1)
    if residual is not None:
        A = A + residual(XU)
    return A


def sample_XU(N, dev):
    X = torch.stack([
        torch.zeros(N, device=dev),                    # px (加速度与 px 无关)
        0.300 + 0.045 * torch.rand(N, device=dev),     # pz ~U(0.30,0.345) 轻接触
        0.24 * (torch.rand(N, device=dev) - 0.5),      # θ  ~U(-0.12,0.12)
        -0.1 + 0.6 * torch.rand(N, device=dev),        # vx ~U(-0.1,0.5)
        0.6 * (torch.rand(N, device=dev) - 0.5),       # vz ~U(-0.3,0.3)
        2.0 * (torch.rand(N, device=dev) - 0.5),       # ω  ~U(-1,1)
    ], dim=-1)
    U = -0.01 + 0.03 * torch.rand(N, 2, device=dev)    # ext ~U(-0.01,0.02)
    return torch.cat([X, U], dim=-1)


class Residual(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # 起点=零残差

    def forward(self, xu):
        return self.net(xu)


def jac_autograd(XU, p, model, residual=None):
    """解析 Jacobian ∂a/∂(x,u): (N,3,8)。3 次反传(每个输出维一次)。"""
    XU = XU.clone().requires_grad_(True)
    A = accel_from_XU(XU, p, model, residual)
    Js = []
    for j in range(3):
        g, = torch.autograd.grad(A[:, j].sum(), XU, retain_graph=(j < 2))
        Js.append(g)
    return torch.stack(Js, dim=1).detach()


def jac_fd(XU, p, model, h=1e-4):
    """有限差分 Jacobian(硬接触 teacher 的"真梯度"参照, 因其解析梯度有误导)。"""
    N = XU.shape[0]
    J = torch.zeros(N, 3, 8, device=XU.device, dtype=XU.dtype)
    for i in range(8):
        dp = torch.zeros_like(XU); dp[:, i] = h
        Ap = accel_from_XU(XU + dp, p, model)
        Am = accel_from_XU(XU - dp, p, model)
        J[:, :, i] = (Ap - Am) / (2 * h)
    return J.detach()


def grad_metrics(J_s, J_t):
    """J_s,J_t: (N,3,8)。返回 余弦相似度(均值), 相对Frobenius误差, 符号不一致率。"""
    a = J_s.reshape(J_s.shape[0], -1)
    b = J_t.reshape(J_t.shape[0], -1)
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)
    rel = (J_s - J_t).reshape(-1).norm() / (J_t.reshape(-1).norm() + 1e-12)
    mask = J_t.abs() > (0.05 * J_t.abs().mean())
    sign_dis = ((torch.sign(J_s) != torch.sign(J_t)) & mask).sum().float() / (mask.sum() + 1e-9)
    return float(cos.mean()), float(rel), float(sign_dis)


def fwd_rms(XU, p_nom, p_real, teacher_model, residual=None):
    with torch.no_grad():
        A_T = accel_from_XU(XU, p_real, teacher_model)
        A_S = accel_from_XU(XU, p_nom, "smooth", residual)
    return float(((A_S - A_T) ** 2).mean().sqrt())


def takeover_ratio(XU, p_nom, residual):
    with torch.no_grad():
        A_N = accel_from_XU(XU, p_nom, "smooth")
        r = residual(XU)
    return float(r.norm(dim=-1).mean() / (A_N.norm(dim=-1).mean() + 1e-9))


def train_residual(teacher_model, n_iters, dev, eval_XU, J_teacher, scale, seed=0,
                   weight_decay=0.0, eval_every=50):
    torch.manual_seed(seed)
    r = Residual().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(r.parameters(), lr=2e-3, weight_decay=weight_decay)
    hist = {"iter": [], "fwd": [], "gcos": [], "gsign": [], "takeover": []}
    for it in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = accel_from_XU(XU, REAL, teacher_model)
        A_S = accel_from_XU(XU, NOMINAL, "smooth", residual=r)
        loss = (((A_S - A_T) / scale) ** 2).mean()   # 按维度归一化(否则被 stiff 法向 az 主导)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % eval_every == 0 or it == n_iters - 1:
            J_s = jac_autograd(eval_XU, NOMINAL, "smooth", residual=r)
            cos, rel, sgn = grad_metrics(J_s, J_teacher)
            hist["iter"].append(it)
            hist["fwd"].append(fwd_rms(eval_XU, NOMINAL, REAL, teacher_model, r))
            hist["gcos"].append(cos); hist["gsign"].append(sgn)
            hist["takeover"].append(takeover_ratio(eval_XU, NOMINAL, r))
    return r, hist


def rollout_accel(state0, U_seq, p, model, residual, dt, n):
    s = list(state0)
    traj = [list(s)]
    for t in range(n):
        XU = torch.tensor([[s[0], s[1], s[2], s[3], s[4], s[5],
                            float(U_seq[t, 0]), float(U_seq[t, 1])]],
                           device=U_seq.device, dtype=U_seq.dtype)
        with torch.no_grad():
            A = accel_from_XU(XU, p, model, residual)[0]
        s[3] += dt * float(A[0]); s[4] += dt * float(A[1]); s[5] += dt * float(A[2])
        s[0] += dt * s[3]; s[1] += dt * s[4]; s[2] += dt * s[5]
        traj.append(list(s))
    return np.array(traj)


def bptt_grad_vs_horizon(p, model, residual, horizons, dt, dev, ext_val=0.012, v0_val=0.2):
    """可微 rollout: |∂pz_T/∂v0| 随视野。teacher/nominal 应有界, 不稳的残差会发散。"""
    out = []
    z = lambda: torch.zeros((), device=dev)
    for n in horizons:
        v0 = torch.tensor(v0_val, device=dev, requires_grad=True)
        px, pz, th = z(), torch.tensor(0.315, device=dev), z()
        vx, vz, om = v0, z(), z()
        e = torch.tensor(ext_val, device=dev)
        for _ in range(n):
            XU = torch.stack([px, pz, th, vx, vz, om, e, e]).unsqueeze(0)
            A = accel_from_XU(XU, p, model, residual)[0]
            vx = vx + dt * A[0]; vz = vz + dt * A[1]; om = om + dt * A[2]
            px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
        g, = torch.autograd.grad(pz, v0, retain_graph=False)
        val = abs(float(g))
        out.append(val if math.isfinite(val) else 1e30)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 200 if args.quick else 1500
    print(f"device={dev} n_iters={n_iters}")

    eval_XU = sample_XU(512, dev)
    # 真梯度参照: 平滑 teacher 用解析; 硬接触 teacher 用有限差分(其解析梯度有误导)
    J_teacher_smooth = jac_autograd(eval_XU, REAL, "smooth")
    J_teacher_hard = jac_fd(eval_XU, REAL, "soft")

    # 按维度归一化尺度(残差目标 a_T-a_N 的逐维标准差) —— 防止 stiff 法向 az 主导损失
    big = sample_XU(4096, dev)
    with torch.no_grad():
        scale = (accel_from_XU(big, REAL, "smooth")
                 - accel_from_XU(big, NOMINAL, "smooth")).std(dim=0).clamp_min(1e-6)
    print(f"[per-dim residual-target std] ax,az,al = {scale.tolist()}")

    # 基线: 未校正 nominal 的前向误差与梯度保真
    J_nom = jac_autograd(eval_XU, NOMINAL, "smooth")
    nom_fwd = fwd_rms(eval_XU, NOMINAL, REAL, "smooth")
    nom_cos, nom_rel, nom_sgn = grad_metrics(J_nom, J_teacher_smooth)
    print(f"[nominal vs smooth-teacher] fwd_rms={nom_fwd:.4f} gcos={nom_cos:.4f} gsign={nom_sgn:.3f}")

    # R1: 平滑 teacher 残差
    rS, histS = train_residual("smooth", n_iters, dev, eval_XU, J_teacher_smooth, scale, seed=0)
    # R3 危险: 硬接触 teacher 残差 (真梯度用有限差分)
    rH, histH = train_residual("soft", n_iters, dev, eval_XU, J_teacher_hard, scale, seed=0)

    # 最终指标
    J_sS = jac_autograd(eval_XU, NOMINAL, "smooth", residual=rS)
    cosS, relS, sgnS = grad_metrics(J_sS, J_teacher_smooth)
    J_sH = jac_autograd(eval_XU, NOMINAL, "smooth", residual=rH)
    cosH, relH, sgnH = grad_metrics(J_sH, J_teacher_hard)
    fwdS = histS["fwd"][-1]; fwdH = histH["fwd"][-1]

    # ---------- 图 R1: 前向 vs 梯度误差 + 余弦 bars + 接管比 ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    it = histS["iter"]
    ax[0].plot(it, histS["fwd"], color=C_STU, lw=2, label="forward RMS error")
    ax[0].plot(it, [1 - c for c in histS["gcos"]], color=C_HARD, lw=2, label="gradient error (1−cosine)")
    ax[0].axhline(nom_fwd, color=C_STU, ls=":", lw=1, alpha=.7)
    ax[0].axhline(1 - nom_cos, color=C_HARD, ls=":", lw=1, alpha=.7)
    ax[0].set_yscale("log"); ax[0].set_xlabel("training iteration"); ax[0].set_ylabel("error (log)")
    ax[0].set_title("(a) smooth teacher: residual improves\nboth forward & gradient (dotted=nominal)")
    ax[0].legend(fontsize=8)

    labels = ["nominal", "residual\n(smooth teacher)", "residual\n(hard teacher)"]
    cosvals = [nom_cos, cosS, cosH]
    ax[1].bar(range(3), cosvals, color=[C_NOM, C_STU, C_HARD])
    ax[1].set_ylim(0, 1.05); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("gradient cosine sim to TRUE J"); ax[1].axhline(1.0, color="k", lw=.6, ls=":")
    ax[1].set_title("(b) gradient fidelity vs true Jacobian")

    ax[2].plot(it, histS["takeover"], color=C_STU, lw=2, label="smooth teacher")
    ax[2].plot(it, histH["takeover"], color=C_HARD, lw=2, label="hard teacher")
    ax[2].set_xlabel("training iteration"); ax[2].set_ylabel(r"takeover  $\|r_\phi\|/\|a_{\rm nom}\|$")
    ax[2].set_title("(c) residual magnitude (takeover risk)"); ax[2].legend(fontsize=8)
    fig.suptitle("R1  Neural residual dynamics: forward accuracy ≠ gradient fidelity "
                 "(param-mismatch teacher)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R1_residual_gradfidelity.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 R2: rollout 漂移 + BPTT 梯度随视野(梯度保真经 rollout 的严格检验) ----------
    dt, n = 2e-3, 1200
    s0 = (0.0, 0.315, 0.0, 0.2, 0.0, 0.0)
    U_seq = torch.full((n, 2), 0.012, device=dev)  # 固定轻压腿
    tr_tea = rollout_accel(s0, U_seq, REAL, "smooth", None, dt, n)
    tr_nom = rollout_accel(s0, U_seq, NOMINAL, "smooth", None, dt, n)
    tr_stu = rollout_accel(s0, U_seq, NOMINAL, "smooth", rS, dt, n)
    tr_stuH = rollout_accel(s0, U_seq, NOMINAL, "smooth", rH, dt, n)
    tt = np.arange(n + 1) * dt

    def drift(tr):  # [pz,θ,vx] 与 teacher 的偏差范数, 截断溢出便于作图
        d = np.linalg.norm(tr[:, [1, 2, 3]] - tr_tea[:, [1, 2, 3]], axis=1)
        return np.nan_to_num(np.clip(d, 0, 1e6), nan=1e6)

    horizons = [int(h) for h in np.linspace(100, 1200, 18)]
    grad_tea = bptt_grad_vs_horizon(REAL, "smooth", None, horizons, dt, dev)
    grad_res = bptt_grad_vs_horizon(NOMINAL, "smooth", rS, horizons, dt, dev)
    grad_resH = bptt_grad_vs_horizon(NOMINAL, "smooth", rH, horizons, dt, dev)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(tt, drift(tr_nom), color=C_NOM, lw=1.8, label="nominal (no residual)")
    ax[0].plot(tt, drift(tr_stu), color=C_STU, lw=1.8, label="residual (smooth teacher)")
    ax[0].plot(tt, drift(tr_stuH), color=C_HARD, lw=1.8, label="residual (hard teacher)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("‖state − teacher‖ ([pz,θ,vx], log)")
    ax[0].set_title("(a) rollout drift vs real teacher")
    ax[0].legend(fontsize=8)
    ax[1].plot(horizons, grad_tea, color=C_TEA, lw=2, marker="o", ms=3, label="teacher (real, TRUE grad)")
    ax[1].plot(horizons, grad_res, color=C_STU, lw=2, marker="^", ms=3, label="residual (smooth teacher)")
    ax[1].plot(horizons, grad_resH, color=C_HARD, lw=2, marker="s", ms=3, label="residual (hard teacher)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("rollout horizon (#steps)")
    ax[1].set_ylabel(r"$|\partial p_{z,T}/\partial v_0|$ (BPTT grad, log)")
    ax[1].set_title("(b) BPTT gradient through rollout:\nsmooth-teacher residual tracks TRUE; hard-teacher deviates")
    ax[1].legend(fontsize=7)
    fig.suptitle("R2  Through rollout: smooth-teacher residual stays gradient-faithful; "
                 "hard-teacher residual does not", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R2_residual_rollout.png"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "nominal_fwd_rms": nom_fwd, "nominal_grad_cos": nom_cos, "nominal_grad_sign_disagree": nom_sgn,
        "smooth_teacher": {"fwd_rms_final": fwdS, "grad_cos": cosS, "grad_rel_err": relS, "grad_sign_disagree": sgnS},
        "hard_teacher": {"fwd_rms_final": fwdH, "grad_cos": cosH, "grad_rel_err": relH, "grad_sign_disagree": sgnH},
        "takeover_final_smooth": histS["takeover"][-1], "takeover_final_hard": histH["takeover"][-1],
        "rollout_drift_final_nominal": float(drift(tr_nom)[-1]),
        "rollout_drift_final_residual": float(drift(tr_stu)[-1]),
    }
    path = os.path.join(HERE, "results_phase3.json")
    json.dump(summary, open(path, "w"), indent=2, ensure_ascii=False)
    print("[R-summary]", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
