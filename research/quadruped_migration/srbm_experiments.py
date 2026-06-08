"""
SRBM 第二阶段实验 F1(摩擦/切向接触) + F2(姿态/可微平坦性丢失)
=================================================================
产出 figures/F1_friction.png, figures/F2_attitude.png 与 results_phase2.json 部分字段。
F3(闭环可微训练) 在 srbm_train.py。
"""
from __future__ import annotations
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import srbm_dynamics as srb
from srbm_dynamics import SRBMParams, rollout_srbm

C_BASE, C_CONTACT, C_SMOOTH, C_HARD = "#4C72B0", "#C44E52", "#55A868", "#8C8C8C"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
res = {}


def _np(t):
    return t.detach().cpu().numpy()


# =====================================================================
# F1 摩擦/切向接触：平滑库仑 vs 硬库仑，及摩擦驱动前向推进
# =====================================================================
def exp_friction(dev):
    p = SRBMParams()
    fn = 50.0  # 固定法向力
    v = torch.linspace(-0.6, 0.6, 1201, device=dev)

    # 三种摩擦律 (归一化 f_t/f_n)
    f_smooth = -p.mu * torch.tanh(v / p.v_eps)
    f_hard = -p.mu * torch.sign(v)
    kt = p.mu / p.v_eps  # 线性段斜率与 smooth 在 0 处一致
    f_soft = torch.clamp(-kt * v, min=-p.mu, max=p.mu)

    # 解析梯度 d(f_t/f_n)/dv
    g_smooth = -p.mu / p.v_eps * (1 - torch.tanh(v / p.v_eps) ** 2)
    g_hard = torch.zeros_like(v)  # sign 的导数几乎处处为 0

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].plot(_np(v), _np(f_hard), color=C_HARD, lw=2, label="hard  -μ·sign(v)")
    ax[0].plot(_np(v), _np(f_soft), color=C_BASE, lw=1.6, ls="--", label="soft  clamp(-kt·v)")
    ax[0].plot(_np(v), _np(f_smooth), color=C_SMOOTH, lw=2, label="smooth  -μ·tanh(v/vε)")
    ax[0].set_xlabel("tangential foot velocity v_t [m/s]")
    ax[0].set_ylabel(r"friction $f_t/f_n$")
    ax[0].set_title("(a) friction law")
    ax[0].legend(fontsize=8)
    ax[1].plot(_np(v), _np(g_hard), color=C_HARD, lw=2, label="hard  (∂≈0 a.e., +Dirac)")
    ax[1].plot(_np(v), _np(g_smooth), color=C_SMOOTH, lw=2, label="smooth  (bounded)")
    ax[1].set_xlabel("tangential foot velocity v_t [m/s]")
    ax[1].set_ylabel(r"$\partial (f_t/f_n)/\partial v_t$")
    ax[1].set_title("(b) tangential FoG (analog of E1)")
    ax[1].legend(fontsize=8)

    # 摩擦驱动前向推进：机身初始前倾 + 双腿周期蹬伸 -> 摩擦把竖直蹬地转成前向
    def hop_forward(mu):
        pp = SRBMParams(mu=mu)
        s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.34, device=dev),
              torch.tensor(0.12, device=dev),  # 初始前倾 0.12 rad
              torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev),
              torch.tensor(0.0, device=dev))
        n = 5000
        t = torch.arange(n, device=dev) * 1e-3
        push = 0.04 * torch.clamp(torch.sin(2 * math.pi * 2.0 * t), min=0.0)  # 2Hz 蹬伸
        seq = torch.stack([push, push], dim=1)
        return rollout_srbm(s0, seq, model="smooth", p=pp, dt=1e-3)

    tr_fric = hop_forward(0.9)
    tr_nofric = hop_forward(0.0)
    ax[2].plot(_np(tr_fric["px"]), color=C_SMOOTH, lw=2, label="μ=0.9 (friction)")
    ax[2].plot(_np(tr_nofric["px"]), color=C_HARD, lw=2, ls="--", label="μ=0.0 (no friction)")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("forward position px [m]")
    ax[2].set_title("(c) friction enables propulsion")
    ax[2].legend(fontsize=8)
    fig.suptitle("F1  Tangential contact (friction): smooth Coulomb gives informative FoG; "
                 "friction is what converts vertical push to forward motion", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F1_friction.png"), bbox_inches="tight")
    plt.close(fig)
    res["F1"] = {
        "smooth_fric_grad_max": float(g_smooth.abs().max()),
        "hard_fric_grad_max": float(g_hard.abs().max()),
        "px_with_friction": float(tr_fric["px"][-1]),
        "px_no_friction": float(tr_nofric["px"][-1]),
    }
    print("[F1] done.", {k: round(v, 4) for k, v in res["F1"].items()})


# =====================================================================
# F2 姿态：俯仰由接触力矩积分产生 -> 可微平坦性丢失的"长记忆"签名
# =====================================================================
def exp_attitude(dev):
    p = SRBMParams()
    dt = 1e-3
    n = 2500

    # (a) 初始俯仰扰动下的姿态二阶动力学(θ,ω 相图) —— 它是被积分出来的动态状态
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for th0, c in [(0.15, C_CONTACT), (-0.10, C_BASE), (0.30, "#DD8452")]:
        s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.36, device=dev),
              torch.tensor(th0, device=dev), torch.tensor(0.0, device=dev),
              torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev))
        seq = torch.zeros(n, 2, device=dev)
        tr = rollout_srbm(s0, seq, model="smooth", p=p, dt=dt)
        ax[0].plot(_np(tr["th"]), _np(tr["om"]), color=c, lw=1.4, label=f"θ₀={th0:+.2f}")
    ax[0].axhline(0, color="k", lw=.6, ls=":"); ax[0].axvline(0, color="k", lw=.6, ls=":")
    ax[0].set_xlabel("pitch θ [rad]"); ax[0].set_ylabel("pitch rate ω [rad/s]")
    ax[0].set_title("(a) attitude is a 2nd-order ODE state\nIθ̈=Σrᵢ×Fᵢ  (NOT algebraically flat)")
    ax[0].legend(fontsize=8)

    # (b) 可微平坦性丢失签名: 终端俯仰对"每一步腿伸长"的灵敏度 ∂θ_T/∂ext_t
    #     无人机: 姿态≈瞬时代数函数(短记忆); SRBM: 灵敏度沿时间铺开(积分/长记忆)
    e = torch.zeros(n, 2, device=dev, requires_grad=True)
    s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.36, device=dev),
          torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev),
          torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev))
    tr = rollout_srbm(s0, e, model="smooth", p=p, dt=dt)
    tr["th"][-1].backward()
    sens = e.grad.abs().sum(dim=1)  # 每步灵敏度 (对两腿求和)
    t_axis = np.arange(n) * dt
    ax[1].plot(t_axis, _np(sens), color=C_SMOOTH, lw=1.5)
    ax[1].fill_between(t_axis, 0, _np(sens), color=C_SMOOTH, alpha=0.25)
    ax[1].set_xlabel("time t of applied leg-extension [s]")
    ax[1].set_ylabel(r"$|\partial \theta_T/\partial\, \mathrm{ext}_t|$")
    ax[1].set_title("(b) terminal attitude depends on the WHOLE\ncontact-force history (integrator memory)")
    # 量化记忆长度: 灵敏度的"质量中心"时间
    sens_np = _np(sens); mass = sens_np.sum() + 1e-12
    t_centroid = float((t_axis * sens_np).sum() / mass)
    ax[1].axvline(t_centroid, color=C_CONTACT, ls="--", lw=1.2,
                  label=f"sensitivity centroid ≈ {t_centroid:.2f}s")
    ax[1].legend(fontsize=8)
    fig.suptitle("F2  Quadruped attitude is produced by contact torque (lost differential flatness)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F2_attitude.png"), bbox_inches="tight")
    plt.close(fig)
    res["F2"] = {
        "sensitivity_centroid_s": t_centroid,
        "horizon_s": n * dt,
        "note": "attitude sensitivity spread across history -> integrator, not flat algebraic map",
    }
    print("[F2] done.", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["F2"].items()})


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print("device =", dev)
    exp_friction(dev)
    exp_attitude(dev)
    with open(os.path.join(HERE, "results_phase2.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("F1/F2 done.")


if __name__ == "__main__":
    main()
