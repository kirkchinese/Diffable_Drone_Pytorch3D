"""
闭环验证 [R6]: 残差路线的"最终考验" —— 策略 sim-to-sim 迁移与钻漏洞
==========================================================================
前面 R1-R5 都在监督设定下验证残差**梯度保真**(必要条件)。R6 测**充分条件**:
用残差模型的 BPTT 梯度**训出一个策略**, 部署到**真实(teacher)系统**还 work 吗?

三个策略, 各经一种动力学训练(同一平衡任务: 从扰动恢复直立+目标高度):
  π_nominal  : 经标称 SRBM 训练
  π_residual : 经 标称+约束接触力残差C(已拟合teacher) 训练
  π_teacher  : 经 真实 SRBM 训练 (oracle 上界)
全部**部署到 teacher** 比平衡 loss。并测**利用漏洞 gap**:
  gap = deploy_loss(on teacher) − train_loss(on own model)
  gap 大 = 策略在自己模型上看着好、到真实系统就崩(钻了模型漏洞)。

teacher = 参数失配 REAL(平滑, C 在此梯度保真) -> 检验"梯度保真的残差能否给出可迁移策略"。
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import srbm_dynamics as srb
from srbm_dynamics import g_decay, SRBMParams
from srbm_residual import NOMINAL, accel_from_XU, sample_XU, Residual as AccelResidual
from srbm_residual_contact import ContactResidual
from srbm_residual_constrained import contact_accel_c
from srbm_train import Policy, Z_TARGET, EXT_SCALE

# R6 用**更大失配**的 teacher(局部, 不动共享 REAL/R1-R5), 让模型误差真正影响策略迁移
REAL_BIG = SRBMParams(m=16.0, I=0.65, k_n=14000.0, k_d=65.0, mu=0.45)

C_NOM, C_FREE, C_RES, C_TEA = "#8C8C8C", "#C44E52", "#55A868", "#4C72B0"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


# ---- 三种动力学的加速度函数: XU (N,8) -> (N,3) ----
def f_nominal(XU):
    return contact_accel_c(XU, NOMINAL, None)            # = 标称平滑接触

def f_teacher(XU):
    return accel_from_XU(XU, REAL_BIG, "smooth")         # 真实系统(大失配)


def train_residual_local(kind, n_iters, dev, scale, seed=0, lr=2e-3):
    """局部训练残差拟合 REAL_BIG。kind: 'free'=自由加速度残差 / 'constr'=R4约束接触力残差。"""
    torch.manual_seed(seed)
    r = (AccelResidual() if kind == "free" else ContactResidual()).to(dev).to(torch.get_default_dtype())
    if kind == "free":
        student = lambda XU: accel_from_XU(XU, NOMINAL, "smooth") + r(XU)
    else:
        student = lambda XU: contact_accel_c(XU, NOMINAL, r, gated=True, cone=True, alpha_n=0.6)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = accel_from_XU(XU, REAL_BIG, "smooth")
        loss = (((student(XU) - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return r, student


def sample_init(B, dev):
    return (torch.zeros(B, device=dev),
            0.40 + 0.03 * torch.randn(B, device=dev),
            0.20 * (2 * torch.rand(B, device=dev) - 1),     # θ0 ∈ ±0.2
            0.05 * torch.randn(B, device=dev),
            torch.zeros(B, device=dev),
            0.2 * torch.randn(B, device=dev))


def closed_loop(policy, accel_fn, state0, n, dt, grad_decay=1.0):
    px, pz, th, vx, vz, om = state0
    decay = grad_decay ** dt
    loss = px.new_zeros(())
    ths = []
    for _ in range(n):
        if grad_decay != 1.0:
            px, pz, th = g_decay(px, decay), g_decay(pz, decay), g_decay(th, decay)
            vx, vz, om = g_decay(vx, decay), g_decay(vz, decay), g_decay(om, decay)
        feat = torch.stack([pz - Z_TARGET, th, vx, vz, om], dim=-1)
        ext = policy(feat)                                  # (B,2)
        XU = torch.stack([px, pz, th, vx, vz, om, ext[:, 0], ext[:, 1]], dim=-1)
        A = accel_fn(XU)
        vx = vx + dt * A[:, 0]; vz = vz + dt * A[:, 1]; om = om + dt * A[:, 2]
        px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
        loss = loss + (4.0 * th**2 + 10.0 * (pz - Z_TARGET)**2
                       + 0.2 * (vx**2 + vz**2 + om**2) + 0.05 * (ext**2).sum(-1)).mean()
        ths.append(th)
    return loss / n, torch.stack(ths)


def train_policy(accel_fn, n_iters, dev, dt=2e-3, horizon=300, B=64, grad_decay=0.9,
                 lr=2e-3, clip=1.0, seed=0):
    torch.manual_seed(seed)
    pol = Policy().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    curve = []
    for _ in range(n_iters):
        s0 = sample_init(B, dev)
        loss, _ = closed_loop(pol, accel_fn, s0, horizon, dt, grad_decay)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), clip); opt.step()
        curve.append(float(loss.item()) if torch.isfinite(loss) else float("nan"))
    return pol, curve


@torch.no_grad()
def evaluate(policy, accel_fn, dev, dt=2e-3, horizon=300, B=256, seed=999):
    torch.manual_seed(seed)
    s0 = sample_init(B, dev)
    loss, ths = closed_loop(policy, accel_fn, s0, horizon, dt, 1.0)
    return float(loss.item()), ths.cpu().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_res = 300 if args.quick else 1500
    n_pol = 60 if args.quick else 200
    print(f"device={dev} residual_iters={n_res} policy_iters={n_pol}")

    # 1. 训练两种残差(自由加速度 / R4约束接触力)拟合 REAL_BIG
    big = sample_XU(4096, dev)
    with torch.no_grad():
        scale = (accel_from_XU(big, REAL_BIG, "smooth")
                 - accel_from_XU(big, NOMINAL, "smooth")).std(0).clamp_min(1e-6)
    _, f_free = train_residual_local("free", n_res, dev, scale, seed=0)
    _, f_constr = train_residual_local("constr", n_res, dev, scale, seed=0)

    # 2. 四个策略, 各经一种动力学训练
    models = {"nominal": f_nominal, "free-res": f_free, "constr-res": f_constr, "teacher": f_teacher}
    pols, train_curves = {}, {}
    for name, fn in models.items():
        pols[name], train_curves[name] = train_policy(fn, n_pol, dev, seed=0)
        print(f"  trained π_{name}: final train loss = {train_curves[name][-1]:.4f}")

    # 3. 每个策略都部署到 teacher; 同时记录在自己模型上的 loss(算 gap)
    rows = {}
    traj_teacher = {}
    for name in models:
        own_loss, _ = evaluate(pols[name], models[name], dev)        # 在自己训练的模型上
        dep_loss, ths = evaluate(pols[name], f_teacher, dev)          # 部署到 teacher
        rows[name] = dict(own=own_loss, deploy_teacher=dep_loss, gap=dep_loss - own_loss)
        traj_teacher[name] = ths
        print(f"  π_{name:8s}: own={own_loss:.4f}  deploy@teacher={dep_loss:.4f}  gap={rows[name]['gap']:+.4f}")
    json.dump(rows, open(os.path.join(HERE, "results_phase3e.json"), "w"), indent=2, ensure_ascii=False)

    # ---------- 图 R6 ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    names = ["nominal", "free-res", "constr-res", "teacher"]
    cols = [C_NOM, C_FREE, C_RES, C_TEA]
    # (a) 部署到 teacher 的平衡 loss
    dep = [rows[n]["deploy_teacher"] for n in names]
    ax[0].bar(range(4), dep, color=cols)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"π_{n}" for n in names], fontsize=8, rotation=10)
    ax[0].set_ylabel("balance loss deployed on TEACHER (log)")
    ax[0].set_yscale("log")
    ax[0].axhline(rows["teacher"]["deploy_teacher"], color=C_TEA, ls="--", lw=1, label="oracle")
    ax[0].set_title("(a) policy transfer to real system\n(lower=better; oracle=π_teacher)")
    ax[0].legend(fontsize=8)
    for i, n in enumerate(names):
        ax[0].text(i, dep[i], f"{dep[i]:.3f}", ha="center", va="bottom", fontsize=7)
    # (b) 利用漏洞 gap: own-model loss vs deploy-teacher loss
    x = np.arange(4); w = 0.38
    ax[1].bar(x - w / 2, [rows[n]["own"] for n in names], w, color=cols, alpha=.5, label="on own model")
    ax[1].bar(x + w / 2, [rows[n]["deploy_teacher"] for n in names], w, color=cols, hatch="//",
              edgecolor="k", label="deployed on teacher")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"π_{n}" for n in names], fontsize=8, rotation=10)
    ax[1].set_ylabel("balance loss (log)"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
    ax[1].set_title("(b) own-model vs teacher loss\n(free-res: unstable in closed loop)")
    # (c) 部署到 teacher 的姿态轨迹
    tt = np.arange(traj_teacher["nominal"].shape[0]) * 2e-3
    for n, c in zip(names, cols):
        for i in range(0, traj_teacher[n].shape[1], 48):
            ax[2].plot(tt, traj_teacher[n][:, i], color=c, lw=0.7, alpha=0.5)
        ax[2].plot([], [], color=c, label=f"π_{n}")
    ax[2].axhline(0, color="k", lw=.6, ls=":")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("pitch θ on teacher [rad]")
    ax[2].set_title("(c) pitch on real system"); ax[2].legend(fontsize=8)
    fig.suptitle("R6  Closed-loop sim-to-sim: policy trained through residual, deployed on real teacher",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R6_closedloop_sim2sim.png"), bbox_inches="tight")
    plt.close(fig)
    print("[R6] figure saved.")


if __name__ == "__main__":
    main()
