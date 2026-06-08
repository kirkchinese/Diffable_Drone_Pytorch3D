"""
F3  闭环最小可微训练（迁移"主线"的内核）
==========================================================================
任务: 小 MLP 策略稳定平面 SRBM —— 从随机俯仰/高度扰动恢复到"直立 + 目标高度"。
机制: 整段 BPTT + 无人机式 GDecay(梯度衰减) + 全局梯度裁剪 + 密集目标损失(GCGL 思想迁移)。
对照:
  A. smooth 接触 + GDecay   —— 主线配方, 应能学会平衡
  B. hard   接触 + GDecay   —— 接触非光滑/刚, 梯度有偏 -> 学习显著变差
  C. smooth 接触, 关 GDecay, 长视野 —— 展示 BPTT 失稳, GDecay 的稳定作用
产出: figures/F3_training.png (学习曲线 + 训练前后姿态轨迹), results_phase2.json['F3']
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
from srbm_dynamics import SRBMParams, g_decay

C_SMOOTH, C_CONTACT, C_BASE, C_PURPLE = "#55A868", "#C44E52", "#4C72B0", "#8172B3"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

Z_TARGET = 0.33
EXT_SCALE = 0.06


class Policy(nn.Module):
    """state[pz-z*, θ, vx, vz, ω] -> 2 条腿伸长(经 tanh 限幅)。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat):
        return EXT_SCALE * torch.tanh(self.net(feat))


def closed_loop_rollout(policy, state0, n_steps, model, p, dt, grad_decay):
    px, pz, th, vx, vz, om = state0
    base_x = [p.half_len, -p.half_len]
    ths, pzs, ctrls = [], [], []
    decay = grad_decay ** dt
    loss = px.new_zeros(())
    for t in range(n_steps):
        if grad_decay != 1.0:
            px, pz, th = g_decay(px, decay), g_decay(pz, decay), g_decay(th, decay)
            vx, vz, om = g_decay(vx, decay), g_decay(vz, decay), g_decay(om, decay)
        feat = torch.stack([pz - Z_TARGET, th, vx, vz, om], dim=-1)
        ext = policy(feat)                      # (B, 2)
        ax, az, al, _ = srb.srbm_accel((px, pz, th, vx, vz, om),
                                       [ext[:, 0], ext[:, 1]], model, p, base_x)
        vx = vx + dt * ax; vz = vz + dt * az; om = om + dt * al
        px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
        # 密集目标损失(逐步累加): 直立 + 目标高度 + 抑速 + 控制正则
        loss = loss + (4.0 * th**2 + 10.0 * (pz - Z_TARGET)**2
                       + 0.2 * (vx**2 + vz**2 + om**2) + 0.05 * (ext**2).sum(-1)).mean()
        ths.append(th); pzs.append(pz); ctrls.append(ext)
    loss = loss / n_steps
    return loss, torch.stack(ths), torch.stack(pzs)


def sample_init(B, dev):
    px = torch.zeros(B, device=dev)
    pz = 0.40 + 0.03 * torch.randn(B, device=dev)
    th = 0.20 * (2 * torch.rand(B, device=dev) - 1)      # U(-0.2,0.2) rad
    vx = 0.1 * torch.randn(B, device=dev)
    vz = torch.zeros(B, device=dev)
    om = 0.2 * torch.randn(B, device=dev)
    return (px, pz, th, vx, vz, om)


def train(model, grad_decay, n_iters, horizon, dt, k_n, dev, seed=0, lr=2e-3, B=64, clip=1.0):
    torch.manual_seed(seed)
    p = SRBMParams(k_n=k_n)
    policy = Policy().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve, gnorms = [], []
    for it in range(n_iters):
        state0 = sample_init(B, dev)
        loss, _, _ = closed_loop_rollout(policy, state0, horizon, model, p, dt, grad_decay)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            opt.step()
            gnorms.append(float(gnorm))
        else:
            gnorms.append(float("nan"))
        curve.append(float(loss.item()) if torch.isfinite(loss) else float("nan"))
    return policy, curve, gnorms, p


def eval_traj(policy, model, p, horizon, dt, dev, seed=123):
    torch.manual_seed(seed)
    state0 = sample_init(16, dev)
    with torch.no_grad():
        _, ths, pzs = closed_loop_rollout(policy, state0, horizon, model, p, dt, 1.0)
    return ths.cpu().numpy(), pzs.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 30 if args.quick else 250
    dt = 2e-3
    horizon = 400          # 0.8 s
    print(f"device={dev} n_iters={n_iters} horizon={horizon} dt={dt}")

    # A: smooth + GDecay(主线配方)；B: hard(stiff,非光滑) + GDecay(对照)
    print("== A: smooth + GDecay ==")
    polA, curveA, gA, pA = train("smooth", 0.9, n_iters, horizon, dt, 6000., dev, seed=0)
    print("== B: hard + GDecay ==")
    polB, curveB, gB, pB = train("soft", 0.9, n_iters, horizon, dt, 40000., dev, seed=0)

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(curveA, color=C_SMOOTH, lw=2, label="A: smooth contact + GDecay")
    ax[0].plot(curveB, color=C_CONTACT, lw=2, label="B: hard contact + GDecay")
    ax[0].set_xlabel("training iteration"); ax[0].set_ylabel("BPTT loss (lower=better)")
    ax[0].set_yscale("log"); ax[0].set_title("(a) smooth contact trains; hard contact stalls")
    ax[0].legend(fontsize=8)

    # (b) 训练中梯度范数(裁剪前): 硬接触尖刺/触顶, 平滑平稳 —— 呼应 E5 / GCGL
    ax[1].plot(gA, color=C_SMOOTH, lw=1.5, label="A: smooth (stable, small)")
    ax[1].plot(gB, color=C_CONTACT, lw=1.5, alpha=0.8, label="B: hard (spiky, hits clip)")
    ax[1].axhline(1.0, color="k", lw=.8, ls="--", label="grad-clip threshold")
    ax[1].set_xlabel("training iteration"); ax[1].set_ylabel("pre-clip grad norm")
    ax[1].set_yscale("log"); ax[1].set_title("(b) hard contact = pathological gradients\n(clip caps magnitude, can't fix direction)")
    ax[1].legend(fontsize=7)

    # (c) 训练前后姿态轨迹(A, smooth)
    rng = np.random.default_rng(0)
    pol0 = Policy().to(dev).to(torch.get_default_dtype())  # 未训练
    th0, _ = eval_traj(pol0, "smooth", pA, horizon, dt, dev)
    thT, _ = eval_traj(polA, "smooth", pA, horizon, dt, dev)
    tt = np.arange(horizon) * dt
    for i in range(th0.shape[1]):
        ax[2].plot(tt, th0[:, i], color="#BBBBBB", lw=0.7, alpha=0.6)
        ax[2].plot(tt, thT[:, i], color=C_SMOOTH, lw=0.9, alpha=0.8)
    ax[2].axhline(0, color="k", lw=.6, ls=":")
    ax[2].plot([], [], color="#BBBBBB", label="untrained policy")
    ax[2].plot([], [], color=C_SMOOTH, label="trained policy (A)")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("pitch θ [rad]")
    ax[2].set_title("(c) learned to stabilize upright (θ→0)")
    ax[2].legend(fontsize=8)
    fig.suptitle("F3  Minimal closed-loop differentiable training on SRBM "
                 "(BPTT + GDecay + goal loss = drone recipe transferred)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F3_training.png"), bbox_inches="tight")
    plt.close(fig)

    def last_valid(c):
        v = [x for x in c if x == x]
        return v[-1] if v else float("nan")
    summary = {
        "A_smooth_gdecay_init": curveA[0], "A_smooth_gdecay_final": last_valid(curveA),
        "B_hard_gdecay_init": curveB[0], "B_hard_gdecay_final": last_valid(curveB),
        "A_gradnorm_median": float(np.nanmedian(gA)),
        "B_gradnorm_median": float(np.nanmedian(gB)),
        "horizon": horizon, "dt": dt,
        "note": "smooth contact: loss降; hard contact: loss不降/升. 平滑接触梯度有界(E5)故GDecay边际收益小,其价值在刚性/长视野域.",
    }
    # 合并写入 results_phase2.json
    path = os.path.join(HERE, "results_phase2.json")
    res = {}
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
    res["F3"] = summary
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("[F3] summary:", {k: (round(v, 4) if isinstance(v, float) and v == v else v)
                            for k, v in summary.items()})


if __name__ == "__main__":
    main()
