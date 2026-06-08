"""
更难任务上重测残差路线 [R7]: 移动参考轨迹跟踪 (高度 + 俯仰), 多维指标
==========================================================================
R6 的平衡任务**反馈太鲁棒** —— 稳态时负反馈能掩盖模型误差, 故 nominal≈oracle,
说服力有限。R7 换一个**前馈主导、模型敏感**的任务来真正分开三种残差:

  任务: 跟踪**移动**的高度参考 z_ref(t) 与俯仰参考 θ_ref(t) (正弦, 较快)。
  为什么更难/更模型敏感:
    - 跟踪移动 z_ref 需要净法向力 ≈ m(z̈_ref+g); 标称模型以为 m=10, 真实 m=16
      -> 标称训练的策略前馈力偏小 -> 真机上**高度跟踪滞后/下沉**。
    - 跟踪移动 θ_ref 需要力矩 ≈ I·θ̈_ref; 标称 I=0.35, 真实 I=0.65
      -> 前馈力矩偏小 -> 真机上**姿态跟踪峰值误差大** (呼应 R5 姿态通道)。
  反馈只能补稳态, 补不了快参考的前馈 -> 模型精度真正起作用。

  四个策略各经一种动力学的 BPTT 训练, 全部部署到**真实 teacher**, 比多维指标:
    高度跟踪 RMS / 姿态跟踪 RMS / 姿态峰值误差 / 速度跟踪误差 / 摔倒率 / 控制能耗 /
    自模型 loss 与 teacher loss 的利用 gap。
  teacher 用 R6 的大失配 REAL_BIG。残差/加速度函数复用 R6, 避免重复实现。

注: 残差仅在 sample_XU 的状态域 (pz∈[0.30,0.345], θ∈±0.12) 内受过训练, 故参考
幅度取小 (AZ=0.010, ATH=0.06) 以留在域内; 控制幅度 EXT_SCALE=0.06 同 R6 (略超 ext
训练带, 这正是"结构化残差有界外推稳, 自由残差外推差"叙事的一部分)。
"""
from __future__ import annotations
import argparse, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from srbm_dynamics import g_decay
from srbm_residual import NOMINAL, sample_XU
from srbm_residual_closedloop import REAL_BIG, f_nominal, f_teacher, train_residual_local
from srbm_train import EXT_SCALE

C_NOM, C_FREE, C_RES, C_TEA = "#8C8C8C", "#C44E52", "#55A868", "#4C72B0"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# ---- 移动参考轨迹 (留在残差训练域内, 但**足够快**以让前馈/模型精度真正起作用) ----
# 频率取较高: z̈_ref∝f² 大 -> 跟踪需要的净法向力 ≈ m(z̈+g) 中 m 误差被放大(质量敏感);
# θ̈_ref∝f² 大 -> 跟踪需要的力矩 ≈ I·θ̈ 中 I 误差被放大(转动惯量敏感, 呼应 R5 姿态通道)。
Z0 = 0.322         # 高度基线(略抬, 使 pz 谷值≥0.30 留在残差域)
AZ = 0.012         # 高度幅度 -> pz∈[0.310,0.334]
FZ = 2.5           # 高度频率 [Hz] (z̈ 峰值≈2.5 m/s² -> 前馈力主导)
ATH = 0.06         # 俯仰幅度 [rad] (留在 θ∈±0.12 域内)
FTH = 2.0          # 俯仰频率 [Hz] (θ̈ 峰值≈9.5 rad/s² -> 前馈力矩主导)
WZ, WTH = 2 * math.pi * FZ, 2 * math.pi * FTH
FALL_THRESH = 0.5  # |θ|>此值算摔倒

# 损失权重 (训练 + 评估同一套, 使 own/deploy gap 有意义)。加重跟踪项以暴露前馈误差。
W_Z, W_TH, W_VZ, W_OM, W_VX, W_EXT = 12.0, 8.0, 0.5, 0.5, 0.1, 0.05


def refs(t: float, device):
    """t 时刻参考: z_ref, ż_ref, θ_ref, θ̇_ref (标量张量, 广播到 batch)。"""
    zr = Z0 + AZ * math.sin(WZ * t)
    zdr = AZ * WZ * math.cos(WZ * t)
    thr = ATH * math.sin(WTH * t)
    thdr = ATH * WTH * math.cos(WTH * t)
    f = lambda v: torch.tensor(v, device=device)
    return f(zr), f(zdr), f(thr), f(thdr)


class TrackPolicy(nn.Module):
    """[pz-zref, θ-θref, vx, vz-żref, om-θ̇ref, sin/cos(WZ t), sin/cos(WTH t)] (9) -> 2 腿伸长。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat):
        return EXT_SCALE * torch.tanh(self.net(feat))


def closed_loop_track(policy, accel_fn, state0, n, dt, grad_decay=1.0, log_traj=False):
    """跟踪 rollout。返回 (loss, logs)。log_traj=True 时记录逐步轨迹(仅评估用,
    训练时关闭以免给 BPTT 图增加 9×n 的栈开销)。"""
    px, pz, th, vx, vz, om = state0
    decay = grad_decay ** dt
    loss = px.new_zeros(())
    log = {k: [] for k in ["pz", "th", "vz", "om", "zr", "thr", "zdr", "thdr", "ext2"]}
    for step in range(n):
        if grad_decay != 1.0:
            px, pz, th = g_decay(px, decay), g_decay(pz, decay), g_decay(th, decay)
            vx, vz, om = g_decay(vx, decay), g_decay(vz, decay), g_decay(om, decay)
        t = step * dt
        zr, zdr, thr, thdr = refs(t, px.device)
        cwz, swz = math.cos(WZ * t), math.sin(WZ * t)
        cwt, swt = math.cos(WTH * t), math.sin(WTH * t)
        z1 = torch.ones_like(px)
        feat = torch.stack([pz - zr, th - thr, vx, vz - zdr, om - thdr,
                            swz * z1, cwz * z1, swt * z1, cwt * z1], dim=-1)
        ext = policy(feat)                                  # (B,2)
        XU = torch.stack([px, pz, th, vx, vz, om, ext[:, 0], ext[:, 1]], dim=-1)
        A = accel_fn(XU)
        vx = vx + dt * A[:, 0]; vz = vz + dt * A[:, 1]; om = om + dt * A[:, 2]
        px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
        loss = loss + (W_Z * (pz - zr) ** 2 + W_TH * (th - thr) ** 2
                       + W_VZ * (vz - zdr) ** 2 + W_OM * (om - thdr) ** 2
                       + W_VX * vx ** 2 + W_EXT * (ext ** 2).sum(-1)).mean()
        if log_traj:
            with torch.no_grad():
                for k, v in zip(["pz", "th", "vz", "om", "ext2"],
                                [pz, th, vz, om, (ext ** 2).sum(-1)]):
                    log[k].append(v)
                for k, v in zip(["zr", "thr", "zdr", "thdr"], [zr, thr, zdr, thdr]):
                    log[k].append(v * z1)
    loss = loss / n
    if log_traj:
        log = {k: torch.stack(v) for k, v in log.items()}   # each (n, B)
    return loss, log


def sample_init_track(B, dev):
    return (torch.zeros(B, device=dev),
            Z0 + 0.010 * torch.randn(B, device=dev),
            0.08 * (2 * torch.rand(B, device=dev) - 1),     # θ0 ∈ ±0.08 (域内)
            0.04 * torch.randn(B, device=dev),
            0.04 * torch.randn(B, device=dev),
            0.15 * torch.randn(B, device=dev))


def train_policy_track(accel_fn, n_iters, dev, dt=2e-3, horizon=400, B=64,
                       grad_decay=0.9, lr=2e-3, clip=1.0, seed=0):
    torch.manual_seed(seed)
    pol = TrackPolicy().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    curve = []
    for _ in range(n_iters):
        s0 = sample_init_track(B, dev)
        loss, _ = closed_loop_track(pol, accel_fn, s0, horizon, dt, grad_decay)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), clip)
            opt.step()
        curve.append(float(loss.item()) if torch.isfinite(loss) else float("nan"))
    return pol, curve


@torch.no_grad()
def evaluate_track(policy, accel_fn, dev, dt=2e-3, horizon=400, B=256, seed=999):
    """部署评估: 返回多维指标 dict + 姿态轨迹 (供画图)。"""
    torch.manual_seed(seed)
    s0 = sample_init_track(B, dev)
    loss, log = closed_loop_track(policy, accel_fn, s0, horizon, dt, 1.0, log_traj=True)
    pz, th, vz, om = log["pz"], log["th"], log["vz"], log["om"]
    zr, thr, zdr, thdr = log["zr"], log["thr"], log["zdr"], log["thdr"]
    finite = torch.isfinite(th).all(dim=0)                  # 过滤发散 rollout 再统计峰值
    th_peak = (th - thr).abs().max(dim=0).values            # (B,) 每条最大姿态跟踪误差
    metrics = dict(
        loss=float(loss.item()) if torch.isfinite(loss) else float("nan"),
        z_track_rms=float(((pz - zr) ** 2).mean().sqrt()),
        th_track_rms=float(((th - thr) ** 2).mean().sqrt()),
        th_peak_err=float(th_peak[finite].mean()) if finite.any() else float("nan"),
        vel_err=float((((vz - zdr) ** 2 + (om - thdr) ** 2).mean()).sqrt()),
        fall_rate=float((th.abs().max(dim=0).values > FALL_THRESH).float().mean()),
        energy=float(log["ext2"].mean()),
    )
    return metrics, th.cpu().numpy(), thr.cpu().numpy(), pz.cpu().numpy(), zr.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_res = 300 if args.quick else 1500
    n_pol = 80 if args.quick else 250
    print(f"device={dev} residual_iters={n_res} policy_iters={n_pol}", flush=True)

    # 1. 训练两种残差拟合 REAL_BIG (复用 R6)
    big = sample_XU(4096, dev)
    with torch.no_grad():
        from srbm_residual import accel_from_XU
        scale = (accel_from_XU(big, REAL_BIG, "smooth")
                 - accel_from_XU(big, NOMINAL, "smooth")).std(0).clamp_min(1e-6)
    _, f_free = train_residual_local("free", n_res, dev, scale, seed=0)
    _, f_constr = train_residual_local("constr", n_res, dev, scale, seed=0)

    # 2. 四个策略, 各经一种动力学训练同一跟踪任务
    models = {"nominal": f_nominal, "free-res": f_free, "constr-res": f_constr, "teacher": f_teacher}
    pols, curves = {}, {}
    for name, fn in models.items():
        pols[name], curves[name] = train_policy_track(fn, n_pol, dev, seed=0)
        last = next((x for x in reversed(curves[name]) if x == x), float("nan"))
        print(f"  trained π_{name:9s}: final train loss = {last:.4f}", flush=True)

    # 3. 每个策略部署到 teacher (多维指标) + 记录自模型 loss 算 gap
    rows, traj = {}, {}
    for name in models:
        own, *_ = evaluate_track(pols[name], models[name], dev)
        dep, th, thr, pz, zr = evaluate_track(pols[name], f_teacher, dev)
        dep["gap"] = dep["loss"] - own["loss"]
        dep["own_loss"] = own["loss"]
        rows[name] = dep
        traj[name] = (th, thr, pz, zr)
        print(f"  π_{name:9s} @teacher: loss={dep['loss']:.4f} z_rms={dep['z_track_rms']:.4f} "
              f"θ_rms={dep['th_track_rms']:.4f} θ_peak={dep['th_peak_err']:.4f} "
              f"vel={dep['vel_err']:.3f} fall={dep['fall_rate']:.3f} E={dep['energy']:.4f} gap={dep['gap']:+.4f}",
              flush=True)
    json.dump(rows, open(os.path.join(HERE, "results_phase3f.json"), "w"), indent=2, ensure_ascii=False)

    _plot(rows, traj, curves)
    print("[R7] figure saved.")


def _plot(rows, traj, curves):
    names = ["nominal", "free-res", "constr-res", "teacher"]
    cols = [C_NOM, C_FREE, C_RES, C_TEA]
    fig = plt.figure(figsize=(15, 8))

    # (a) 部署到 teacher 的任务 loss
    ax = fig.add_subplot(2, 3, 1)
    dep = [rows[n]["loss"] for n in names]
    ax.bar(range(4), dep, color=cols)
    ax.axhline(rows["teacher"]["loss"], color=C_TEA, ls="--", lw=1, label="oracle")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"π_{n}" for n in names], fontsize=8, rotation=12)
    ax.set_ylabel("tracking loss on TEACHER (log)"); ax.set_yscale("log")
    ax.set_title("(a) deploy loss on real system\n(lower=better)"); ax.legend(fontsize=8)
    for i, n in enumerate(names):
        ax.text(i, dep[i], f"{dep[i]:.3f}", ha="center", va="bottom", fontsize=7)

    # (b) 多维指标雷达式分组条 (相对 oracle 归一, >1 = 比 oracle 差)
    ax = fig.add_subplot(2, 3, 2)
    keys = ["z_track_rms", "th_track_rms", "th_peak_err", "vel_err", "energy"]
    klab = ["z RMS", "θ RMS", "θ peak", "vel err", "energy"]
    base = {k: rows["teacher"][k] for k in keys}
    x = np.arange(len(keys)); w = 0.2
    for j, n in enumerate(["nominal", "free-res", "constr-res"]):
        vals = [rows[n][k] / (base[k] + 1e-9) for k in keys]
        ax.bar(x + (j - 1) * w, vals, w, color=cols[names.index(n)], label=f"π_{n}")
    ax.axhline(1.0, color=C_TEA, ls="--", lw=1, label="oracle=1")
    ax.set_xticks(x); ax.set_xticklabels(klab, fontsize=8, rotation=12)
    ax.set_ylabel("metric / oracle (lower=better)")
    ax.set_title("(b) multi-metric vs oracle\n(>1 = worse than π_teacher)"); ax.legend(fontsize=7)

    # (c) 利用 gap: own-model vs deploy-teacher loss
    ax = fig.add_subplot(2, 3, 3)
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, [rows[n]["own_loss"] for n in names], w, color=cols, alpha=.5, label="on own model")
    ax.bar(x + w / 2, [rows[n]["loss"] for n in names], w, color=cols, hatch="//",
           edgecolor="k", label="deployed on teacher")
    ax.set_xticks(x); ax.set_xticklabels([f"π_{n}" for n in names], fontsize=8, rotation=12)
    ax.set_ylabel("loss (log)"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("(c) own-model vs teacher (gap=exploitation)")

    # (d) 摔倒率
    ax = fig.add_subplot(2, 3, 4)
    fr = [rows[n]["fall_rate"] for n in names]
    ax.bar(range(4), fr, color=cols)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"π_{n}" for n in names], fontsize=8, rotation=12)
    ax.set_ylabel(f"fall rate (|θ|>{FALL_THRESH})"); ax.set_title("(d) fall rate on teacher")
    for i, n in enumerate(names):
        ax.text(i, fr[i], f"{fr[i]:.2f}", ha="center", va="bottom", fontsize=7)

    # (e) 高度跟踪轨迹 (teacher 上, 几条样本 + 参考)
    ax = fig.add_subplot(2, 3, 5)
    th0, thr0, pz0, zr0 = traj["nominal"]
    tt = np.arange(pz0.shape[0]) * 2e-3
    ax.plot(tt, zr0[:, 0], "k--", lw=1.2, label="z_ref")
    for n in ["nominal", "constr-res"]:
        _, _, pz, _ = traj[n]
        ax.plot(tt, pz[:, :6], color=cols[names.index(n)], lw=0.7, alpha=0.6)
        ax.plot([], [], color=cols[names.index(n)], label=f"π_{n}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("height pz [m]")
    ax.set_title("(e) height tracking on teacher"); ax.legend(fontsize=7)

    # (f) 俯仰跟踪轨迹
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tt, thr0[:, 0], "k--", lw=1.2, label="θ_ref")
    for n in ["nominal", "constr-res"]:
        th, _, _, _ = traj[n]
        ax.plot(tt, th[:, :6], color=cols[names.index(n)], lw=0.7, alpha=0.6)
        ax.plot([], [], color=cols[names.index(n)], label=f"π_{n}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("pitch θ [rad]")
    ax.set_title("(f) pitch tracking on teacher"); ax.legend(fontsize=7)

    fig.suptitle("R7  Harder task (moving height+pitch tracking): does the physically-"
                 "constrained residual train a genuinely better policy?", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R7_taskhard_tracking.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
