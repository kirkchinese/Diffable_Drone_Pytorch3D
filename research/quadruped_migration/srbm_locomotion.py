"""
F4  前向 locomotion: SRBM + 平滑接触 + 预定步态 + 可微训练
==========================================================================
从 F3 的"原地平衡"扩展到"定速前进"。机制:
  - 预定步态(prescribed gait): 两腿反相, 支撑相压腿+足端向后扫(摩擦->前推),
    摆动相抬腿+足端前摆复位 —— 即 Song 2024 的"预定接触序列"思想的最小实现。
  - 足端地面相对速度用**位置差分**自动涵盖机身运动+扫腿运动 -> 摩擦驱动前进。
  - 策略只学"在步态之上的修正量"以跟踪目标前速并保持直立(Song 2024: 步态/落足预定, 策略调制)。
对照: 平滑 vs 硬接触, 复用 BPTT + GDecay + 目标损失。
__main__ 为开环步态自检(无策略), 先验证步态能前进且不倒。
"""
from __future__ import annotations
import argparse, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import srbm_dynamics as srb
from srbm_dynamics import SRBMParams, foot_world, contact_force_2d, g_decay, G

TWO_PI = 2 * math.pi
C_SMOOTH, C_CONTACT, C_BASE, C_PURPLE = "#55A868", "#C44E52", "#4C72B0", "#8172B3"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# shuffle 步态: 两腿始终轻接触(ext 恒正 -> 不抬腿 -> 保持宽支撑基 -> 俯仰稳定),
# 前进靠"支撑相足端向后扫 + 摩擦"。求稳优先, 策略再在其上放大。
DEFAULT_GAIT = dict(freq=2.0, ext_mid=0.013, ext_amp=0.009, sweep=0.050)
LOCO_PARAMS = dict(k_d=60.0, mu=0.9, v_eps=0.10)  # 大法向阻尼抑弹跳; v_eps放宽使摩擦梯度更平缓
Z_TARGET = 0.31


def gait_cmd(phase_k, gp):
    """给定某腿相位 phase_k(标量/张量), 返回 (ext, foot_dx) 标称步态指令。"""
    ext = gp["ext_mid"] + gp["ext_amp"] * torch.cos(TWO_PI * phase_k)
    fdx = -gp["sweep"] * torch.sin(TWO_PI * phase_k)
    return ext, fdx


def locomotion_rollout(state0, n_steps, policy=None, model="smooth", p: SRBMParams = SRBMParams(),
                       dt=2e-3, gp=None, grad_decay=1.0, target_vx=0.4, base_x=None):
    """可微 locomotion rollout。policy=None 时为纯开环步态。

    返回 traj dict('px','pz','th','vx',...), 以及(若 target_vx 给定)逐步累计 loss。
    """
    if gp is None:
        gp = DEFAULT_GAIT
    if base_x is None:
        base_x = [p.half_len, -p.half_len]
    px, pz, th, vx, vz, om = state0
    wk = TWO_PI * gp["freq"]

    out = {kk: [v] for kk, v in zip(["px", "pz", "th", "vx", "vz", "om"],
                                    [px, pz, th, vx, vz, om])}
    contacts = [[] for _ in base_x]
    loss = px.new_zeros(()) if target_vx is not None else None
    decay = grad_decay ** dt

    for t in range(n_steps):
        if grad_decay != 1.0:
            px, pz, th = g_decay(px, decay), g_decay(pz, decay), g_decay(th, decay)
            vx, vz, om = g_decay(vx, decay), g_decay(vz, decay), g_decay(om, decay)
        phase = (gp["freq"] * (t * dt)) % 1.0
        phase_t = px.new_full(px.shape if px.dim() else (), phase) if hasattr(px, "dim") else phase

        # 策略修正量(可选): 输入状态+步态相位 -> 每腿(Δext, Δfdx)
        if policy is not None:
            feat = torch.stack([
                pz - Z_TARGET, th, vx - target_vx, vz, om,
                math.cos(TWO_PI * phase) + 0 * px, math.sin(TWO_PI * phase) + 0 * px,
            ], dim=-1)
            mod = policy(feat)  # (..., 2*n_legs)
        Fx = torch.zeros_like(px); Fz = torch.zeros_like(px); tau = torch.zeros_like(px)
        cth, sth = torch.cos(th), torch.sin(th)
        for k, bx in enumerate(base_x):
            ph_k = (phase + 0.5 * k) % 1.0
            cph, sph = math.cos(TWO_PI * ph_k), math.sin(TWO_PI * ph_k)
            ext_nom = gp["ext_mid"] + gp["ext_amp"] * cph
            fdx_nom = -gp["sweep"] * sph
            dext_dt = -gp["ext_amp"] * sph * wk        # 解析(仅 t 的函数) -> autograd 梯度为 0
            dfdx_dt = -gp["sweep"] * cph * wk
            ext_k = ext_nom + torch.zeros_like(px)
            fdx_k = fdx_nom + torch.zeros_like(px)
            if policy is not None:
                ext_k = ext_k + 0.02 * torch.tanh(mod[..., 2 * k])
                fdx_k = fdx_k + 0.04 * torch.tanh(mod[..., 2 * k + 1])
            fx, fz = foot_world(px, pz, th, ext_k, bx, p.leg_nominal, fdx_k)
            rx, rz = fx - px, fz - pz
            # 足端世界速度 = v_com + ω×r + R(θ)·(dfdx_dt, -dext_dt)  (扫腿速度解析, 不被 /dt 放大)
            lvx = cth * dfdx_dt - sth * (-dext_dt)
            lvz = sth * dfdx_dt + cth * (-dext_dt)
            vfx = vx - om * rz + lvx
            vfz = vz + om * rx + lvz
            Fcx, Fcz = contact_force_2d(fx, fz, vfx, vfz, model, p)
            Fx = Fx + Fcx; Fz = Fz + Fcz
            tau = tau + (rx * Fcz - rz * Fcx)
            contacts[k].append(Fcz)
        ax, az, al = Fx / p.m, Fz / p.m - G, tau / p.I
        vx = vx + dt * ax; vz = vz + dt * az; om = om + dt * al
        px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
        for kk, v in zip(["px", "pz", "th", "vx", "vz", "om"], [px, pz, th, vx, vz, om]):
            out[kk].append(v)
        if loss is not None:
            # 目标损失: 跟踪前速 + 保持高度 + 直立 + 抑垂直/角速度
            loss = loss + (3.0 * (vx - target_vx) ** 2 + 8.0 * (pz - Z_TARGET) ** 2
                           + 2.0 * th ** 2 + 0.3 * (vz ** 2 + om ** 2)).mean()
    out = {kk: torch.stack(v) for kk, v in out.items()}
    out["contact0"] = torch.stack(contacts[0])
    out["contact1"] = torch.stack(contacts[1])
    if loss is not None:
        loss = loss / n_steps
    return out, loss


class LocoPolicy(nn.Module):
    """state+phase(7) -> (Δext, Δfdx) x 2 legs = 4。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 4))

    def forward(self, feat):
        return self.net(feat)


def _scalar_state(dev, px=0.0, pz=0.32, th=0.0, vx=0.0):
    f = lambda v: torch.tensor(v, device=dev)
    return (f(px), f(pz), f(th), f(vx), f(0.0), f(0.0))


def open_loop_test(dev):
    """开环步态自检: 纯步态(无策略)能否前进且不倒。"""
    p = SRBMParams(**LOCO_PARAMS)
    n, dt = 3000, 2e-3
    out, _ = locomotion_rollout(_scalar_state(dev), n, policy=None, model="smooth",
                                p=p, dt=dt, target_vx=None)
    px = out["px"].detach().cpu().numpy()
    pz = out["pz"].detach().cpu().numpy()
    th = out["th"].detach().cpu().numpy()
    vx = out["vx"].detach().cpu().numpy()
    t = np.arange(n + 1) * dt
    print(f"[open-loop gait] forward px: {px[0]:.3f} -> {px[-1]:.3f} m  (Δ={px[-1]-px[0]:+.3f})")
    print(f"[open-loop gait] mean vx (last 1s): {vx[int(-1.0/dt):].mean():+.3f} m/s")
    print(f"[open-loop gait] pz range [{pz.min():.3f},{pz.max():.3f}]  |θ|max={np.abs(th).max():.3f} rad")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(t, px, color=C_SMOOTH, lw=2); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("px [m]")
    ax[0].set_title("(a) forward progress")
    ax[1].plot(t, pz, color=C_BASE, lw=1.5, label="pz"); ax[1].plot(t, th, color=C_CONTACT, lw=1.5, label="θ")
    ax[1].axhline(Z_TARGET, color="k", ls=":", lw=.8); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8)
    ax[1].set_title("(b) height & pitch")
    ax[2].plot(t, vx, color=C_PURPLE, lw=1.5); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("vx [m/s]")
    ax[2].set_title("(c) forward velocity")
    fig.suptitle("F4 open-loop gait self-check (no policy)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_openloop_check.png"), bbox_inches="tight")
    plt.close(fig)


def sample_init_loco(B, dev):
    f = lambda v: v
    return (torch.zeros(B, device=dev),
            0.31 + 0.02 * torch.randn(B, device=dev),
            0.05 * torch.randn(B, device=dev),
            0.05 * torch.randn(B, device=dev),
            torch.zeros(B, device=dev), torch.zeros(B, device=dev))


def train_loco(model, k_n, target_vx, n_iters, horizon, dt, dev, grad_decay=0.5,
               B=48, clip=1.0, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    p = SRBMParams(k_n=k_n, **LOCO_PARAMS)
    policy = LocoPolicy().to(dev).to(torch.get_default_dtype())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve, gnorms = [], []
    for it in range(n_iters):
        s0 = sample_init_loco(B, dev)
        _, loss = locomotion_rollout(s0, horizon, policy=policy, model=model, p=p,
                                     dt=dt, target_vx=target_vx, grad_decay=grad_decay)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            opt.step(); gnorms.append(float(gn))
        else:
            gnorms.append(float("nan"))
        curve.append(float(loss.item()) if torch.isfinite(loss) else float("nan"))
    return policy, curve, gnorms, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--openloop", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    if args.openloop:
        open_loop_test(dev)
        return

    n_iters = 25 if args.quick else 200
    dt, horizon, tv = 2e-3, 700, 0.35
    print(f"device={dev} n_iters={n_iters} horizon={horizon} target_vx={tv}")

    print("== A: smooth contact ==")
    polA, curveA, gA, pA = train_loco("smooth", 6000., tv, n_iters, horizon, dt, dev, seed=0)
    print("== B: hard contact ==")
    polB, curveB, gB, pB = train_loco("soft", 40000., tv, n_iters, horizon, dt, dev, seed=0)

    # 评估: 单条 episode 的 vx(t), px(t) (未训练 vs 训练后 smooth)
    def eval_ep(policy, model, p):
        s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.31, device=dev),
              torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev),
              torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev))
        with torch.no_grad():
            out, _ = locomotion_rollout(s0, 1500, policy=policy, model=model, p=p, dt=dt,
                                        target_vx=tv, grad_decay=1.0)
        return out["vx"].cpu().numpy(), out["px"].cpu().numpy(), out["th"].cpu().numpy()
    pol0 = LocoPolicy().to(dev).to(torch.get_default_dtype())
    vx0, px0, th0 = eval_ep(pol0, "smooth", pA)
    vxT, pxT, thT = eval_ep(polA, "smooth", pA)
    tt = np.arange(len(vx0)) * dt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(curveA, color=C_SMOOTH, lw=2, label="A: smooth contact")
    ax[0].plot(curveB, color=C_CONTACT, lw=2, label="B: hard contact")
    ax[0].set_yscale("log"); ax[0].set_xlabel("training iteration"); ax[0].set_ylabel("BPTT loss")
    ax[0].set_title("(a) learning curves"); ax[0].legend(fontsize=8)
    ax[1].plot(tt, vx0, color="#BBBBBB", lw=1.5, label="untrained")
    ax[1].plot(tt, vxT, color=C_SMOOTH, lw=1.8, label="trained (A)")
    ax[1].axhline(tv, color="k", ls="--", lw=1, label=f"target {tv} m/s")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("forward vx [m/s]"); ax[1].set_title("(b) velocity tracking")
    ax[1].legend(fontsize=8)
    ax[2].plot(tt, px0, color="#BBBBBB", lw=1.5, label="untrained")
    ax[2].plot(tt, pxT, color=C_SMOOTH, lw=1.8, label="trained (A)")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("forward distance px [m]"); ax[2].set_title("(c) forward progress")
    ax[2].legend(fontsize=8)
    fig.suptitle("F4  Forward locomotion: prescribed gait + smooth contact + drone recipe "
                 "learns velocity tracking", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_training.png"), bbox_inches="tight")
    plt.close(fig)

    def lastv(c):
        v = [x for x in c if x == x]; return v[-1] if v else float("nan")
    summary = {
        "A_smooth_init": curveA[0], "A_smooth_final": lastv(curveA),
        "B_hard_init": curveB[0], "B_hard_final": lastv(curveB),
        "A_gradnorm_median": float(np.nanmedian(gA)), "B_gradnorm_median": float(np.nanmedian(gB)),
        "trained_mean_vx_last1s": float(vxT[int(-1.0/dt):].mean()),
        "untrained_mean_vx_last1s": float(vx0[int(-1.0/dt):].mean()),
        "trained_px_final": float(pxT[-1]), "target_vx": tv,
        "trained_abs_theta_max": float(np.abs(thT).max()),
    }
    path = os.path.join(HERE, "results_phase2.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res["F4"] = summary
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    print("[F4]", {k: (round(v, 4) if isinstance(v, float) and v == v else v) for k, v in summary.items()})


if __name__ == "__main__":
    main()
