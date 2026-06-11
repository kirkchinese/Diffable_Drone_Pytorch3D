"""E3D-7 Stage 1: 在真实系统闭环数据上拟合双头残差（sim2real 管线版）。

预注册坑#3（分布外失效）的对策：E3D-6b 的残差是在 settle 快照 + leg_ext∈±0.05 上回归的，
而策略动作幅度 ±0.10——闭环会把残差推到训练分布外。本阶段改为现实中的做法：
**拿标称基线策略（动态跟踪任务）在"真实系统"里滚轨迹（加探索噪声），在这个闭环分布上
重训双头**。数据 = 你真能从真机拿到的东西；残差训练域 = 残差将被使用的域。

判读：fit（含 20% held-out，防 model exploitation）、路由 ρ（加速度空间消融归因，应与
E3D-6b 方向一致）、∂a/∂leg_ext 梯度保真、覆盖/拟合可视化。残差训练 float64/CPU（梯度
质量），使用时 cast 回 float32/GPU。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _plotstyle  # noqa: E402
_plotstyle.use_cjk()
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from floating_base_srbd import FloatingBaseState  # noqa: E402
from srbd_standing import build_standing_config, level_error  # noqa: E402
from residual_3d import DualHead, accel, mismatch, standing_step_hooked  # noqa: E402
from e3d6_channel_matching import grad_fidelity  # noqa: E402
from e3d3_standing_train import sample_init  # noqa: E402
import e3d7_common as C  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d7_models"
K_N_EQ = 1.0e4
LAM = 1e-4


def collect_real_rollouts(cfg, policy, mkind, horizon=C.H_EVAL, B=64, every=5,
                          noise=0.01, seed=3):
    """基线策略 + 探索噪声在真实系统滚动态跟踪轨迹，采 (state, action) 对。"""
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, B, gen)
    S, A = [], []
    with torch.no_grad():
        for t in range(horizon):
            zs, phi = C.z_star_phase(t, cfg)
            a = policy(C.observe_dyn(s, zs, phi))
            a = (a + noise * torch.randn(a.shape, generator=gen, device=cfg.device,
                                         dtype=cfg.dtype)).clamp(-0.10, 0.10)
            if t >= 10 and t % every == 0:
                S.append(s.detach()); A.append(a.detach())
            fe, dx = mismatch(mkind, s, a, cfg)
            s, _ = standing_step_hooked(s, a, cfg, fe, dx)
    state = FloatingBaseState(*[torch.cat([getattr(x, k) for x in S], 0) for k in "pqvw"])
    return state, torch.cat(A, 0)


def to64(state, leg, n, seed=0):
    """GPU float32 采样 → CPU float64 训练张量（随机打乱取前 n）。"""
    idx = torch.randperm(state.p.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    f = lambda x: x[idx].detach().double().cpu()
    return FloatingBaseState(f(state.p), f(state.q), f(state.v), f(state.w)), \
        leg[idx].detach().double().cpu()


def index_slice(state, leg, a, b):
    f = lambda x: x[a:b]
    return FloatingBaseState(f(state.p), f(state.q), f(state.v), f(state.w)), leg[a:b]


def train_dual(data, leg, cfg64, aT, iters, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    dual = DualHead().double()
    opt = torch.optim.Adam(dual.parameters(), lr=lr)
    for _ in range(iters):
        fe, dx = dual.extras(data, leg, cfg64)
        fit = ((accel(data, leg, cfg64, fe, dx) - aT) ** 2).mean()
        reg = (fe ** 2).mean() + ((K_N_EQ * dx) ** 2).mean()
        opt.zero_grad(); (fit + LAM * reg).backward(); opt.step()
    return dual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--replot", action="store_true",
                    help="载入已存残差只重绘图（不重训、不覆盖 .pt）")
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)

    pol = C.PolicyDyn().to(cfg.device, cfg.dtype)
    pol.load_state_dict(torch.load(MODELS / "baseline_dyn_seed0.pt",
                                   map_location=cfg.device))
    print("E3D-7 Stage 1: 真实系统闭环数据（动态跟踪基线+噪声）→ 双头残差拟合")

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    summary = {}
    for row, mkind in enumerate(["force", "kin"]):
        t0 = time.time()
        S, A = collect_real_rollouts(cfg, pol, mkind)
        data, leg = to64(S, A, n=args.n_train + args.n_train // 4)
        ntr = args.n_train
        dtr, ltr = index_slice(data, leg, 0, ntr)
        dho, lho = index_slice(data, leg, ntr, data.p.shape[0])
        with torch.no_grad():
            aT_tr = accel(dtr, ltr, cfg64, *mismatch(mkind, dtr, ltr, cfg64))
            aT_ho = accel(dho, lho, cfg64, *mismatch(mkind, dho, lho, cfg64))
            base_tr = ((accel(dtr, ltr, cfg64) - aT_tr) ** 2).mean().item()
        if args.replot:
            dual = DualHead().double()
            dual.load_state_dict(torch.load(MODELS / f"residual_{mkind}.pt",
                                            map_location="cpu", weights_only=True))
        else:
            dual = train_dual(dtr, ltr, cfg64, aT_tr, args.iters)
        with torch.no_grad():
            fe, dx = dual.extras(dtr, ltr, cfg64)
            aP_tr = accel(dtr, ltr, cfg64, fe, dx)
            fit_tr = ((aP_tr - aT_tr) ** 2).mean().item()
            feh, dxh = dual.extras(dho, lho, cfg64)
            fit_ho = ((accel(dho, lho, cfg64, feh, dxh) - aT_ho) ** 2).mean().item()
            a_nof = accel(dtr, ltr, cfg64, None, dx)
            a_nok = accel(dtr, ltr, cfg64, fe, None)
            C_f = (aP_tr - a_nof).norm(dim=-1).mean().item()
            C_k = (aP_tr - a_nok).norm(dim=-1).mean().item()
        rho = (C_f if mkind == "force" else C_k) / (C_f + C_k + 1e-12)
        gerr = grad_fidelity(dtr, ltr, cfg64,
                             lambda s, l, d=dual: d.extras(s, l, cfg64),
                             lambda s, l, m=mkind: mismatch(m, s, l, cfg64))
        print(f"  [M_{mkind:5s}] N={ntr}  fit train={fit_tr:.3e} held-out={fit_ho:.3e} "
              f"(标称 {base_tr:.3e})  ρ(正确头)={rho:.3f}  ∂a/∂leg_ext 误差={gerr:.3f} "
              f"[{time.time()-t0:.0f}s]")
        if not args.replot:
            torch.save(dual.state_dict(), MODELS / f"residual_{mkind}.pt")
        summary[mkind] = dict(fit_train=fit_tr, fit_holdout=fit_ho, base=base_tr,
                              C_f=C_f, C_k=C_k, rho=rho, grad_err=gerr)

        # ---- 可视化：覆盖 + 拟合 + 归因 ----
        a = ax[row, 0]
        a.hist(ltr.numpy().ravel(), bins=40, color="tab:green", alpha=0.8)
        a.axvline(-0.05, color="r", ls="--", lw=1)
        a.axvline(0.05, color="r", ls="--", lw=1, label="E3D-6 旧范围 ±0.05")
        a.set_title(f"M_{mkind}: leg_ext 覆盖 (闭环)"); a.legend(fontsize=7)
        a = ax[row, 1]
        tilt = np.degrees(np.arccos(np.clip(1 - level_error(dtr.q).numpy(), -1, 1)))
        a.hist2d(dtr.p[:, 2].numpy(), tilt, bins=40, cmap="viridis")
        a.set_xlabel("z (m)"); a.set_ylabel("tilt (deg)"); a.set_title("状态覆盖 z×tilt")
        a = ax[row, 2]
        ii = np.random.default_rng(0).choice(aT_tr.shape[0], 800, replace=False)
        a.scatter(aT_tr.numpy()[ii].ravel(), aP_tr.numpy()[ii].ravel(), s=2, alpha=0.3)
        lo, hi = np.percentile(aT_tr.numpy(), [1, 99])
        a.plot([lo, hi], [lo, hi], "r-", lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("real accel"); a.set_ylabel("corrected-twin accel")
        a.set_title(f"拟合 (held-out MSE {fit_ho:.2e})")
        a = ax[row, 3]
        a.bar(["C_force", "C_kin"], [C_f, C_k],
              color=["tab:red" if mkind == "force" else "tab:gray",
                     "tab:blue" if mkind == "kin" else "tab:gray"])
        a.set_title(f"消融归因 ρ(正确头)={rho:.3f}")
    fig.suptitle("E3D-7 Stage 1: dual-head residual fitted on REAL-system closed-loop data",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / "e3d7_fit_residual.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    (RESULTS / "e3d7_fit_residual.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {out}\n  saved {RESULTS / 'e3d7_fit_residual.json'}")


if __name__ == "__main__":
    main()
