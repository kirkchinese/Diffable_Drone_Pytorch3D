"""E3D-6b: 双头残差 + 自动路由 —— 2D R10/R11 的 3D SRBD 复现。

E3D-6a 证明"残差必须放对通道"。本实验证明"两个头都给它，网络自己会放对"：
力头 + 运动学头同时挂、联合前向回归 + 正则，看修正自动流向正确通道。

避坑设计（预注册，逐条对应）：
  1) 单位不可通约(N vs m) → 正则统一折算**等效牛顿**：λ·(‖f‖² + ‖k_n·Δx‖²)
     （Δx 经接触刚度 k_n 换算成其能产生的法向力，两头同一物理汇率付费）。
  2) 权限不对称(kin 头经 k_n 放大) → "主导"判定用**加速度空间消融归因**
     C_i = E‖a(双头) − a(去头 i)‖，效果说话，不看输出范数。
  3) 退化簇(两头互相抵消的大输出对) → 正则杀零空间；λ 扫量级 + λ=0 对照，
     证明路由方向对 λ 稳健、非正则 artifact。
  4) 力头偷活(E3D-6a 它把 M_kin 前向拟好 19×) → 路由考题本身；要求双头 fit
     ≈ E3D-6a 匹配单头水平（力 ~78 / kin ~0.01 量级）才算收敛、路由才算数。
  5) 种子噪声 → 主 λ 下 3 seeds 验证方向稳定。
  6) 状态采自标称 settle + 一步加速度回归（失配 rollout 会发散）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "dynamics"))
sys.path.insert(0, str(_HERE.parent / "models"))
sys.path.insert(0, str(_HERE))
from srbd_standing import build_standing_config  # noqa: E402
from residual_3d import (KAPPA, KIN_OFF, DualHead, accel, mismatch)  # noqa: E402
from e3d6_channel_matching import collect_states, grad_fidelity  # noqa: E402

RESULTS = _HERE.parent / "results"
K_N_EQ = 1.0e4          # Δx→等效牛顿换算刚度（= ContactParams.k_n）
LAM_MAIN = 1e-4


def train_dual(data, leg_ext, cfg, aT, lam, seed, iters, lr=3e-3):
    torch.manual_seed(seed)
    dual = DualHead().double()
    opt = torch.optim.Adam(dual.parameters(), lr=lr)
    for _ in range(iters):
        fe, dx = dual.extras(data, leg_ext, cfg)
        fit = ((accel(data, leg_ext, cfg, fe, dx) - aT) ** 2).mean()
        reg = (fe ** 2).mean() + ((K_N_EQ * dx) ** 2).mean()
        loss = fit + lam * reg
        opt.zero_grad(); loss.backward(); opt.step()
    return dual


def eval_dual(data, leg_ext, cfg, aT, dual):
    """→ (fit, C_force, C_kin)。C_i = 去掉头 i 的加速度空间消融贡献（单位统一为加速度）。"""
    with torch.no_grad():
        fe, dx = dual.extras(data, leg_ext, cfg)
        a_full = accel(data, leg_ext, cfg, fe, dx)
        a_nof = accel(data, leg_ext, cfg, None, dx)
        a_nok = accel(data, leg_ext, cfg, fe, None)
        fit = ((a_full - aT) ** 2).mean().item()
        C_f = (a_full - a_nof).norm(dim=-1).mean().item()
        C_k = (a_full - a_nok).norm(dim=-1).mean().item()
    return fit, C_f, C_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    cfg = build_standing_config(device="cpu", dtype=torch.float64)
    data, leg_ext = collect_states(cfg, seed=args.seed)
    print(f"E3D-6b 双头自动路由  (N={data.p.shape[0]}, float64/CPU, iters={args.iters})")
    print(f"  失配: M_force κ={KAPPA} | M_kin {list(KIN_OFF)} m")
    print(f"  正则: λ·(‖f‖²+‖{K_N_EQ:.0f}·Δx‖²) 等效牛顿 | "
          f"E3D-6a 匹配单头 fit 参考: 力 ~78 / kin ~0.009\n")

    runs = []
    for mk in ["force", "kin"]:
        for lam in [0.0, 1e-5, 1e-4, 1e-3]:
            runs.append((mk, lam, 0))
        for seed in [1, 2]:
            runs.append((mk, LAM_MAIN, seed))

    targets = {mk: accel(data, leg_ext, cfg, *mismatch(mk, data, leg_ext, cfg)).detach()
               for mk in ["force", "kin"]}

    results, models = {}, {}
    t0 = time.time()
    for mk, lam, seed in runs:
        dual = train_dual(data, leg_ext, cfg, targets[mk], lam, seed, args.iters)
        fit, C_f, C_k = eval_dual(data, leg_ext, cfg, targets[mk], dual)
        rho = (C_f if mk == "force" else C_k) / (C_f + C_k + 1e-12)
        results[f"{mk}|{lam:g}|{seed}"] = dict(fit=fit, C_f=C_f, C_k=C_k, rho=rho)
        models[(mk, lam, seed)] = dual
        print(f"  M_{mk:5s} λ={lam:7.0e} seed{seed}: fit={fit:9.3e}  "
              f"C_force={C_f:8.3f} C_kin={C_k:8.3f}  ρ(正确头)={rho:.3f}")

    # 梯度保真：主配置双头 vs 真实系统（应达到 E3D-6a 匹配单头水平）
    print("\n梯度保真（主配置 λ=1e-4 seed0，∂a/∂leg_ext 雅可比相对误差）:")
    gerrs = {}
    for mk in ["force", "kin"]:
        dual = models[(mk, LAM_MAIN, 0)]
        ge = grad_fidelity(data, leg_ext, cfg,
                           lambda s, l, d=dual: d.extras(s, l, cfg),
                           lambda s, l, m=mk: mismatch(m, s, l, cfg))
        gerrs[mk] = ge
        print(f"  M_{mk:5s}: 双头梯度误差 = {ge:.3f}（E3D-6a 匹配单头参考: 力 0.133 / kin 0.001）")

    rhos = [(k, v["rho"]) for k, v in results.items() if not k.split("|")[1] == "0"]
    rho0 = {mk: results[f"{mk}|0|0"]["rho"] for mk in ["force", "kin"]}
    rmin = min(r for _, r in rhos)
    ok = rmin > 0.7
    print(f"\nλ=0 对照: M_force ρ={rho0['force']:.3f}  M_kin ρ={rho0['kin']:.3f}")
    print(f"判定: 全部 λ>0/seed 的 ρ(正确头) 最小值 = {rmin:.3f}  "
          f"→ {'✅ 自动路由在 3D SRBD 成立（修正稳定流向正确通道）' if ok else '⚠ 路由不稳，需排查'}"
          f"   [{time.time()-t0:.0f}s]")

    RESULTS.mkdir(exist_ok=True)
    out = dict(kappa=KAPPA, kin_off=list(KIN_OFF), iters=args.iters, k_n_eq=K_N_EQ,
               lam_main=LAM_MAIN, results=results, grad_err=gerrs, rho_min=rmin)
    (RESULTS / "e3d6_dualhead_routing.json").write_text(json.dumps(out, indent=2))
    print(f"结果已存 {RESULTS / 'e3d6_dualhead_routing.json'}")


if __name__ == "__main__":
    main()
