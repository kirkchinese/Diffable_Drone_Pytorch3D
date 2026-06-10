"""E3D-6a: 误差通道匹配 2×2 —— 2D R3/R8/R9 核心论点在 3D Go2 SRBD 上的复现。

注入两类已知失配（真实系统 = SRBD + 失配），各训两种残差头（2×2）：
  M_force（载荷比例切向力）× {R_force, R_kin}；M_kin（足端几何偏移）× {R_force, R_kin}。
判据 = 前向一步加速度 MSE + **梯度保真**：∂(a_lin,a_ang)/∂leg_ext 雅可比 vs 真实系统——
这正是闭环 BPTT 实际消费的策略梯度通道。预测：对角（匹配）前向+梯度都好；非对角即使
前向能拟合，雅可比也偏（"拟合前向但梯度错"）。

协议（避坑，沿用已验证做法）：
  · 状态采自**标称**系统扰动 settle 的早中期在位快照（失配 rollout 会发散，且回归只需
    一步加速度）；动作 leg_ext 独立随机采样（±0.05），与状态配对成数据集。
  · float64 + CPU（梯度保真实验，精度优先；规模小无需 GPU）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "dynamics"))
sys.path.insert(0, str(_HERE.parent / "models"))
from floating_base_srbd import FloatingBaseState  # noqa: E402
from srbd_standing import build_standing_config, standing_step  # noqa: E402
from residual_3d import (KAPPA, KIN_OFF, F_SCALE, K_SCALE, ResidualHead, accel,  # noqa: E402
                         index_state, mismatch, stack_states)

RESULTS = _HERE.parent / "results"


def collect_states(cfg, B=256, steps=140, snaps=(30, 60, 90, 120), seed=0):
    """标称扰动 settle，采早中期在位快照（含自然 v/w/暂态滑动）。"""
    rng = np.random.default_rng(seed)
    t = lambda a: torch.tensor(np.asarray(a), device=cfg.device, dtype=cfg.dtype)
    ang = t(np.c_[rng.uniform(-0.08, 0.08, B), rng.uniform(-0.08, 0.08, B), np.zeros(B)])
    q = matrix_to_quaternion(euler_angles_to_matrix(ang, "XYZ"))
    p = t(np.c_[np.zeros((B, 2)), cfg.rest_height + rng.uniform(-0.03, 0.03, B)])
    v = t(rng.uniform(-0.25, 0.25, (B, 3)))
    w = t(rng.uniform(-0.4, 0.4, (B, 3)))
    state = FloatingBaseState(p, q, v, w)
    zero_ext = torch.zeros(B, 4, device=cfg.device, dtype=cfg.dtype)
    pool = []
    for k in range(max(snaps) + 1):
        if k in snaps:
            pool.append(state.detach())
        state, _ = standing_step(state, zero_ext, cfg)
    data = stack_states(pool)
    N = data.p.shape[0]
    leg_ext = torch.tensor(rng.uniform(-0.05, 0.05, (N, 4)),
                           device=cfg.device, dtype=cfg.dtype)
    return data, leg_ext


def jac_wrt_action(state1, leg1, cfg, extra_fn):
    """单状态：∂accel(6)/∂leg_ext(4) → (6,4)。梯度穿过 extra_fn（残差头/失配）。"""
    le = leg1.clone().requires_grad_(True)
    fe, dx = extra_fn(state1, le)
    a = accel(state1, le, cfg, fe, dx)[0]
    return torch.stack([torch.autograd.grad(a[i], le, retain_graph=True)[0][0]
                        for i in range(6)])


def grad_fidelity(data, leg_ext, cfg, extra_fn, true_fn, n=16):
    err = 0.0
    for i in range(n):
        s1, l1 = index_state(data, i), leg_ext[i:i + 1]
        JT = jac_wrt_action(s1, l1, cfg, true_fn)
        JR = jac_wrt_action(s1, l1, cfg, extra_fn)
        err += ((JR - JT).norm() / (JT.norm() + 1e-9)).item()
    return err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)   # 须收敛充分：M_force 目标是接触
    # 状态的陡峭函数(k_n=1e4)，拟合值≠拟合导数，400 迭代时匹配力头梯度误差仅 0.26，
    # 2000 迭代到 0.13；M_kin 目标是常数(易)不受影响。
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    cfg = build_standing_config(device="cpu", dtype=torch.float64)
    data, leg_ext = collect_states(cfg, seed=args.seed)
    N = data.p.shape[0]
    print(f"E3D-6a 通道匹配 2×2  (N={N} 状态, float64/CPU)")
    print(f"  失配: M_force=−κ·f_n·x̂ (κ={KAPPA}) | M_kin=足偏移 {list(KIN_OFF)} m\n")

    rows = {}
    t0 = time.time()
    for mkind in ["force", "kin"]:
        true_fn = lambda s, l, mk=mkind: mismatch(mk, s, l, cfg)
        with torch.no_grad():
            aT = accel(data, leg_ext, cfg, *mismatch(mkind, data, leg_ext, cfg))
            aN = accel(data, leg_ext, cfg)                       # 标称（无残差）
        base_mse = ((aN - aT) ** 2).mean().item()
        base_g = grad_fidelity(data, leg_ext, cfg, lambda s, l: (None, None), true_fn)

        for rkind in ["force", "kin"]:
            head = ResidualHead(rkind, F_SCALE if rkind == "force" else K_SCALE).double()
            opt = torch.optim.Adam(head.parameters(), lr=args.lr)
            for _ in range(args.iters):
                fe, dx = head.extras(data, leg_ext, cfg)
                aP = accel(data, leg_ext, cfg, fe, dx)
                loss = ((aP - aT) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                fe, dx = head.extras(data, leg_ext, cfg)
                fit = ((accel(data, leg_ext, cfg, fe, dx) - aT) ** 2).mean().item()
            gerr = grad_fidelity(data, leg_ext, cfg,
                                 lambda s, l, h=head: h.extras(s, l, cfg), true_fn)
            tag = "✓匹配 " if mkind == rkind else " 错通道"
            rows[f"{mkind}|{rkind}"] = dict(fit=fit, base=base_mse, gerr=gerr,
                                            base_gerr=base_g)
            print(f"  M_{mkind:5s} × R_{rkind:5s} {tag}: 前向MSE {fit:.3e}"
                  f" (标称 {base_mse:.3e}) | ∂a/∂leg_ext 雅可比相对误差 {gerr:.3f}"
                  f" (标称 {base_g:.3f})")
        print()

    diag = np.mean([rows[k]["gerr"] for k in ("force|force", "kin|kin")])
    off = np.mean([rows[k]["gerr"] for k in ("force|kin", "kin|force")])
    verdict = off > diag * 2
    print(f"匹配通道平均梯度误差 {diag:.3f}  vs  错通道 {off:.3f}  "
          f"→ {'✅ 误差通道匹配在 3D SRBD 成立（错通道梯度明显更差）' if verdict else '⚠ 区分不明显'}"
          f"   [{time.time()-t0:.0f}s]")

    RESULTS.mkdir(exist_ok=True)
    out = dict(kappa=KAPPA, kin_off=list(KIN_OFF), iters=args.iters, seed=args.seed,
               rows=rows, diag_gerr=diag, off_gerr=off)
    (RESULTS / "e3d6_channel_matching.json").write_text(json.dumps(out, indent=2))
    print(f"结果已存 {RESULTS / 'e3d6_channel_matching.json'}")


if __name__ == "__main__":
    main()
