"""E3D-7 Stage 0: 失配敏感性审计（实验前置闸门）。

E3D-7 的论点是"残差修正孪生训出的策略迁回真实系统更好"。前提：**失配对任务有实质影响**
——否则三老师表现相同、实验空转（预注册坑#1）。本审计量化并可视化：被动 / 标称基线策略
在 标称 vs 真实(M_force) vs 真实(M_kin) 的长 rollout 差。

--task static  : 原 E3D-3 定高站立。**已审计结论（存档）**：被动下失配影响巨大
  （real_kin 被动倒地 tilt 41°），但反馈策略几乎完全掩盖（eval loss 仅 1.2×，残留为
  稳态小偏置：M_kin 高度偏置 +7mm、M_force 不可抗漂移 −3cm/s）→ 坑#2 应验，对比度不足。
--task dynamic : 对策后的正弦高度跟踪任务（见 e3d7_common 文档），闸门标准 loss 比 >1.5×。

附带回归检查：standing_step_hooked(hooks=None) ≡ standing_step 逐位一致（坑#6）。
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
from srbd_standing import build_standing_config, standing_step  # noqa: E402
from residual_3d import mismatch, standing_step_hooked  # noqa: E402
from e3d3_standing_train import sample_init  # noqa: E402
import e3d7_common as C  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d7_models"
SYSTEMS = ["nominal", "real_force", "real_kin"]


def extras_of(system: str, cfg):
    if system == "nominal":
        return lambda s, a: (None, None)
    return lambda s, a: mismatch(system.split("_")[1], s, a, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--task", choices=["static", "dynamic"], default="dynamic")
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    print(f"E3D-7 Stage 0 失配敏感性审计 (task={args.task}, {args.device}, "
          f"rest={cfg.rest_height:.4f})")

    # ---- 回归检查：hooked(None) ≡ standing_step ----
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    s0 = sample_init(cfg, 8, gen)
    a0 = (torch.rand(8, 4, device=cfg.device, dtype=cfg.dtype) - 0.5) * 0.1
    n1, _ = standing_step(s0, a0, cfg)
    n2, _ = standing_step_hooked(s0, a0, cfg)
    derr = max((getattr(n1, k) - getattr(n2, k)).abs().max().item() for k in "pqvw")
    print(f"  回归检查 hooked(None)≡standing_step: 最大差 {derr:.2e} "
          f"{'✅' if derr < 1e-6 else '❌'}")
    if args.task == "static":
        print("  static 任务的审计结果已存档于 results/e3d7_mismatch_audit.json（坑#2 证据）")
        return

    # ---- 动态任务：训标称基线 + 两个 oracle（闸门需要的正确对照）----
    # 闸门指标修正：baseline_real/baseline_nominal 把"任务难度变化"与"老师可分性"混为
    # 一谈（第一版审计 real_kin 反而 0.66× 变容易）。E3D-7 真正需要的潜在差距 =
    # **标称策略 vs oracle 策略，都在真实系统里评**。
    print(f"  训练标称基线 + oracle×2 (H={C.H_TRAIN}, iters={args.iters}, seed0)...")
    t0 = time.time()
    pol, hist = C.train(cfg, extras_of("nominal", cfg), iters=args.iters, seed=0)
    print(f"  baseline loss {hist['loss'][0]:.4f}->{hist['loss'][-1]:.4f} "
          f"梯度中位 {np.median(hist['gnorm']):.2e} [{time.time()-t0:.0f}s]")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), MODELS / "baseline_dyn_seed0.pt")
    oracles = {}
    for mk in ["force", "kin"]:
        oracles[mk], oh = C.train(cfg, extras_of(f"real_{mk}", cfg),
                                  iters=args.iters, seed=0)
        torch.save(oracles[mk].state_dict(), MODELS / f"oracle_dyn_{mk}_seed0.pt")
        print(f"  oracle M_{mk} loss {oh['loss'][0]:.4f}->{oh['loss'][-1]:.4f} "
              f"[{time.time()-t0:.0f}s]")

    # ---- 评测：被动/基线/oracle 在各系统 ----
    res = {}
    for who, p in [("passive", None), ("baseline", pol)]:
        for sysname in SYSTEMS:
            res[f"{who}|{sysname}"] = C.rollout_eval(p, cfg, extras_of(sysname, cfg))
            r = res[f"{who}|{sysname}"]
            print(f"  [{who:8s}|{sysname:10s}] loss={r['loss']:.4f} "
                  f"跟踪RMSE={r['track_rmse']*1e3:.1f}mm tilt={r['final_tilt']:.2f}° "
                  f"|x|末={r['final_x']:.3f}")
    for mk in ["force", "kin"]:
        res[f"oracle|real_{mk}"] = C.rollout_eval(oracles[mk], cfg,
                                                  extras_of(f"real_{mk}", cfg))
        r = res[f"oracle|real_{mk}"]
        print(f"  [oracle  |real_{mk:5s}] loss={r['loss']:.4f} "
              f"跟踪RMSE={r['track_rmse']*1e3:.1f}mm tilt={r['final_tilt']:.2f}°")

    gaps = {mk: res[f"baseline|real_{mk}"]["loss"] / res[f"oracle|real_{mk}"]["loss"]
            for mk in ["force", "kin"]}
    print(f"\n  闸门(潜在差距): L_real(标称策略)/L_real(oracle策略) = "
          f"force {gaps['force']:.2f}× | kin {gaps['kin']:.2f}×")
    ok = all(g > 1.3 for g in gaps.values())
    print(f"  → {'✅ 老师可分性存在，进入 Stage 1/2' if ok else '⚠ 标称策略已接近 oracle，闭环对比度不足——主结论将依赖策略梯度保真指标'}")

    # ---- 可视化 ----
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    t = np.arange(C.H_EVAL) * cfg.dt
    colors = dict(nominal="tab:gray", real_force="tab:red", real_kin="tab:blue")
    for row, who in enumerate(["passive", "baseline"]):
        for col, (key, ylab) in enumerate([("zs", "COM z (m)"), ("atts", "tilt (deg)"),
                                           ("xs", "COM x (m)")]):
            a = ax[row, col]
            for sysname in SYSTEMS:
                a.plot(t, res[f"{who}|{sysname}"][key], color=colors[sysname],
                       label=sysname, lw=1.2, alpha=0.9)
            if key == "zs":
                a.plot(t, res[f"{who}|nominal"]["zt"], "k:", lw=1, label="z*(t)")
            a.set_title(f"{who} | {ylab}"); a.set_xlabel("t (s)")
            if row == 0 and col == 0:
                a.legend(fontsize=8)
    fig.suptitle("E3D-7 Stage 0 (dynamic tracking): nominal vs real(M_force) vs real(M_kin)",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / "e3d7_mismatch_audit_dyn.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    summary = dict(task="dynamic", step_equiv_err=derr,
                   baseline_loss_final=hist["loss"][-1],
                   eval={k: dict(loss=v["loss"], track_rmse_mm=v["track_rmse"] * 1e3,
                                 tilt=v["final_tilt"], final_x=v["final_x"])
                         for k, v in res.items()},
                   gate_ratio_nom_over_oracle=gaps)
    (RESULTS / "e3d7_mismatch_audit_dyn.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {out}\n  saved {RESULTS / 'e3d7_mismatch_audit_dyn.json'}")


if __name__ == "__main__":
    main()
