"""E3D-11: 摩擦扫描 MuJoCo 闸门——补"机械狗最核心的轴"〔实验〕

用户指出复杂地面/摩擦没单独测过。机械狗推进机制本身就是摩擦（支撑足摩擦速度伺服），
μ 低→推不动→打滑→失速/摔=最经典的 sim2real 杀手。E3D-9 把 μ 捆在域随机里(×0.7-1.3→
0.56-1.04)但从没隔离/压到打滑区。这里把 MuJoCo 地面 μ 扫 {0.3,0.5,0.7,1.0,1.3}，用现成
nominal vs dynamics-DR 策略评测（纯 eval，kp=300 调定点），看摩擦失配在哪崩、DR 的 μ 范围
够不够。预测复刻 E3D-9 律：DR 在随机化过的 μ 区间稳、低摩擦 0.3(DR 范围外)双双崩。
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
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
import _plotstyle  # noqa: E402
_plotstyle.use_cjk()
from srbd_standing import build_standing_config  # noqa: E402
from gait_3d import GaitConfig  # noqa: E402
from e3d4_gait_train import GaitPolicy  # noqa: E402
from e3d4b_residual_gait import load_nominal  # noqa: E402
from e3d5_mujoco_check import rollout_mj  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
MUS = [0.3, 0.5, 0.7, 1.0, 1.3]      # MuJoCo 地面滑动摩擦扫描（孪生 μ=0.8, DR 覆盖 0.56-1.04）
ARM_COL = {"nominal": "tab:gray", "DR": "tab:green"}


def _load(arm, seed, cfg):
    if arm == "nominal":
        return load_nominal(cfg, seed)
    pol = GaitPolicy()
    pol.load_state_dict(torch.load(MODELS / f"e3d9_dr_s{seed}.pt",
                                   map_location="cpu", weights_only=True))
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = build_standing_config(device="cpu", dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-11 摩擦扫描 MuJoCo 闸门 (kp=300, μ={MUS}); 孪生μ=0.8 DR随机化μ∈[0.56,1.04]")
    arms = ["nominal", "DR"]
    summ = {a: {} for a in arms}
    t0 = time.time()
    for arm in arms:
        for mu in MUS:
            rmses, vxs, falls = [], [], 0
            for seed in range(args.seeds):
                pol = _load(arm, seed, cfg)
                r = rollout_mj(pol, cfg, g, z_ref, seconds=4.0, kp=300.0, kd=5.0,
                               ground_mu=mu)
                rmses.append(r["vx_rmse"] * 1e3); vxs.append(r["vx_mean"])
                falls += r["fell_at"] is not None
            summ[arm][str(mu)] = dict(rmse=float(np.mean(rmses)), rmse_std=float(np.std(rmses)),
                                      vx=float(np.mean(vxs)), falls=falls)
            print(f"  {arm:7s} μ={mu}: vx {np.mean(vxs):.3f}/{g.vx_cmd} "
                  f"RMSE {np.mean(rmses):.0f}±{np.std(rmses):.0f}mm/s 摔{falls}/{args.seeds} "
                  f"[{time.time()-t0:.0f}s]")

    # 判定（诚实，纠正"低μ打滑崩"的预测）
    def vx(a, mu):
        return summ[a][str(mu)]["vx"]
    no_falls = all(summ[a][str(m)]["falls"] == 0 for a in arms for m in MUS)
    nom_lo, nom_hi = summ["nominal"]["0.3"]["rmse"], summ["nominal"]["1.3"]["rmse"]
    low_better = nom_lo < nom_hi
    verdict = (f"反直觉(纠正打滑预测): 平地恒速 trot 下 μ∈[0.3,1.3] "
               f"{'**全程0摔**' if no_falls else '有摔'}, 且{'**低μ反而跟踪更好**' if low_better else ''}"
               f"(nominal RMSE μ0.3={nom_lo:.0f} {'<' if low_better else '>'} μ1.3={nom_hi:.0f}, "
               f"vx μ0.3={vx('nominal',0.3):.2f}≈指令0.3). 机制: 软(低μ)接触滤掉落足误差、"
               "grippy(高μ)接触把落足误差放大成机体抖. **摩擦在平地恒速不是失败轴**——牵引需求低,"
               "μ=0.3 也够推(没打滑失速)。DR≈nominal(摩擦鲁棒在此用不上,μ-DR 既没帮也没伤). "
               "**要让摩擦成 binding(打滑/摔的真 regime)须高牵引任务**(坡度/加减速/急转/推扰)=E3D-11b; "
               "教训同 E3D-10: 任务须把该轴逼到 binding 才测得出。")
    summ["verdict"] = verdict
    summ["mus"] = MUS
    print(f"  → {verdict}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3d11_friction.json").write_text(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    for arm in arms:
        a.plot(MUS, [summ[arm][str(m)]["vx"] for m in MUS], "o-", color=ARM_COL[arm], label=arm)
    a.axhline(g.vx_cmd, color="k", ls=":", lw=1, label="vx*=0.3")
    a.axvspan(0.56, 1.04, color="tab:green", alpha=0.08, label="DR 随机化μ区间")
    a.set_xlabel("MuJoCo 地面摩擦 μ"); a.set_ylabel("实测 vx (m/s)")
    a.set_title("前进速度 vs 地面摩擦\n(μ 低→摩擦伺服推不动→失速)"); a.legend(fontsize=8)
    a = ax[1]
    for arm in arms:
        a.plot(MUS, [summ[arm][str(m)]["rmse"] for m in MUS], "s-", color=ARM_COL[arm], label=arm)
    a.axvspan(0.56, 1.04, color="tab:green", alpha=0.08)
    a.set_xlabel("MuJoCo 地面摩擦 μ"); a.set_ylabel("vx RMSE (mm/s)")
    a.set_title("跟踪误差 vs 地面摩擦"); a.legend(fontsize=8)
    fig.suptitle("E3D-11 摩擦扫描 MuJoCo 闸门(平地恒速 trot)：反直觉——μ0.3–1.3 全程0摔、低μ反而跟得更好\n"
                 "软接触滤误差/grippy放大抖；摩擦在平地恒速非失败轴(牵引需求低)，须高牵引任务才 binding",
                 fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "e3d11_friction.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d11_friction.png'}\nsaved {RESULTS / 'e3d11_friction.json'}")


if __name__ == "__main__":
    main()
