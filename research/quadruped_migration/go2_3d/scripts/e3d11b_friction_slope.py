"""E3D-11b: 摩擦×坡度 MuJoCo 闸门——把摩擦逼到 binding〔实验〕

E3D-11 发现平地恒速下摩擦不是失败轴（牵引需求低，μ=0.3 也够推、低μ反而更好）。结论：
要让摩擦 binding 须**高牵引任务**。最干净的高牵引=**斜坡**：上坡时支撑摩擦既要推进、又要
扛住重力沿坡分量 mg·sinθ，可用切向 = μ·mg·cosθ → **打滑判据 μ < tanθ**（与几何无关的硬边界）。

做法（忠实、便宜）：MuJoCo 地板仍平=坡面对齐系，倾斜重力 g=(−g sinθ,0,−g cosθ)，+x 上坡
（rollout_mj 加 slope_deg 钩）。扫 θ×μ，现成 nominal vs dynamics-DR 评测（kp=300，平地训的
策略上坡=真实 sim2real 问题）。看 vx(还爬不爬得动)/摔 在哪崩，是否沿 μ≈tanθ 边界。
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
MUS = [0.3, 0.6, 1.0]
THETAS = [0, 8, 16, 24, 30]          # 坡度 deg；tanθ = 0, .14, .29, .45, .58
HOLD_VX = 0.10                        # vx>此=还在爬（未失速/打滑）
MU_COL = {0.3: "tab:red", 0.6: "tab:orange", 1.0: "tab:green"}


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
    print(f"E3D-11b 摩擦×坡度 MuJoCo 闸门 (kp=300); θ={THETAS}deg μ={MUS}; 打滑判据 μ<tanθ")
    arms = ["nominal", "DR"]
    summ = {a: {} for a in arms}
    t0 = time.time()
    for arm in arms:
        for mu in MUS:
            for th in THETAS:
                vxs, falls = [], 0
                for seed in range(args.seeds):
                    pol = _load(arm, seed, cfg)
                    r = rollout_mj(pol, cfg, g, z_ref, seconds=4.0, kp=300.0, kd=5.0,
                                   ground_mu=mu, slope_deg=th)
                    vxs.append(r["vx_mean"]); falls += r["fell_at"] is not None
                summ[arm][f"{mu}_{th}"] = dict(vx=float(np.mean(vxs)), falls=falls)
            row = " ".join(f"θ{th}:{summ[arm][f'{mu}_{th}']['vx']:+.2f}"
                           f"{'✗' if summ[arm][f'{mu}_{th}']['falls'] else ''}" for th in THETAS)
            print(f"  {arm:7s} μ={mu}: {row} [{time.time()-t0:.0f}s]")

    # 判定：打滑随 μ≈tanθ 边界出现？低μ在更缓的坡就崩(=摩擦 binding)？
    def onset(arm, mu):
        for th in THETAS:
            s = summ[arm][f"{mu}_{th}"]
            if s["vx"] < HOLD_VX or s["falls"] >= 2:
                return th
        return None  # 全程 hold
    ons = {mu: onset("nominal", mu) for mu in MUS}
    pred = {mu: np.degrees(np.arctan(mu)) for mu in MUS}
    binding = (ons[0.3] is not None) and (ons[1.0] is None or ons[1.0] > (ons[0.3] or 99))
    verdict = (f"摩擦在斜坡上成 binding(对照 E3D-11 平地非失败轴): "
               f"上坡失速/打滑起始坡度 nominal μ0.3={ons[0.3]}° / μ0.6={ons[0.6]}° / μ1.0={ons[1.0]}°"
               f"(理论 μ<tanθ 打滑: μ0.3→{pred[0.3]:.0f}° μ0.6→{pred[0.6]:.0f}° μ1.0→{pred[1.0]:.0f}°). "
               f"{'**低μ在更缓坡就崩、高μ撑更陡——摩擦确成 binding**' if binding else '边界不清,查策略上坡能力'}. "
               "印证 E3D-11 的方法学: 平地测不出摩擦,坡度把牵引逼到 μ·f_n 上限才让 μ 见真章。"
               "注:平地训的策略未学上坡,高θ崩含'策略不会爬坡'成分,但**同θ跨μ的差异隔离出摩擦轴**。")
    summ["verdict"] = verdict
    summ["mus"] = MUS; summ["thetas"] = THETAS
    print(f"  → {verdict}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3d11b_friction_slope.json").write_text(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    for mu in MUS:
        a.plot(THETAS, [summ["nominal"][f"{mu}_{th}"]["vx"] for th in THETAS], "o-",
                color=MU_COL[mu], label=f"μ={mu}")
        a.axvline(np.degrees(np.arctan(mu)), color=MU_COL[mu], ls=":", lw=1)
    a.axhline(HOLD_VX, color="k", ls="--", lw=0.8, label="hold 阈")
    a.axhline(g.vx_cmd, color="gray", ls=":", lw=0.8)
    a.set_xlabel("坡度 θ (deg)"); a.set_ylabel("上坡 vx (m/s)")
    a.set_title("上坡速度 vs 坡度(nominal)\n虚线=μ<tanθ 理论打滑起始"); a.legend(fontsize=8)
    a = ax[1]
    for mu in MUS:
        a.plot(THETAS, [summ["nominal"][f"{mu}_{th}"]["falls"] for th in THETAS], "s-",
                color=MU_COL[mu], label=f"μ={mu} nominal")
        a.plot(THETAS, [summ["DR"][f"{mu}_{th}"]["falls"] for th in THETAS], "^--",
                color=MU_COL[mu], alpha=0.5)
    a.set_xlabel("坡度 θ (deg)"); a.set_ylabel(f"摔倒数 /{args.seeds}")
    a.set_title("摔倒 vs 坡度(实线nominal/虚线DR)"); a.legend(fontsize=7)
    fig.suptitle("E3D-11b 摩擦×坡度 MuJoCo 闸门：把摩擦逼到 binding(对照 E3D-11 平地非失败轴)\n"
                 "上坡牵引须扛 mg·sinθ → 低μ在更缓坡就失速/打滑(μ≈tanθ 边界)，摩擦终见真章",
                 fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "e3d11b_friction_slope.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d11b_friction_slope.png'}\nsaved {RESULTS / 'e3d11b_friction_slope.json'}")


if __name__ == "__main__":
    main()
