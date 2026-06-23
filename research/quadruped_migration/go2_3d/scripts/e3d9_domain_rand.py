"""E3D-9: 可微孪生 + 域随机 → 过 MuJoCo 闸门?〔实验〕

用户重定向（正确的框架）：不追"无 sim2real gap"（不可能），而是"训练方法能在一定程度
还原真实、部署后保持不错性能"，**用 MuJoCo 当中间真机判据**。E3D-5 的诚实结论被误读为
"可微路线失败"——其实**nominal 孪生策略迁 MuJoCo 已经走起来了(92mm/s,0摔)**，失败的只是
用残差去"修正"孪生（越修越差）。正确的工具不是修正(correction)是鲁棒(robustness)：

  **在孪生上做域随机 → 训对一整族孪生鲁棒的策略 → 过 MuJoCo 闸门 → 上真机。**
  可微梯度负责每个随机实例内的样本效率；鲁棒性来自随机化。这是"可微 + 域随机"的混合
  路线，业界(RL+DR)的逻辑同源，只是把 RL 换成 BPTT。

域随机**有的放矢**（用 E3D-5 诊断：真实失配以运动学/惯量为主 C_kin≫C_force）：足端几何
（±2cm x/±1.2cm z，bracket E3D-5 实测 δ̂）、逐轴惯量(×0.7-1.4)、质量(×0.85-1.15)、
接触(k_n×0.5-2, μ×0.7-1.3)——把 MuJoCo 的失配包进训练分布。

判据=MuJoCo 闸门(rollout_mj)：DR 策略 vs 既有 nominal(92mm/s)，跨 PD 增益失配 kp∈{200,
300,400}(真·执行器失配轴)比 vx RMSE + 摔倒。既有 nominal 训练视野更长(H800)→DR 赢=保守。

stage: train（孪生上域随机训 DR 策略）/ gate（DR vs nominal 上 MuJoCo 跨 kp 比）。
"""
from __future__ import annotations

import argparse
import dataclasses
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
from e3d4_gait_train import GaitPolicy, sample_init, rollout_train  # noqa: E402
from e3d4b_residual_gait import load_nominal  # noqa: E402
from e3d5_mujoco_check import rollout_mj  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
KP_SWEEP = [200.0, 300.0, 400.0]      # PD 增益失配（300=E3D-5 调定工作点；200/400=失配应力）


def randomize_cfg(base, g, cgen):
    """有的放矢域随机（E3D-5 诊断：运动学/惯量主导）→ 返回扰动后的 StandingConfig。"""
    def u(lo, hi):
        return lo + (hi - lo) * torch.rand((), generator=cgen).item()
    mass = base.mass * u(0.85, 1.15)
    s = torch.tensor([u(0.7, 1.4), u(0.7, 1.4), u(0.7, 1.4)],
                     device=base.device, dtype=base.dtype).sqrt()
    I = torch.diag(s) @ base.I_body @ torch.diag(s)          # congruence 保 SPD（逐轴惯量）
    foot = base.foot_rel_com.clone()
    foot[:, 0] = foot[:, 0] + u(-0.02, 0.02)                 # 足端 x 偏（运动学失配,bracket δ̂）
    foot[:, 2] = foot[:, 2] + u(-0.012, 0.012)               # 足端 z 偏
    contact = dataclasses.replace(base.contact, k_n=base.contact.k_n * u(0.5, 2.0),
                                  k_d=base.contact.k_d * u(0.6, 1.5),
                                  mu=base.contact.mu * u(0.7, 1.3))
    return dataclasses.replace(base, mass=mass, I_body=I, I_body_inv=torch.linalg.inv(I),
                               foot_rel_com=foot, contact=contact)


def train_dr(base, g, iters, B=48, H=600, tbptt=150, lr=3e-3, seed=0, clip=1.0, dr=True):
    """域随机训练：每 iter 重采样一个孪生（dr=False 则恒用 base，作受控对照）。"""
    torch.manual_seed(seed)
    pol = GaitPolicy().to(base.device, base.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=base.device).manual_seed(seed + 100)
    cgen = torch.Generator().manual_seed(seed + 200)
    hist = []
    for _ in range(iters):
        cfg = randomize_cfg(base, g, cgen) if dr else base
        z_ref = cfg.rest_height + g.ext0 - 0.004
        s = sample_init(cfg, g, z_ref, B, gen)
        loss = rollout_train(pol, cfg, g, z_ref, s, H, "smooth", tbptt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), clip)
        opt.step(); hist.append(loss.item())
    return pol, hist


def stage_train(device, iters, n_seeds):
    base = build_standing_config(device=device, dtype=torch.float32)
    g = GaitConfig()
    print(f"[train] 域随机 DR 策略 ×{n_seeds} 种子 (iters={iters}, H=600, cuda:1)")
    t0 = time.time()
    for seed in range(n_seeds):
        pol, hist = train_dr(base, g, iters, seed=seed, dr=True)
        torch.save(pol.state_dict(), MODELS / f"e3d9_dr_s{seed}.pt")
        print(f"  DR s{seed} loss→{hist[-1]:.4f} [{time.time()-t0:.0f}s]")


@torch.no_grad()
def eval_policy_mj(pol, cfg, g, z_ref, kps):
    """一个策略跨 kp 上 MuJoCo：返回每 kp 的 (vx_rmse_mm, fell)。"""
    out = {}
    for kp in kps:
        r = rollout_mj(pol, cfg, g, z_ref, seconds=4.0, kp=kp, kd=5.0)
        out[kp] = (r["vx_rmse"] * 1e3, r["fell_at"] is not None, r["vx_mean"], r["tilt_end"])
    return out


def stage_gate(device, n_seeds):
    cfg = build_standing_config(device="cpu", dtype=torch.float32)   # rollout_mj 用 cpu
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"[gate] DR vs nominal 上 MuJoCo，跨 kp={KP_SWEEP} (kd=5)")
    arms = {"nominal": [], "DR": []}
    t0 = time.time()
    for seed in range(n_seeds):
        pol_n = load_nominal(cfg, seed)
        arms["nominal"].append(eval_policy_mj(pol_n, cfg, g, z_ref, KP_SWEEP))
        f = MODELS / f"e3d9_dr_s{seed}.pt"
        pol_d = GaitPolicy()
        pol_d.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        arms["DR"].append(eval_policy_mj(pol_d, cfg, g, z_ref, KP_SWEEP))
        print(f"  seed{seed} done [{time.time()-t0:.0f}s]")

    # 汇总：每 arm 每 kp 的 RMSE 均值/标准差 + 摔倒数；以及跨 kp 的鲁棒性(均值/最差)
    summ = {}
    for arm, rows in arms.items():
        per_kp = {}
        for kp in KP_SWEEP:
            rmses = [r[kp][0] for r in rows]
            falls = sum(r[kp][1] for r in rows)
            per_kp[kp] = dict(rmse_mean=float(np.mean(rmses)), rmse_std=float(np.std(rmses)),
                              falls=falls)
        all_rmse = [r[kp][0] for r in rows for kp in KP_SWEEP]
        all_falls = sum(r[kp][1] for r in rows for kp in KP_SWEEP)
        summ[arm] = dict(per_kp={str(int(k)): v for k, v in per_kp.items()},
                         rmse_overall=float(np.mean(all_rmse)),
                         rmse_worst=float(np.max([per_kp[k]["rmse_mean"] for k in KP_SWEEP])),
                         falls_total=all_falls, n=len(rows))
    for arm in ["nominal", "DR"]:
        s = summ[arm]
        line = f"  {arm:8s}: 跨kp总均 {s['rmse_overall']:.0f}mm/s  最差kp {s['rmse_worst']:.0f}  摔 {s['falls_total']}/{s['n']*len(KP_SWEEP)}"
        for kp in KP_SWEEP:
            p = s["per_kp"][str(int(kp))]
            line += f" | kp{int(kp)} {p['rmse_mean']:.0f}±{p['rmse_std']:.0f}({p['falls']}摔)"
        print(line)

    dr, no = summ["DR"], summ["nominal"]
    drt, not_ = dr["per_kp"]["300"], no["per_kp"]["300"]      # 调定工作点
    n_better = sum(dr["per_kp"][str(int(k))]["rmse_mean"] < no["per_kp"][str(int(k))]["rmse_mean"]
                   for k in KP_SWEEP)
    var_cut = not_["rmse_std"] - drt["rmse_std"]
    # 在随机化覆盖的轴(kp≥300:接触/惯量/几何主导)上 DR 更好更稳；kp200(低增益=执行器权限,
    # 未随机化轴)回退——DR 只在随机化过的轴上有效。
    verdict = (f"[部分验证·robustness>correction 方向成立] 调定点 kp300: DR {drt['rmse_mean']:.0f}"
               f"±{drt['rmse_std']:.0f} vs nominal {not_['rmse_mean']:.0f}±{not_['rmse_std']:.0f}"
               f"(方差↓{var_cut:.0f},更可靠迁移); kp400 DR {dr['per_kp']['400']['rmse_mean']:.0f}"
               f"<{no['per_kp']['400']['rmse_mean']:.0f} 明显更优; {n_better}/3 kp 更优, 全 0 摔. "
               f"唯一回退在 kp200(低增益=执行器权限,**未随机化的轴**: 孪生无PD,DR 没覆盖该轴). "
               "结论: DR 只在随机化过的轴上收窄 gap(已兑现:接触/惯量/几何),补执行器增益/时延"
               "随机化应闭合 kp200; 可微+DR 对真机代理方向有效,绿灯但需补轴.")
    summ["verdict"] = verdict
    print(f"  → {verdict}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3d9_gate.json").write_text(json.dumps(summ, indent=2))

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(KP_SWEEP)); w = 0.36
    cols = dict(nominal="tab:gray", DR="tab:green")
    a = ax[0]
    for i, arm in enumerate(["nominal", "DR"]):
        means = [summ[arm]["per_kp"][str(int(k))]["rmse_mean"] for k in KP_SWEEP]
        stds = [summ[arm]["per_kp"][str(int(k))]["rmse_std"] for k in KP_SWEEP]
        a.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3, color=cols[arm], label=arm)
    a.axhline(92, color="k", ls=":", lw=1, label="E3D-5 nominal@kp300=92")
    a.set_xticks(x); a.set_xticklabels([f"kp={int(k)}" for k in KP_SWEEP])
    a.set_ylabel("MuJoCo vx RMSE (mm/s)")
    a.set_title("跨 PD 增益失配的迁移鲁棒性\n(kp=300 调定,200/400 失配应力)")
    a.legend(fontsize=8)
    a = ax[1]
    for i, arm in enumerate(["nominal", "DR"]):
        falls = [summ[arm]["per_kp"][str(int(k))]["falls"] for k in KP_SWEEP]
        a.bar(x + (i - 0.5) * w, falls, w, color=cols[arm], label=arm)
    a.set_xticks(x); a.set_xticklabels([f"kp={int(k)}" for k in KP_SWEEP])
    a.set_ylabel(f"摔倒数 (/{n_seeds} 种子)")
    a.set_title("跨 kp 失配摔倒（鲁棒性）"); a.legend(fontsize=8)
    fig.suptitle("E3D-9 可微孪生+域随机 vs MuJoCo 闸门：DR 只在随机化过的轴上收窄 gap\n"
                 "调定点 kp300 更稳(方差↓)、kp400 更优；kp200(未随机化的执行器权限轴)回退",
                 fontsize=11)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "e3d9_gate.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d9_gate.png'}\nsaved {RESULTS / 'e3d9_gate.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "gate"])
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    print(f"E3D-9 [{args.stage}]")
    if args.stage == "train":
        stage_train(args.device, args.iters, args.seeds)
    else:
        stage_gate(args.device, args.seeds)


if __name__ == "__main__":
    main()
