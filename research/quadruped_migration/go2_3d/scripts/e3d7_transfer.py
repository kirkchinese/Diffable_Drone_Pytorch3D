"""E3D-7 Stage 2-3: 三师迁移实验——梯度保真的最终兑现（动态高度跟踪任务）。

三个"老师"（可微训练环境）：
  nominal   = 标称孪生（带模型误差；与失配无关，三 seeds 跨两失配共享）
  corrected = 标称孪生 + Stage 1 在真实闭环数据上拟好的**冻结**双头残差
  oracle    = 真实系统本身（上界；现实中不可得，仅作参照）
同款策略/超参/seeds（PolicyDyn, smooth, noGDecay）各自训练，**全部迁回真实系统评测**
（固定 eval 初态批、4 周期长 rollout）。

判据：gap closure = (L_nom − L_corr)/(L_nom − L_oracle) → 1 为完美修正。
预注册坑：#2 反馈掩盖 → 动态任务 + 3 seeds + 前馈签名（前后腿伸长差）；
#4 残差进回路的梯度 → 训前各老师梯度自检 + 训练梯度曲线同图监控。
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
from srbd_standing import build_standing_config  # noqa: E402
from residual_3d import DualHead, mismatch  # noqa: E402
from e3d3_standing_train import sample_init  # noqa: E402
import e3d7_common as C  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d7_models"
SEEDS = [0, 1, 2]
TEACHERS = ["nominal", "corrected", "oracle"]


def make_extra_fn(teacher: str, mkind: str, cfg, residuals):
    if teacher == "nominal":
        return lambda s, a: (None, None)
    if teacher == "oracle":
        return lambda s, a: mismatch(mkind, s, a, cfg)
    dual = residuals[mkind]
    return lambda s, a: dual.extras(s, a, cfg)


"""注：策略梯度保真主指标已移至专用脚本 e3d7_grad_fidelity.py（全局拼接聚合）。
本脚本只做闭环迁移确认，默认仅 M_kin——M_force 含不可控漂移模式（垂直腿无法产生持续
水平力），oracle 训练发散（Stage 0 实测 loss→6.4、倒至 111°），闭环对比无意义，如实记录。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--tbptt", type=int, default=150,
                    help="截断BPTT窗口(步); E3D-7视野扫描:梯度只在H≲150可用; 0=全程")
    ap.add_argument("--mismatches", default="kin",
                    help="逗号分隔；默认仅 kin（force 的 oracle 因不可控漂移发散）")
    args = ap.parse_args()
    mks = args.mismatches.split(",")
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    print(f"E3D-7 Stage 3 闭环迁移确认 (动态跟踪, {args.device}; mismatches={mks}; "
          f"H={C.H_TRAIN} train / {C.H_EVAL} eval; tbptt={args.tbptt}; seeds {SEEDS})")

    residuals = {}
    for mk in ["force", "kin"]:
        d = DualHead()
        d.load_state_dict(torch.load(MODELS / f"residual_{mk}.pt", map_location="cpu"))
        residuals[mk] = d.to(cfg.device, cfg.dtype).eval()
        for p in residuals[mk].parameters():
            p.requires_grad_(False)                      # 冻结残差，梯度只进策略

    # ---- 坑#4: 训前梯度自检 ----
    print("\n[grad sanity] 未训练策略 BPTT 梯度范数 (H=%d):" % C.H_TRAIN)
    for mk in mks:
        for teacher in TEACHERS:
            torch.manual_seed(0)
            pol = C.PolicyDyn().to(cfg.device, cfg.dtype)
            gen = torch.Generator(device=cfg.device).manual_seed(1)
            s = sample_init(cfg, 16, gen)
            loss = C.rollout_train(pol, cfg, s, C.H_TRAIN,
                                   make_extra_fn(teacher, mk, cfg, residuals))
            loss.backward()
            g = torch.sqrt(sum((p.grad ** 2).sum() for p in pol.parameters())).item()
            print(f"  M_{mk:5s} {teacher:9s}: {g:.2e}")

    # ---- Stage 2: 训练（nominal 与失配无关 → 三 seeds 共享）----
    runs = {}
    t0 = time.time()
    for seed in SEEDS:
        runs[("-", "nominal", seed)] = C.train(
            cfg, make_extra_fn("nominal", "-", cfg, residuals),
            iters=args.iters, seed=seed, tbptt=args.tbptt or None)
        print(f"  trained nominal seed{seed} "
              f"loss→{runs[('-', 'nominal', seed)][1]['loss'][-1]:.4f} "
              f"[{time.time()-t0:.0f}s]")
    for mk in mks:
        for teacher in ["corrected", "oracle"]:
            for seed in SEEDS:
                runs[(mk, teacher, seed)] = C.train(
                    cfg, make_extra_fn(teacher, mk, cfg, residuals),
                    iters=args.iters, seed=seed, tbptt=args.tbptt or None)
                print(f"  trained {teacher} M_{mk} seed{seed} "
                      f"loss→{runs[(mk, teacher, seed)][1]['loss'][-1]:.4f} "
                      f"[{time.time()-t0:.0f}s]")
    for key, (pol, _) in runs.items():
        torch.save(pol.state_dict(), MODELS / f"pol_{key[1]}_{key[0]}_s{key[2]}.pt")

    # ---- Stage 3: 全部策略迁回真实系统评测 ----
    print("\n[eval in REAL system] (H=%d, 固定初态批)" % C.H_EVAL)
    evals = {}
    for mk in mks:
        real_fn = make_extra_fn("oracle", mk, cfg, residuals)
        for teacher in TEACHERS:
            for seed in SEEDS:
                pol = runs[("-" if teacher == "nominal" else mk, teacher, seed)][0]
                evals[(mk, teacher, seed)] = C.rollout_eval(pol, cfg, real_fn)
        for teacher in TEACHERS:
            ls = [evals[(mk, teacher, s)]["loss"] for s in SEEDS]
            rm = [evals[(mk, teacher, s)]["track_rmse"] * 1e3 for s in SEEDS]
            ti = [evals[(mk, teacher, s)]["final_tilt"] for s in SEEDS]
            ff = [evals[(mk, teacher, s)]["ff"] * 1e3 for s in SEEDS]
            print(f"  M_{mk:5s} {teacher:9s}: loss {np.mean(ls):.4f}±{np.std(ls):.4f}  "
                  f"跟踪RMSE {np.mean(rm):.1f}±{np.std(rm):.1f}mm  tilt {np.mean(ti):.2f}°  "
                  f"前后腿差 {np.mean(ff):+.1f}mm")
        Ln = np.mean([evals[(mk, "nominal", s)]["loss"] for s in SEEDS])
        Lc = np.mean([evals[(mk, "corrected", s)]["loss"] for s in SEEDS])
        Lo = np.mean([evals[(mk, "oracle", s)]["loss"] for s in SEEDS])
        gc = (Ln - Lc) / (Ln - Lo + 1e-12)
        print(f"  M_{mk:5s} gap closure = ({Ln:.4f}−{Lc:.4f})/({Ln:.4f}−{Lo:.4f}) = {gc:.2f}")

    # ---- 可视化 ----
    tcol = dict(nominal="tab:gray", corrected="tab:green", oracle="tab:orange")
    fig, ax = plt.subplots(len(mks), 4, figsize=(18, 4.5 * len(mks)), squeeze=False)
    t = np.arange(C.H_EVAL) * cfg.dt
    for row, mk in enumerate(mks):
        a = ax[row, 0]
        for teacher in TEACHERS:
            key = ("-" if teacher == "nominal" else mk, teacher, 0)
            a.plot(runs[key][1]["loss"], color=tcol[teacher], label=teacher)
        a.set_yscale("log"); a.set_title(f"M_{mk}: 训练 loss (各自老师, seed0)")
        a.set_xlabel("iter"); a.legend(fontsize=8)
        a = ax[row, 1]
        zoom = slice(C.H_EVAL - 800, C.H_EVAL)              # 末 2 周期放大
        for teacher in TEACHERS:
            a.plot(t[zoom], evals[(mk, teacher, 0)]["zs"][zoom], color=tcol[teacher],
                   label=teacher)
        a.plot(t[zoom], evals[(mk, "nominal", 0)]["zt"][zoom], "k:", lw=1.2, label="z*(t)")
        a.set_title(f"M_{mk}: 真实系统 z(t) 跟踪 (末2周期)"); a.set_xlabel("t (s)")
        if row == 0:
            a.legend(fontsize=8)
        a = ax[row, 2]
        for teacher in TEACHERS:
            a.plot(t, evals[(mk, teacher, 0)]["atts"], color=tcol[teacher], label=teacher)
        a.set_yscale("log"); a.set_title(f"M_{mk}: 真实系统 tilt(t)"); a.set_xlabel("t (s)")
        a = ax[row, 3]
        for i, teacher in enumerate(TEACHERS):
            ls = [evals[(mk, teacher, s)]["loss"] for s in SEEDS]
            a.bar(i, np.mean(ls), 0.6, yerr=np.std(ls), color=tcol[teacher], capsize=4)
        a.set_xticks(range(3)); a.set_xticklabels(TEACHERS)
        a.set_title(f"M_{mk}: 真实系统 eval loss (3 seeds)")
    fig.suptitle("E3D-7: policies trained in nominal/corrected/oracle twins, ALL evaluated "
                 "in the REAL system (dynamic height tracking)", fontsize=13)
    fig.tight_layout()
    out = FIG / "e3d7_transfer.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    summary = {}
    for mk in mks:
        summary[mk] = {}
        for teacher in TEACHERS:
            es = [evals[(mk, teacher, s)] for s in SEEDS]
            summary[mk][teacher] = dict(
                loss_mean=float(np.mean([e["loss"] for e in es])),
                loss_std=float(np.std([e["loss"] for e in es])),
                track_rmse_mm=float(np.mean([e["track_rmse"] * 1e3 for e in es])),
                tilt_deg=float(np.mean([e["final_tilt"] for e in es])),
                ff_mm=float(np.mean([e["ff"] * 1e3 for e in es])))
        Ln, Lc, Lo = (summary[mk][tc]["loss_mean"] for tc in TEACHERS)
        summary[mk]["gap_closure"] = float((Ln - Lc) / (Ln - Lo + 1e-12))
    (RESULTS / "e3d7_transfer.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}\nsaved {RESULTS / 'e3d7_transfer.json'}")


if __name__ == "__main__":
    main()
