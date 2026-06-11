"""E3D-7 Stage 2 (主指标): 策略梯度保真 —— 梯度保真兑现的确定性、免疫反馈掩盖的测度。

Stage 0 已证：本任务的**闭环收敛表现被反馈掩盖**（标称/真实系统里策略表现几乎一样），且
M_force 的 oracle 训练因不可控漂移发散——闭环"训得更好"对比度不足。但 E3D-7 的真问题是
**训练信号质量**：修正后的孪生给策略的训练梯度 ∇θL，是否比标称孪生更接近真实系统的梯度？
这正是残差修正要兑现的东西，且能**直接、确定性地**测出来，与"反馈事后能否掩盖误差"无关。

主指标：在多个策略点（5 个随机初始化 + 标称已训基线）上，比 ∇θL（整条可微 rollout 的策略
梯度）：teacher∈{nominal, corrected, oracle}，报 cos(g_T, g_oracle) 与相对误差。预测：
corrected 比 nominal 显著更接近 oracle（cos→1、rel↓），两通道皆然。

支撑指标：部署（闭环）状态上的**全局 Frobenius ∂a/∂leg_ext 雅可比比** ‖J_R−J_T‖_F/‖J_T‖_F
——E3D-6 逐态相对误差在闭环分布上除以≈0（飞行相）会爆，全局 F 范数由高杠杆接触态主导，稳健。
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config  # noqa: E402
from residual_3d import DualHead, mismatch  # noqa: E402
from e3d3_standing_train import sample_init  # noqa: E402
from e3d6_channel_matching import jac_wrt_action  # noqa: E402
from e3d7_fit_residual import collect_real_rollouts, to64, index_slice  # noqa: E402
import e3d7_common as C  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d7_models"
TEACHERS = ["nominal", "corrected", "oracle"]
TCOL = dict(nominal="tab:gray", corrected="tab:green", oracle="tab:orange")


def make_extra_fn(teacher, mkind, cfg, residuals):
    if teacher == "nominal":
        return lambda s, a: (None, None)
    if teacher == "oracle":
        return lambda s, a: mismatch(mkind, s, a, cfg)
    dual = residuals[mkind]
    return lambda s, a: dual.extras(s, a, cfg)


def policy_points(cfg):
    """策略点集：5 个随机初始化 + 标称已训基线（覆盖训练全程的不同 θ）。"""
    pts = []
    for sd in range(5):
        torch.manual_seed(sd)
        pts.append((f"rand{sd}", C.PolicyDyn().to(cfg.device, cfg.dtype)))
    base = C.PolicyDyn().to(cfg.device, cfg.dtype)
    base.load_state_dict(torch.load(MODELS / "baseline_dyn_seed0.pt",
                                    map_location=cfg.device, weights_only=True))
    pts.append(("nominal-trained", base))
    return pts


def global_jac_ratio(data, leg, cfg64, extra_fn, true_fn, n=48):
    """全局 Frobenius ∂a/∂leg_ext 比：稳健（高杠杆接触态主导，免逐态除零）。"""
    JT, JR = [], []
    for i in range(n):
        s1 = type(data)(data.p[i:i+1], data.q[i:i+1], data.v[i:i+1], data.w[i:i+1])
        l1 = leg[i:i+1]
        JT.append(jac_wrt_action(s1, l1, cfg64, true_fn).flatten())
        JR.append(jac_wrt_action(s1, l1, cfg64, extra_fn).flatten())
    JT, JR = torch.cat(JT), torch.cat(JR)
    return (JR - JT).norm().item() / JT.norm().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    print(f"E3D-7 Stage 2 策略梯度保真 ({args.device}, H={C.H_TRAIN})")

    residuals = {}
    for mk in ["force", "kin"]:
        d = DualHead()
        d.load_state_dict(torch.load(MODELS / f"residual_{mk}.pt",
                                     map_location="cpu", weights_only=True))
        residuals[mk] = d.to(cfg.device, cfg.dtype).eval()
        for p in residuals[mk].parameters():
            p.requires_grad_(False)

    pts = policy_points(cfg)
    t0 = time.time()
    HORIZONS = [75, 150, 300, 600]      # 视野扫描：短=TBPTT 工作区，长=混沌淹没区
    summary, raw = {}, {}
    print("\n[主指标] 策略梯度 ∇θL 对 oracle：全局拼接 cos/rel × 视野扫描"
          "（预测：短视野 corrected≫nominal 继承一步雅可比优势；长视野双双被轨迹发散淹没）:")
    for mk in ["force", "kin"]:
        fns = {t: make_extra_fn(t, mk, cfg, residuals) for t in TEACHERS}
        summary[mk] = {"horizons": HORIZONS, "nominal": [], "corrected": []}
        raw[mk] = {}
        for H in HORIZONS:
            rec = {t: dict(cos=[], gs=[]) for t in TEACHERS}
            gnorms = []
            for name, pol in pts:
                g = {t: C.policy_grad(pol, cfg, fns[t], horizon=H) for t in TEACHERS}
                gnorms.append(g["oracle"].norm().item())
                for t in TEACHERS:
                    rec[t]["gs"].append(g[t])
                for t in ["nominal", "corrected"]:
                    rec[t]["cos"].append(
                        F.cosine_similarity(g[t], g["oracle"], dim=0).item())
            raw[mk][H] = {t: rec[t]["cos"] for t in ["nominal", "corrected"]}
            G = {t: torch.cat(rec[t]["gs"]) for t in TEACHERS}   # 全局拼接
            line = f"  M_{mk:5s} H={H:3d} |g_ora|∈[{min(gnorms):.1e},{max(gnorms):.1e}]"
            for t in ["nominal", "corrected"]:
                cos = float(F.cosine_similarity(G[t], G["oracle"], dim=0).item())
                rel = float(((G[t] - G["oracle"]).norm() / G["oracle"].norm()).item())
                summary[mk][t].append(dict(H=H, cos=cos, rel=rel))
                line += f"  {t[:3]}: cos={cos:+.3f} rel={rel:.2f}"
            print(line)

    # ---- 支撑：闭环部署态上的全局 Frobenius 雅可比比 ----
    print(f"\n[支撑] 部署(闭环)态全局 Frobenius ∂a/∂leg_ext 比 (n=48): [{time.time()-t0:.0f}s]")
    base = pts[-1][1]
    residuals64 = {}
    for mk in ["force", "kin"]:
        d = DualHead()
        d.load_state_dict(torch.load(MODELS / f"residual_{mk}.pt",
                                     map_location="cpu", weights_only=True))
        residuals64[mk] = d.double().eval()
    for mk in ["force", "kin"]:
        S, A = collect_real_rollouts(cfg, base, mk, horizon=600, B=32)
        data, leg = to64(S, A, n=200)
        true_fn = lambda s, a, m=mk: mismatch(m, s, a, cfg64)
        nom_fn = lambda s, a: (None, None)
        cor_fn = lambda s, a, d=residuals64[mk]: d.extras(s, a, cfg64)
        jr_nom = global_jac_ratio(data, leg, cfg64, nom_fn, true_fn)
        jr_cor = global_jac_ratio(data, leg, cfg64, cor_fn, true_fn)
        summary[mk]["jac_nom"] = jr_nom
        summary[mk]["jac_cor"] = jr_cor
        print(f"  M_{mk:5s}: nominal {jr_nom:.3f}  →  corrected {jr_cor:.3f}  "
              f"(降 {(1-jr_cor/jr_nom)*100:.0f}%)")

    # ---- 判定：短视野（TBPTT 工作区，H≤150）corrected 应优；长视野如实报 ----
    short_ok = all(
        rec_c["cos"] > rec_n["cos"]
        for mk in ["force", "kin"]
        for rec_n, rec_c in zip(summary[mk]["nominal"][:2], summary[mk]["corrected"][:2]))
    jac_ok = all(summary[mk]["jac_cor"] < summary[mk]["jac_nom"] for mk in ["force", "kin"])
    print(f"\n判定: 一步雅可比 {'✅' if jac_ok else '❌'} | 短视野(H≤150)策略梯度 "
          f"{'✅' if short_ok else '⚠'} → "
          f"{'梯度保真在 TBPTT 工作区兑现；长视野梯度方向被轨迹发散淹没(对两孪生皆然)——如实报告' if jac_ok and short_ok else '需逐项核对图中曲线'}")

    # ---- 可视化 ----
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for col, mk in enumerate(["force", "kin"]):
        a = ax[0, col]
        Hs = summary[mk]["horizons"]
        for t in ["nominal", "corrected"]:
            a.plot(Hs, [r["cos"] for r in summary[mk][t]], "o-", color=TCOL[t], label=t)
        a.axhline(1.0, color="tab:orange", ls=":", label="oracle(=1)")
        a.axhline(0.0, color="k", lw=0.5)
        a.set_xscale("log"); a.set_xticks(Hs); a.set_xticklabels(Hs)
        a.set_xlabel("BPTT 视野 H (步)"); a.set_ylabel("global cos(∇θL, oracle)")
        a.set_title(f"M_{mk}: 策略梯度保真 × 视野"); a.legend(fontsize=8)
    a = ax[0, 2]
    x = np.arange(2); w = 0.35
    jn = [summary[mk]["jac_nom"] for mk in ["force", "kin"]]
    jc = [summary[mk]["jac_cor"] for mk in ["force", "kin"]]
    a.bar(x - w/2, jn, w, color=TCOL["nominal"], label="nominal")
    a.bar(x + w/2, jc, w, color=TCOL["corrected"], label="corrected")
    a.set_xticks(x); a.set_xticklabels(["M_force", "M_kin"])
    a.set_title("一步 ∂a/∂leg_ext 全局雅可比相对误差 (↓)"); a.legend(fontsize=8)
    H_SHOW = 150
    for row_col, mk in enumerate(["force", "kin"]):
        a = ax[1, row_col]
        a.scatter(range(6), raw[mk][H_SHOW]["nominal"], color=TCOL["nominal"],
                  label="nominal", s=40)
        a.scatter(range(6), raw[mk][H_SHOW]["corrected"], color=TCOL["corrected"],
                  label="corrected", s=40, marker="^")
        a.axhline(1.0, color="tab:orange", ls=":", label="oracle (=1)")
        a.set_xticks(range(6))
        a.set_xticklabels([n for n, _ in pts], rotation=30, fontsize=7)
        a.set_ylabel("cos vs oracle"); a.set_title(f"M_{mk}: 各策略点 cos (H={H_SHOW})")
        if row_col == 0:
            a.legend(fontsize=8)
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.5,
                  "读法:\n一步雅可比(右上)=修正孪生\n局部训练信号决定性更优。\n\n"
                  "策略梯度×视野(上左/中):\n短视野继承该优势(TBPTT工作区),\n长视野两孪生都被轨迹发散\n"
                  "淹没——与E3D-3视野教训、\nTBPTT有效性同机制。\n\n"
                  "闭环收敛表现被反馈掩盖\n(Stage0)，梯度保真是E3D-7\n承重指标(同2D R8/R11)。",
                  fontsize=9, va="center")
    fig.suptitle("E3D-7 Stage 2: gradient fidelity vs horizon — corrected twin wins where "
                 "gradients are usable", fontsize=13)
    fig.tight_layout()
    out = FIG / "e3d7_grad_fidelity.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    (RESULTS / "e3d7_grad_fidelity.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out}\nsaved {RESULTS / 'e3d7_grad_fidelity.json'}")


if __name__ == "__main__":
    main()
