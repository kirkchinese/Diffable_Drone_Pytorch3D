"""
接触梯度病理对照实验 —— 无人机(光滑/可微平坦) vs 四足最小代理(SLIP/Hopper)
================================================================================

产出 figures/ 下 6 张图 + results.json 关键数值。所有结论均有图有数。

实验清单
--------
E1  接触力定律及其解析梯度（复现 DiffSim2Real 2024 Fig.2 的四曲线）
E2  Hopper 弹跳轨迹（飞行+支撑相切换的物理正确性自检）
E3  损失景观：无人机光滑基线(干净抛物线) vs Hopper 软/刚/平滑(锯齿)
E4  解析一阶梯度(FoG) vs 有限差分"真梯度" —— 接触时序事件附近的偏差
E5  BPTT 梯度范数 vs 视野长度 —— 接触致指数增长 vs 光滑有界；grad_decay 压幅
E6  刚度/锐度扫描(soft->hard) —— 景观粗糙度与梯度偏差单调上升

运行: python run_experiments.py [--device cuda:0|cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import slip_dynamics as sd

# ---- 样式（沿用毕业论文图风：基线蓝 / 接触红 / 平滑绿 / 硬接触灰） ----
C_BASE = "#4C72B0"   # 无人机光滑基线 / soft
C_CONTACT = "#C44E52"  # 接触 / stiff
C_SMOOTH = "#55A868"  # 解析平滑
C_HARD = "#8C8C8C"    # 硬接触
C_STOCH = "#64B5CD"   # 随机平滑
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

results: dict = {}


def _np(t):
    return t.detach().cpu().numpy()


# =====================================================================
# E1 接触力定律及其解析梯度（DiffSim2Real Fig.2）
# =====================================================================
def exp_force_law(dev):
    d = torch.linspace(-0.04, 0.04, 1601, device=dev)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    specs = [
        ("hard", C_HARD, "Hard  f=F0·H(d)"),
        ("soft", C_BASE, "Soft  f=relu(k·d)"),
        ("stoch", C_STOCH, "Stochastic-smoothed (FoG still 0)"),
        ("smooth", C_SMOOTH, "Analytic-smooth  f=F0·σ(d/ε)"),
    ]
    for mdl, c, lab in specs:
        f, gr = sd.contact_force_law(d, mdl, F0=50.0, k=4000.0, eps=0.006, sigma=0.012)
        ax[0].plot(_np(d) * 1e3, _np(f), color=c, lw=2, label=lab)
        ax[1].plot(_np(d) * 1e3, _np(gr), color=c, lw=2, label=lab)
    for a in ax:
        a.axvline(0, color="k", lw=0.8, ls=":")
        a.set_xlabel("penetration depth  d  [mm]   (d>0: in contact)")
    ax[0].set_ylabel(r"normal force $f_n$ [N]")
    ax[0].set_title("(a) contact force law")
    ax[0].set_ylim(-2, 80)
    ax[1].set_ylabel(r"analytic gradient $\partial f_n/\partial d$")
    ax[1].set_title("(b) first-order gradient (FoG)")
    ax[1].set_ylim(-100, 1400)
    ax[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("E1  Contact force & its analytic gradient  (reproduces DiffSim2Real 2024, Fig.2)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E1_contact_force_law.png"), bbox_inches="tight")
    plt.close(fig)
    # 关键数值：硬/随机平滑 FoG 恒 0；解析平滑梯度有界
    _, gr_hard = sd.contact_force_law(d, "hard")
    _, gr_smooth = sd.contact_force_law(d, "smooth", F0=50.0, eps=0.006)
    results["E1"] = {
        "hard_grad_abs_max": float(gr_hard.abs().max().item()),
        "smooth_grad_abs_max": float(gr_smooth.abs().max().item()),
        "note": "hard/stoch FoG≡0 (uninformative); smooth FoG bounded & informative",
    }
    print("[E1] done. hard|grad|max=%.3g  smooth|grad|max=%.3g"
          % (results["E1"]["hard_grad_abs_max"], results["E1"]["smooth_grad_abs_max"]))


# =====================================================================
# E2 Hopper 弹跳轨迹（物理正确性自检）
# =====================================================================
def exp_trajectories(dev):
    n, dt = 4000, 1e-3
    y0 = torch.tensor(1.2, device=dev)
    v0 = torch.tensor(0.0, device=dev)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for mdl, c, kn in [("soft", C_BASE, 4000.0), ("stiff", C_CONTACT, 30000.0),
                       ("smooth", C_SMOOTH, 4000.0)]:
        Y, V = sd.rollout_hopper(y0, v0, n, dt=dt, model=mdl, k_n=kn, k_d=12.0, eps=0.008)
        t = np.arange(n + 1) * dt
        ax[0].plot(t, _np(Y), color=c, lw=1.6, label=f"{mdl} (k_n={kn:.0f})")
        ax[1].plot(_np(Y), _np(V), color=c, lw=1.0, alpha=0.8, label=mdl)
    ax[0].axhline(0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("height y [m]")
    ax[0].set_title("(a) bouncing trajectories (flight↔stance)")
    ax[0].legend(fontsize=8)
    ax[1].axvline(0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("height y [m]"); ax[1].set_ylabel("velocity v [m/s]")
    ax[1].set_title("(b) phase portrait")
    ax[1].legend(fontsize=8)
    fig.suptitle("E2  Differentiable vertical hopper — minimal hybrid (contact) system", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E2_trajectories.png"), bbox_inches="tight")
    plt.close(fig)
    print("[E2] done.")


# =====================================================================
# E3 损失景观  +  E4 解析 vs 有限差分梯度
# =====================================================================
def _hopper_cost_curve(dev, v0_grid, n, dt, model, k_n, eps, y_target, grad_decay=1.0,
                       requires_grad=False):
    """对 v0 网格做一次 batched rollout，返回 (cost_curve, v0_grid_leaf)。"""
    y0 = torch.full_like(v0_grid, 1.2)
    v = v0_grid.clone().requires_grad_(requires_grad)
    Y, _ = sd.rollout_hopper(y0, v, n, dt=dt, model=model, k_n=k_n, k_d=12.0, eps=eps,
                             grad_decay=grad_decay)
    cost = (Y[-1] - y_target) ** 2
    return cost, v


def exp_landscape_and_bias(dev):
    y_target = 0.6
    v0_grid = torch.linspace(-1.0, 1.0, 801, device=dev)

    configs = [
        ("smooth-baseline(drone)", "baseline", None, None, C_BASE),
        ("hopper-smooth", "smooth", 4000.0, 0.008, C_SMOOTH),
        ("hopper-soft", "soft", 4000.0, None, "#DD8452"),
        ("hopper-stiff", "stiff", 30000.0, None, C_CONTACT),
    ]

    # ---- E3 景观 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    rough = {}
    curves = {}
    for name, mdl, kn, eps, c in configs:
        if mdl == "baseline":
            # 无人机光滑基线：把 v0 当作"净加速度指令"扫描（光滑控制问题）
            u = v0_grid * 8.0
            y0 = torch.full_like(v0_grid, 1.2)
            v0 = torch.zeros_like(v0_grid)
            Y = sd.rollout_smooth_pointmass(y0, v0, u, n_steps=150, dt=0.02)
            cost = (Y[-1] - y_target) ** 2
        else:
            cost, _ = _hopper_cost_curve(dev, v0_grid, 3000, 1e-3, mdl, kn or 4000.0,
                                         eps or 0.008, y_target)
        cost = cost.detach()
        curves[name] = cost
        rough[name] = sd.landscape_roughness(cost)
        ax[0].plot(_np(v0_grid), _np(cost), color=c, lw=1.6, label=name)
    ax[0].set_xlabel("swept initial parameter  (v0  /  accel-cmd)")
    ax[0].set_ylabel(r"terminal cost  $(y_T - y^*)^2$")
    ax[0].set_title("(a) loss landscape: smooth vs contact")
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(0, float(np.percentile(_np(curves["hopper-stiff"]), 99)) * 1.1)

    # 粗糙度条形
    names = list(rough.keys())
    vals = [rough[n] for n in names]
    cols = [C_BASE, C_SMOOTH, "#DD8452", C_CONTACT]
    ax[1].bar(range(len(names)), vals, color=cols)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax[1].set_ylabel("landscape roughness  (Σ|Δ²cost|, log)")
    ax[1].set_title("(b) roughness metric")
    fig.suptitle("E3  Contact roughens the optimization landscape (cf. SHAC 2022, Fig.2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E3_landscape.png"), bbox_inches="tight")
    plt.close(fig)
    results["E3_roughness"] = rough
    print("[E3] done. roughness:", {k: round(v, 3) for k, v in rough.items()})

    # ---- E4 解析 FoG vs 有限差分 ----
    # 增加 "hopper-stiff + grad_decay" 一列：证明梯度衰减压幅但不压偏(对应验收Q7)
    configs_e4 = [(n, m, k, e, c, 1.0) for (n, m, k, e, c) in configs]
    configs_e4.append(("hopper-stiff\n+grad_decay0.6", "stiff", 30000.0, None, "#8172B3", 0.6))
    bias_cols = [C_BASE, C_SMOOTH, "#DD8452", C_CONTACT, "#8172B3"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bias = {}
    sign_disagree = {}  # 尺度无关的方向偏差(对正标量缩放/裁剪不变)
    for name, mdl, kn, eps, c, gd in configs_e4:
        if mdl == "baseline":
            u_grid = v0_grid * 8.0
            u_leaf = u_grid.clone().requires_grad_(True)
            y0 = torch.full_like(v0_grid, 1.2); v0 = torch.zeros_like(v0_grid)
            Y = sd.rollout_smooth_pointmass(y0, v0, u_leaf, n_steps=150, dt=0.02)
            cost = (Y[-1] - y_target) ** 2
            cost.sum().backward()
            ana = u_leaf.grad.detach()
            cost_d = cost.detach()
            # FD along grid (param = u_grid)
            dparam = (u_grid[2:] - u_grid[:-2])
            fd = (cost_d[2:] - cost_d[:-2]) / dparam
            xg = u_grid
        else:
            cost, vleaf = _hopper_cost_curve(dev, v0_grid, 3000, 1e-3, mdl, kn or 4000.0,
                                             eps or 0.008, y_target, grad_decay=gd,
                                             requires_grad=True)
            cost.sum().backward()
            ana = vleaf.grad.detach()
            cost_d = cost.detach()
            fd = (cost_d[2:] - cost_d[:-2]) / (v0_grid[2:] - v0_grid[:-2])
            xg = v0_grid
        ana_in = ana[1:-1]
        rel_bias = float((ana_in - fd).abs().mean().item())
        bias[name.replace("\n", " ")] = rel_bias
        # 方向偏差：仅在"真梯度非平凡"处统计 sign(analytic)≠sign(finite-diff) 的比例
        mask = fd.abs() > 1.0
        denom = float(mask.sum().item()) + 1e-9
        sd_rate = float(((torch.sign(ana_in) != torch.sign(fd)) & mask).sum().item()) / denom
        sign_disagree[name.replace("\n", " ")] = round(sd_rate, 4)
        if mdl == "baseline" or name.startswith("hopper-stiff") and gd == 1.0:
            ax[0].plot(_np(xg)[1:-1], _np(ana_in), color=c, lw=1.4, label=f"{name} analytic-FoG")
            ax[0].plot(_np(xg)[1:-1], _np(fd), color=c, lw=1.0, ls="--", alpha=0.7,
                       label=f"{name} finite-diff")
    ax[0].set_xlabel("swept parameter")
    ax[0].set_ylabel("dCost/dparam")
    ax[0].set_title("(a) FoG vs finite-diff: smooth agrees, stiff diverges")
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(-60, 60)

    names = list(bias.keys()); vals = [bias[n] for n in names]
    ax[1].bar(range(len(names)), vals, color=bias_cols)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax[1].set_ylabel("mean |analytic − finite-diff|  (log)")
    ax[1].set_title("(b) first-order gradient bias")
    fig.suptitle("E4  Contact injects gradient BIAS (analytic FoG ≠ true gradient)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E4_grad_bias.png"), bbox_inches="tight")
    plt.close(fig)
    results["E4_grad_bias"] = bias
    results["E4_sign_disagree"] = sign_disagree
    print("[E4] done. mean|ana-fd|:", {k: round(v, 4) for k, v in bias.items()})
    print("        sign-disagree:", sign_disagree)


# =====================================================================
# E5 BPTT 梯度范数 vs 视野长度（+ grad_decay 压幅）
# =====================================================================
def exp_bptt_norm(dev):
    horizons = [int(h) for h in np.linspace(200, 6000, 25)]
    dt = 1e-3
    v0_val = 0.3

    def grad_y_T_wrt_v0(n, model, kn, eps, grad_decay=1.0):
        y0 = torch.tensor(1.2, device=dev)
        v0 = torch.tensor(v0_val, device=dev, requires_grad=True)
        Y, _ = sd.rollout_hopper(y0, v0, n, dt=dt, model=model, k_n=kn, k_d=12.0, eps=eps,
                                 grad_decay=grad_decay)
        (g,) = torch.autograd.grad(Y[-1], v0)
        return abs(float(g.item()))

    series = {
        "hopper-stiff (k_n=3e4)": ("stiff", 30000.0, 0.008, 1.0, C_CONTACT),
        "hopper-soft (k_n=4e3)": ("soft", 4000.0, 0.008, 1.0, "#DD8452"),
        "hopper-smooth": ("smooth", 4000.0, 0.008, 1.0, C_SMOOTH),
        "hopper-stiff + grad_decay=0.6": ("stiff", 30000.0, 0.008, 0.6, "#8172B3"),
    }
    # 光滑基线（无接触）作参照
    def grad_baseline(n):
        y0 = torch.tensor(1.2, device=dev)
        v0 = torch.tensor(v0_val, device=dev, requires_grad=True)
        u = torch.tensor(0.0, device=dev)
        Y = sd.rollout_smooth_pointmass(y0, v0, u, n_steps=n, dt=dt)
        (g,) = torch.autograd.grad(Y[-1], v0)
        return abs(float(g.item()))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    base = [grad_baseline(n) for n in horizons]
    ax.plot(horizons, base, color=C_BASE, lw=2, marker="o", ms=3,
            label="smooth point-mass (drone-analog, no contact)")
    data = {"horizons": horizons, "smooth_baseline": base}
    for name, (mdl, kn, eps, gd, c) in series.items():
        ys = [grad_y_T_wrt_v0(n, mdl, kn, eps, gd) for n in horizons]
        ax.plot(horizons, ys, color=c, lw=2, marker="s", ms=3, label=name)
        data[name] = ys
    ax.set_yscale("log")
    ax.set_xlabel("rollout horizon  (#steps,  dt=1e-3)")
    ax.set_ylabel(r"$|\partial y_T / \partial v_0|$   (BPTT gradient magnitude, log)")
    ax.set_title("E5  BPTT gradient: bounded for smooth, blows up through contact\n"
                 "(grad_decay tames magnitude — cf. drone GDecay / Song 2024 state-align α)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E5_bptt_norm.png"), bbox_inches="tight")
    plt.close(fig)
    results["E5"] = {
        "stiff_max": max(data["hopper-stiff (k_n=3e4)"]),
        "smooth_baseline_max": max(base),
        "stiff_decay_max": max(data["hopper-stiff + grad_decay=0.6"]),
    }
    print("[E5] done. stiff_max=%.3g  baseline_max=%.3g  stiff+decay_max=%.3g"
          % (results["E5"]["stiff_max"], results["E5"]["smooth_baseline_max"],
             results["E5"]["stiff_decay_max"]))


# =====================================================================
# E6 刚度/锐度扫描（soft -> hard）：粗糙度 & 偏差单调上升
# =====================================================================
def exp_stiffness_sweep(dev):
    y_target = 0.6
    v0_grid = torch.linspace(-1.0, 1.0, 601, device=dev)
    k_list = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    dt = 5e-4
    n = 4000  # 总时长 2.0s 固定

    rough, bias = [], []
    for kn in k_list:
        cost, vleaf = _hopper_cost_curve(dev, v0_grid, n, dt, "soft", float(kn), 0.008,
                                         y_target, requires_grad=True)
        cost.sum().backward()
        ana = vleaf.grad.detach()[1:-1]
        cost_d = cost.detach()
        fd = (cost_d[2:] - cost_d[:-2]) / (v0_grid[2:] - v0_grid[:-2])
        rough.append(sd.landscape_roughness(cost_d))
        bias.append(float((ana - fd).abs().mean().item()))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].plot(k_list, rough, color=C_CONTACT, lw=2, marker="o")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("contact stiffness  k_n  (soft → hard)")
    ax[0].set_ylabel("landscape roughness (log)")
    ax[0].set_title("(a) roughness ↑ with stiffness")
    ax[1].plot(k_list, bias, color="#8172B3", lw=2, marker="s")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("contact stiffness  k_n  (soft → hard)")
    ax[1].set_ylabel("mean |analytic − finite-diff| (log)")
    ax[1].set_title("(b) FoG bias ↑ with stiffness")
    fig.suptitle("E6  Soft→hard contact annealing trade-off (cf. Schwarke 2024 / DiffSim2Real)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E6_stiffness_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    results["E6"] = {"k_list": k_list, "roughness": rough, "bias": bias}
    print("[E6] done. roughness range %.3g..%.3g  bias range %.3g..%.3g"
          % (min(rough), max(rough), min(bias), max(bias)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)  # 梯度精度微实验 -> float64
    print(f"device = {dev}, dtype = float64")

    exp_force_law(dev)
    exp_trajectories(dev)
    exp_landscape_and_bias(dev)
    exp_bptt_norm(dev)
    exp_stiffness_sweep(dev)

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nAll experiments done. Figures in figures/, metrics in results.json")


if __name__ == "__main__":
    main()
