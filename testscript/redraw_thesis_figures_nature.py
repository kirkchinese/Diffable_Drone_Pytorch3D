#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本科毕业论文图像 —— 统一 Nature 风格重绘脚本
================================================================
数据完整性是最高约束：所有数据点必须来自真实数据源，
不得为美观而平滑 / 修改 / 臆造。柱状图采用规范文件中已逐项核验的精确数值；
时间序列从 metrics.csv 原样读取（仅允许标注的滑动平均，并叠加浅色原始曲线）；
分布图按 testscript/analyze_progress_distribution.py 的口径逐 episode 重算。

去除图内一切 "图4.X" 标题与中文大标题（编号/标题由 LaTeX caption 提供），
仅保留坐标轴标签、图例、必要数值标注与阈值标记。

重绘清单（覆盖同名 PNG）：
  image5  训练 AR 曲线对比          <- checkpoints/thesis/<exp>/metrics.csv : ar
  image6  基线损失尺度分析          <- checkpoints/thesis/exp01_baseline_mse/metrics.csv : loss_collide, loss_v
  image9  速度与无碰撞率对比        <- exp01 / exp21 metrics.csv : avg_speed, collision_free_rate
  image7  目标距离对比             <- exp01 / exp21 metrics.csv : goal_distance_final, goal_distance_best
  image11 五维度稳定AR对比          <- 规范表 4.5 已核验数值（末10步 ar 均值）
  image13 消融实验对比             <- 规范表（稳定AR / 离线SR / 保留率）
  image15 梯度裁剪阈值灵敏度        <- 规范表（稳定AR / 离线SR / 离线CFR）
  image17 失败模式分布             <- 规范表（成功/碰撞/停滞/超时 %）
  image18 失败episode最佳进度分布   <- viz_results/thesis_eval/<exp>/seed<N>/episode_*_log.csv
  grad_norm 梯度范数监控           <- checkpoints/thesis/exp_gradnorm_{baseline,gcgl}/metrics.csv : grad_norm

字体：WenQuanYi Micro Hei；axes.unicode_minus=False；dpi=300；bbox_inches='tight'；白底。
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
CKPT = BASE / "checkpoints" / "thesis"
VIZ = BASE / "viz_results" / "thesis_eval"
OUT = BASE / "docs" / "论文相关" / "5.29需要修订论文" / "thesis" / "thesis-latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 统一 Nature 风格基线
# ---------------------------------------------------------------------------
# 色盲友好统一配色：基线/对照=蓝；本文 GCGL/重点=红；第三系列=绿；第四=橙；第五=紫
C_BASELINE = "#4C72B0"  # 蓝 - 基线 / 对照
C_HILIGHT = "#C44E52"   # 红 - 本文方法 GCGL / 重点
C_THIRD = "#55A868"     # 绿
C_FOURTH = "#CC8963"    # 橙
C_FIFTH = "#8172B3"     # 紫
C_HI_EDGE = "#3A1B1C"   # 高亮描边（深色，用于红色高亮柱上仍可见）

plt.rcParams.update({
    "font.family": "WenQuanYi Micro Hei",
    "font.sans-serif": ["WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# 时间序列实验目录映射
TS_EXP = {
    "Baseline-MSE": "exp01_baseline_mse",
    "GCGL": "exp21_grad_clip_goal",
    "GradClip-Only": "exp22_grad_clip_only",
    "GoalLoss-Only": "exp17_goal_reaching",
}


def _load_metrics(exp_dir):
    """读取某实验的 metrics.csv（原样，不做任何数值修改）。"""
    return pd.read_csv(CKPT / exp_dir / "metrics.csv")


def _roll(y, w=100):
    """滑动平均（仅用于可视化的平滑曲线；原始值另以浅色叠加）。"""
    return pd.Series(y).rolling(w, min_periods=1).mean().values


def _save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] saved {path}")


# ===========================================================================
# image5 —— 训练过程加权到达指数(AR)曲线对比（4 条）
# 数据源: checkpoints/thesis/<exp>/metrics.csv 列 'ar'，横轴 'step'
# ===========================================================================
def make_image5():
    order = ["Baseline-MSE", "GCGL", "GradClip-Only", "GoalLoss-Only"]
    colors = {"Baseline-MSE": C_BASELINE, "GCGL": C_HILIGHT,
              "GradClip-Only": C_THIRD, "GoalLoss-Only": C_FOURTH}
    labels = {"Baseline-MSE": "基线 (MSE)", "GCGL": "GCGL (本文)",
              "GradClip-Only": "仅梯度裁剪", "GoalLoss-Only": "仅目标损失"}

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # 标注基线性能退化区（浅色背景）：基线 ar 峰值后区段
    base = _load_metrics(TS_EXP["Baseline-MSE"])
    peak_step = int(base.loc[base["ar"].idxmax(), "step"])
    ax.axvspan(peak_step, base["step"].max(), color="0.85", alpha=0.35, zorder=0)
    ax.text(peak_step + (base["step"].max() - peak_step) * 0.5,
            0.06, "基线性能退化区", ha="center", va="bottom",
            fontsize=8.5, color="0.4")

    for name in order:
        df = _load_metrics(TS_EXP[name])
        ax.plot(df["step"], df["ar"], color=colors[name], alpha=0.15, lw=0.6, zorder=1)
        ax.plot(df["step"], _roll(df["ar"]), color=colors[name], lw=1.5,
                label=labels[name], zorder=2)

    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("加权到达指数 (AR)")
    ax.set_xlim(left=0)
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    _save(fig, "image5")


# ===========================================================================
# image6 —— 基线实验损失尺度分析（仅 exp01）
# 数据源: checkpoints/thesis/exp01_baseline_mse/metrics.csv 列 'loss_collide','loss_v'
#   上: 对数 y 轴的 loss_collide 与 loss_v
#   下: 比值 loss_collide / loss_v，标注 top15% 高占比区间
# ===========================================================================
def make_image6():
    df = _load_metrics("exp01_baseline_mse")
    step = df["step"].values
    lc = df["loss_collide"].values
    lv = df["loss_v"].values
    # 比值（防 0 除：loss_v 实际为正损失，仍加 eps 以稳健处理）
    ratio = lc / (lv + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # 上面板：对数 y 轴
    ax1.semilogy(step, lc, color=C_HILIGHT, alpha=0.15, lw=0.6)
    ax1.semilogy(step, _roll(lc), color=C_HILIGHT, lw=1.5, label="碰撞损失 $L_{collide}$")
    ax1.semilogy(step, lv, color=C_BASELINE, alpha=0.15, lw=0.6)
    ax1.semilogy(step, _roll(lv), color=C_BASELINE, lw=1.5, label="速度跟踪损失 $L_{v}$")
    ax1.set_ylabel("损失（对数尺度）")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right")

    # 下面板：比值 + top15% 高占比区间标注
    thr = np.nanpercentile(ratio, 85)  # top15% 阈值
    ax2.plot(step, ratio, color="0.55", alpha=0.2, lw=0.6)
    ax2.plot(step, _roll(ratio), color="#8172B3", lw=1.5)
    ax2.axhline(thr, color="black", ls=":", lw=1.0,
                label=f"top15% 阈值 ({thr:.2f})")
    # 浅色标记落入 top15% 的高占比点
    hi_mask = ratio >= thr
    ax2.scatter(step[hi_mask], ratio[hi_mask], s=4, color=C_HILIGHT,
                alpha=0.45, zorder=3, label="高占比区间 (top15%)")
    ax2.set_xlabel("训练迭代次数")
    ax2.set_ylabel("$L_{collide}/L_{v}$")
    ax2.set_xlim(left=0)
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", fontsize=8.5)
    _save(fig, "image6")


# ===========================================================================
# image9 —— 训练过程速度与无碰撞率对比（Baseline vs GCGL）
# 数据源: exp01 / exp21 metrics.csv 列 'avg_speed','collision_free_rate'
# ===========================================================================
def make_image9():
    b = _load_metrics(TS_EXP["Baseline-MSE"])
    g = _load_metrics(TS_EXP["GCGL"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)

    # 上：平均速度
    ax1.plot(b["step"], b["avg_speed"], color=C_BASELINE, alpha=0.15, lw=0.6)
    ax1.plot(b["step"], _roll(b["avg_speed"]), color=C_BASELINE, lw=1.5, label="基线 (MSE)")
    ax1.plot(g["step"], g["avg_speed"], color=C_HILIGHT, alpha=0.15, lw=0.6)
    ax1.plot(g["step"], _roll(g["avg_speed"]), color=C_HILIGHT, lw=1.5, label="GCGL (本文)")
    ax1.set_ylabel("平均速度 (m/s)")
    ax1.grid(axis="y")
    ax1.legend(loc="best")

    # 下：无碰撞率 CFR
    ax2.plot(b["step"], b["collision_free_rate"], color=C_BASELINE, alpha=0.15, lw=0.6)
    ax2.plot(b["step"], _roll(b["collision_free_rate"]), color=C_BASELINE, lw=1.5, label="基线 (MSE)")
    ax2.plot(g["step"], g["collision_free_rate"], color=C_HILIGHT, alpha=0.15, lw=0.6)
    ax2.plot(g["step"], _roll(g["collision_free_rate"]), color=C_HILIGHT, lw=1.5, label="GCGL (本文)")
    ax2.set_xlabel("训练迭代次数")
    ax2.set_ylabel("无碰撞率 (CFR)")
    ax2.set_xlim(left=0)
    ax2.grid(axis="y")
    ax2.legend(loc="best")
    _save(fig, "image9")


# ===========================================================================
# image7 —— 训练过程目标距离对比（Baseline vs GCGL）
# 数据源: exp01 / exp21 metrics.csv 列 'goal_distance_final'(实线),'goal_distance_best'(虚线)
# ===========================================================================
def make_image7():
    b = _load_metrics(TS_EXP["Baseline-MSE"])
    g = _load_metrics(TS_EXP["GCGL"])

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for df, color, lab in [(b, C_BASELINE, "基线 (MSE)"), (g, C_HILIGHT, "GCGL (本文)")]:
        # 原始浅色叠加
        ax.plot(df["step"], df["goal_distance_final"], color=color, alpha=0.12, lw=0.6)
        ax.plot(df["step"], df["goal_distance_best"], color=color, alpha=0.12, lw=0.6)
        # 平滑：final 实线，best 虚线
        ax.plot(df["step"], _roll(df["goal_distance_final"]), color=color, lw=1.5,
                ls="-", label=f"{lab} · 终端距离")
        ax.plot(df["step"], _roll(df["goal_distance_best"]), color=color, lw=1.5,
                ls="--", label=f"{lab} · 最优距离")

    ax.set_xlabel("训练迭代次数")
    ax.set_ylabel("目标距离 (m)")
    ax.set_xlim(left=0)
    ax.grid(axis="y")
    ax.legend(loc="best", fontsize=8.5)
    _save(fig, "image7")


# ===========================================================================
# image11 —— 五维度实验稳定AR对比（5 子面板）
# 数据源: 规范表 4.5 已核验数值（各实验 metrics.csv 末10步 ar 均值）
#   基线柱统一蓝；GCGL 柱红色描边高亮
# ===========================================================================
def make_image11():
    # (标签, AR 值, 是否 GCGL 高亮, 是否基线)
    dims = {
        "损失函数": [("Baseline\nMSE", 0.78, False, True),
                  ("VelDecomp", 0.67, False, False),
                  ("VelAdaptive", 0.70, False, False)],
        "传感器": [("Depth\n(基线)", 0.78, False, True),
                ("LiDAR", 0.60, False, False),
                ("Fusion", 0.58, False, False)],
        "架构": [("CNN-GRU\n(基线)", 0.78, False, True),
               ("CBAM", 0.67, False, False),
               ("Lightweight", 0.44, False, False)],
        "训练策略": [("Baseline", 0.78, False, True),
                  ("GoalLoss", 0.70, False, False),
                  ("GradClip", 0.83, False, False),
                  ("GCGL", 0.91, True, False)],
        "CMA-ES": [("Baseline", 0.78, False, True),
                   ("Decay", 0.67, False, False),
                   ("Guide", 0.63, False, False),
                   ("Meta", 0.30, False, False)],
    }

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.6), sharey=True)
    for ax, (dim, bars) in zip(axes, dims.items()):
        xs = np.arange(len(bars))
        for x, (lab, val, is_gcgl, is_base) in zip(xs, bars):
            if is_gcgl:
                ax.bar(x, val, width=0.7, color=C_HILIGHT,
                       edgecolor=C_HI_EDGE, linewidth=2.2, zorder=2)
            elif is_base:
                ax.bar(x, val, width=0.7, color=C_BASELINE, zorder=2)
            else:
                ax.bar(x, val, width=0.7, color="#B0B7C6", zorder=2)
            ax.text(x, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
        ax.set_title(dim, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("稳定到达指数 (AR)")
    # 统一图例
    handles = [Patch(facecolor=C_BASELINE, label="基线"),
               Patch(facecolor="#B0B7C6", label="对照"),
               Patch(facecolor=C_HILIGHT, edgecolor=C_HI_EDGE, linewidth=2.0, label="GCGL (本文)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, "image11")


# ===========================================================================
# image13 —— 消融实验对比（4 方法 × 3 指标分组柱）
# 数据源: 规范表（稳定AR / 离线SR(%) / 保留率(%)）；SR 加 ±std；GCGL 高亮
#   离线 SR: 74.4 / 77.5 / 76.9 / 83.1
# ===========================================================================
def make_image13():
    methods = ["Baseline-MSE", "GoalLoss-Only", "GradClip-Only", "GCGL"]
    ar = [0.78, 0.70, 0.83, 0.91]
    sr = [74.4, 77.5, 76.9, 83.1]          # 离线 SR (%)
    sr_std = [4.6, 4.1, 5.2, 4.7]
    ret = [60.0, 55.1, 63.7, 69.1]         # 保留率 (%)
    is_gcgl = [False, False, False, True]

    # 统一 y 轴 0–1：SR/保留率以 值/100 绘制，柱顶标注原始 %
    metrics = ["稳定AR", "离线SR", "保留率"]
    vals = np.array([ar, [v / 100 for v in sr], [v / 100 for v in ret]])  # (3 metric, 4 method)
    errs = np.array([[0, 0, 0, 0], [v / 100 for v in sr_std], [0, 0, 0, 0]])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    n_m = len(methods)
    group_w = 0.8
    bw = group_w / n_m
    x = np.arange(len(metrics))
    base_colors = [C_BASELINE, C_THIRD, C_FOURTH, C_HILIGHT]

    for j, method in enumerate(methods):
        offset = (j - (n_m - 1) / 2) * bw
        heights = vals[:, j]
        yerr = errs[:, j]
        bars = ax.bar(x + offset, heights, width=bw * 0.95,
                      color=base_colors[j],
                      edgecolor=(C_HI_EDGE if is_gcgl[j] else "none"),
                      linewidth=(2.0 if is_gcgl[j] else 0),
                      label=method + (" (本文)" if is_gcgl[j] else ""),
                      yerr=[np.where(yerr > 0, yerr, np.nan)],
                      capsize=2.5, error_kw=dict(lw=1.0, ecolor="0.3"), zorder=2)

    # 柱顶原始值标注（AR 原值；SR/保留率原始 %）
    raw = {0: [f"{v:.2f}" for v in ar],
           1: [f"{v:.1f}" for v in sr],
           2: [f"{v:.1f}" for v in ret]}
    for mi in range(len(metrics)):
        for j in range(n_m):
            offset = (j - (n_m - 1) / 2) * bw
            top = vals[mi, j] + (errs[mi, j] if errs[mi, j] > 0 else 0)
            ax.text(x[mi] + offset, top + 0.02, raw[mi][j],
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("归一化指标（AR；SR/100；保留率/100）")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, "image13")


# ===========================================================================
# image15 —— 梯度裁剪阈值灵敏度（4 阈值 × 3 指标分组柱）
# 数据源: 规范表（稳定AR / 离线SR(%) / 离线CFR(%)）；SR、CFR 加 ±std；c=1.0 高亮
#   离线 SR: 81.9 / 83.1 / 79.4 / 75.6
# ===========================================================================
def make_image15():
    cs = ["0.5", "1.0 (默认)", "2.0", "5.0"]
    ar = [0.90, 0.91, 0.88, 0.75]
    sr = [81.9, 83.1, 79.4, 75.6]          # 离线 SR (%)
    sr_std = [3.4, 4.7, 3.6, 4.6]
    cfr = [90.0, 88.1, 88.1, 86.9]         # 离线 CFR (%)
    cfr_std = [4.1, 3.4, 1.4, 2.6]
    is_hi = [False, True, False, False]    # c=1.0 高亮

    metrics = ["稳定AR", "离线SR", "离线CFR"]
    vals = np.array([ar, [v / 100 for v in sr], [v / 100 for v in cfr]])
    errs = np.array([[0, 0, 0, 0], [v / 100 for v in sr_std], [v / 100 for v in cfr_std]])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    n_c = len(cs)
    group_w = 0.82
    bw = group_w / n_c
    x = np.arange(len(metrics))
    base_colors = [C_BASELINE, C_HILIGHT, C_THIRD, C_FOURTH]

    for j, c in enumerate(cs):
        offset = (j - (n_c - 1) / 2) * bw
        heights = vals[:, j]
        yerr = errs[:, j]
        ax.bar(x + offset, heights, width=bw * 0.95,
               color=base_colors[j],
               edgecolor=(C_HI_EDGE if is_hi[j] else "none"),
               linewidth=(2.0 if is_hi[j] else 0),
               label=f"c={c}",
               yerr=[np.where(yerr > 0, yerr, np.nan)],
               capsize=2.5, error_kw=dict(lw=1.0, ecolor="0.3"), zorder=2)

    raw = {0: [f"{v:.2f}" for v in ar],
           1: [f"{v:.1f}" for v in sr],
           2: [f"{v:.1f}" for v in cfr]}
    for mi in range(len(metrics)):
        for j in range(n_c):
            offset = (j - (n_c - 1) / 2) * bw
            top = vals[mi, j] + (errs[mi, j] if errs[mi, j] > 0 else 0)
            ax.text(x[mi] + offset, top + 0.02, raw[mi][j],
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("归一化指标（AR；SR/100；CFR/100）")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y")
    ax.legend(loc="lower left", ncol=4, fontsize=8)
    fig.tight_layout()
    _save(fig, "image15")


# ===========================================================================
# image17 —— 消融实验失败模式分布（4 方法 × 4 类别分组柱，单位 %，N=160/方法）
# 数据源: 规范表（成功/碰撞/停滞/超时 %）
# ===========================================================================
def make_image17():
    methods = ["Baseline-MSE", "GoalLoss-Only", "GradClip-Only", "GCGL"]
    cats = ["成功", "碰撞", "停滞", "超时"]
    data = {
        "Baseline-MSE": [74.4, 11.9, 5.6, 8.1],
        "GoalLoss-Only": [77.5, 13.8, 3.8, 5.0],
        "GradClip-Only": [76.9, 11.2, 4.4, 7.5],
        "GCGL": [83.1, 11.9, 2.5, 2.5],
    }
    is_gcgl = [False, False, False, True]
    base_colors = [C_BASELINE, C_THIRD, C_FOURTH, C_HILIGHT]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    n_m = len(methods)
    group_w = 0.8
    bw = group_w / n_m
    x = np.arange(len(cats))

    for j, method in enumerate(methods):
        offset = (j - (n_m - 1) / 2) * bw
        heights = data[method]
        ax.bar(x + offset, heights, width=bw * 0.95,
               color=base_colors[j],
               edgecolor=(C_HI_EDGE if is_gcgl[j] else "none"),
               linewidth=(2.0 if is_gcgl[j] else 0),
               label=method + (" (本文)" if is_gcgl[j] else ""), zorder=2)
        for xi, h in zip(x, heights):
            ax.text(xi + offset, h + 0.6, f"{h:.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("占比 (%)")
    ax.set_ylim(0, 95)
    ax.grid(axis="y")
    ax.text(0.99, 0.98, "N = 160 / 方法", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="0.4")
    ax.legend(loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(0.99, 0.93))
    fig.tight_layout()
    _save(fig, "image17")


# ===========================================================================
# image18 —— 失败 episode 最佳进度分布图（小提琴/KDE + 箱线）
# 数据源: viz_results/thesis_eval/<exp>/seed<N>/episode_*_log.csv
#   逐 episode：最佳进度 = df['progress_pct'].max()
#   成功判定 = (reached_target==1 任意) 且 (collision 全程无) —— 与
#   testscript/analyze_progress_distribution.py 口径一致（公式 4.21 / 4.25）。
#   必须复现 N=141、<20%:27(19.1%)、>90%:76(53.9%)、per-method 41/36/27/37。
# ===========================================================================
EXPS_18 = {
    "exp01_baseline_mse": "Baseline",
    "exp17_goal_reaching": "GoalLoss",
    "exp21_grad_clip_goal": "GCGL",
    "exp22_grad_clip_only": "GradClip",
}
SEEDS_18 = [0, 42, 123, 456, 789]


def _collect_fail_progress():
    recs = []
    for ek in EXPS_18:
        for s in SEEDS_18:
            seed_dir = VIZ / ek / f"seed{s}"
            if not seed_dir.is_dir():
                raise FileNotFoundError(f"缺少目录: {seed_dir}")
            for cp in sorted(glob.glob(str(seed_dir / "episode_*_log.csv"))):
                df = pd.read_csv(cp)
                reached = (df["reached_target"] == 1).any()
                has_coll = df["collision"].any()
                success = reached and (not has_coll)
                if not success:
                    recs.append({"exp_key": ek,
                                 "progress_pct": float(df["progress_pct"].max())})
    return pd.DataFrame(recs)


def make_image18():
    fail = _collect_fail_progress()
    N = len(fail)
    n_lt20 = int((fail["progress_pct"] < 20).sum())
    n_gt90 = int((fail["progress_pct"] > 90).sum())
    per_method = {EXPS_18[k]: int((fail["exp_key"] == k).sum()) for k in EXPS_18}

    # 数据完整性硬校验：不符即停止，不强改
    assert N == 141, f"N 不符: 期望 141, 实得 {N}"
    assert n_lt20 == 27, f"<20% 不符: 期望 27, 实得 {n_lt20}"
    assert n_gt90 == 76, f">90% 不符: 期望 76, 实得 {n_gt90}"
    expected_pm = {"Baseline": 41, "GoalLoss": 36, "GCGL": 27, "GradClip": 37}
    assert per_method == expected_pm, f"per-method 不符: 期望 {expected_pm}, 实得 {per_method}"
    print(f"[image18] 校验通过: N={N}, <20%={n_lt20} ({100*n_lt20/N:.1f}%), "
          f">90%={n_gt90} ({100*n_gt90/N:.1f}%), per-method={per_method}")

    rng = np.random.default_rng(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    exp_keys = list(EXPS_18.keys())
    labels_short = [EXPS_18[k] for k in exp_keys]
    colors = [C_BASELINE, C_FOURTH, C_HILIGHT, C_THIRD]

    # ---- 左：合计 小提琴(KDE) + 散点 + 箱线 ----
    allp = fail["progress_pct"].values
    kde = gaussian_kde(allp, bw_method=0.15)
    y_grid = np.linspace(-5, 105, 300)
    dens = kde(y_grid)
    dens = dens / dens.max() * 0.55
    ax1.fill_betweenx(y_grid, -dens, 0, color=C_BASELINE, alpha=0.22, zorder=1)
    ax1.plot(-dens, y_grid, color=C_BASELINE, lw=1.4, alpha=0.6, zorder=1)
    jx = rng.uniform(-0.20, 0.20, size=len(allp))
    ax1.scatter(jx, allp, s=12, color=C_BASELINE, alpha=0.5,
                edgecolors="white", linewidth=0.3, zorder=2)
    ax1.boxplot(allp, positions=[0.05], widths=0.28, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.6),
                boxprops=dict(facecolor=C_BASELINE, alpha=0.30),
                whiskerprops=dict(lw=1.0), capprops=dict(lw=1.0), zorder=3)
    ax1.axhline(20, color=C_HILIGHT, ls="--", lw=1.4, label="停滞阈值 (20%)")
    ax1.set_xticks([])
    ax1.set_ylabel("最佳进度 Best-Over-Trajectory (%)")
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(-0.85, 0.85)
    ax1.text(0.97, 0.96,
             f"N = {N}\n< 20%: {n_lt20} ({100*n_lt20/N:.1f}%)\n"
             f"> 90%: {n_gt90} ({100*n_gt90/N:.1f}%)",
             transform=ax1.transAxes, fontsize=8.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FBF2D9", alpha=0.85))
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(axis="y")

    # ---- 右：4 方案分组箱线 ----
    positions = np.arange(len(exp_keys))
    for i, ek in enumerate(exp_keys):
        vals = fail[fail["exp_key"] == ek]["progress_pct"].values
        jx = rng.uniform(-0.20, 0.20, size=len(vals))
        ax2.scatter(i + jx, vals, s=12, color=colors[i], alpha=0.5,
                    edgecolors="white", linewidth=0.3, zorder=2)
        ax2.boxplot(vals, positions=[i], widths=0.45, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.6),
                    boxprops=dict(facecolor=colors[i], alpha=0.30),
                    whiskerprops=dict(lw=1.0), capprops=dict(lw=1.0), zorder=3)
        ax2.text(i, -2.5, f"n={len(vals)}", ha="center", va="top", fontsize=8.5, color="0.4")
    ax2.axhline(20, color=C_HILIGHT, ls="--", lw=1.4, label="停滞阈值 (20%)")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_short)
    ax2.set_ylabel("最佳进度 Best-Over-Trajectory (%)")
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(-0.7, len(exp_keys) - 1 + 0.7)
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(axis="y")

    fig.tight_layout()
    _save(fig, "image18")


# ===========================================================================
# grad_norm —— 梯度范数监控对比（保持数据/统计不变，仅统一风格重导出）
# 数据源: checkpoints/thesis/exp_gradnorm_{baseline,gcgl}/metrics.csv 列 'grad_norm'
#   口径与 §4.4.3 一致：前 3000 步；最大 34.5→26.8、>20 尖峰 10→3。
# ===========================================================================
def make_grad_norm():
    N_STEP = 3000
    CLIP = 1.0

    def load(name):
        df = pd.read_csv(CKPT / name / "metrics.csv").iloc[:N_STEP]
        return df["step"].values, df["grad_norm"].values

    sb, gb = load("exp_gradnorm_baseline")
    sg, gg = load("exp_gradnorm_gcgl")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})

    # 上：时间序列（对数 y）
    ax1.semilogy(sb, gb, color=C_BASELINE, alpha=0.16, lw=0.5)
    ax1.semilogy(sb, _roll(gb, 50), color=C_BASELINE, lw=1.5, label="基线（无裁剪）")
    ax1.semilogy(sg, gg, color=C_HILIGHT, alpha=0.16, lw=0.5)
    ax1.semilogy(sg, _roll(gg, 50), color=C_HILIGHT, lw=1.5, label="GCGL（裁剪+目标损失）")
    ax1.axhline(CLIP, color="0.4", ls="--", lw=1.1, label=f"裁剪阈值 $c={CLIP}$")
    ax1.set_ylabel("梯度范数（对数尺度）")
    ax1.set_xlabel("训练迭代次数")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right")

    # 下：分布直方图（对数 x）
    bins = np.logspace(np.log10(0.1), np.log10(max(gb.max(), gg.max()) * 1.1), 55)
    ax2.hist(gb, bins=bins, alpha=0.55, color=C_BASELINE,
             label=f"基线（中位 {np.median(gb):.1f}，最大 {gb.max():.1f}）")
    ax2.hist(gg, bins=bins, alpha=0.55, color=C_HILIGHT,
             label=f"GCGL（中位 {np.median(gg):.1f}，最大 {gg.max():.1f}）")
    ax2.axvline(20, color="black", ls=":", lw=1.0, label="严重尖峰阈值 (>20)")
    ax2.set_xscale("log")
    ax2.set_xlabel("梯度范数（对数尺度）")
    ax2.set_ylabel("频次")
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    _save(fig, "grad_norm")
    print(f"[grad_norm] baseline: median={np.median(gb):.2f} max={gb.max():.2f} "
          f">20={int((gb > 20).sum())} | gcgl: median={np.median(gg):.2f} "
          f"max={gg.max():.2f} >20={int((gg > 20).sum())}")


# ===========================================================================
def main():
    make_image5()
    make_image6()
    make_image9()
    make_image7()
    make_image11()
    make_image13()
    make_image15()
    make_image17()
    make_image18()
    make_grad_norm()
    print("\n全部 10 张图重绘完成。")


if __name__ == "__main__":
    main()
