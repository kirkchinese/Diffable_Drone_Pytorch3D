#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定性轨迹对比图（论文用，新增）
================================================================
目的：直观展示「基线在障碍物前停滞 vs GCGL 绕行到达」这一核心论点，
为第5章失败模式分析补充定性证据（此前全文仅有标量指标图）。

数据完整性约束（最高优先级）：
  - 全部取自真实离线评估逐步日志，不平滑/不臆造轨迹。
  - 基线代表 episode：viz_results/exp01_eval/episode_000_log.csv（超时停滞，最大进度58.6%，停在距目标≈5.0 m，无碰撞）
  - GCGL 代表 episode：viz_results/exp21_eval/episode_003_log.csv（绕行到达，进度100%，无碰撞）
  - 两者为各自评估中的「代表性」episode（不同随机场景，起止点不同），图题与正文如实标注。
  - 目标点坐标日志未直接给出，由 (位置, 到目标距离) 序列经最小二乘三边定位反解，并用末端距离校验。

风格：复用 redraw_thesis_figures_nature.py 的 Nature 风格（WenQuanYi Micro Hei / 配色 / 300dpi / 去图内标题）。
输出：docs/.../thesis-latex/figures/traj_qualitative.png
脚本独立、只读日志、不重训、不改动其它图。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
VIZ = BASE / "viz_results"
OUT = BASE / "docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/traj_qualitative.png"

C_BASELINE = "#4C72B0"   # 蓝 - 基线
C_HILIGHT = "#C44E52"    # 红 - GCGL
C_START = "#2CA02C"      # 绿 - 起点
C_GOAL = "#8172B3"       # 紫 - 目标

plt.rcParams.update({
    "font.family": "WenQuanYi Micro Hei",
    "font.sans-serif": ["WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})


def solve_goal_xy(x, y, d):
    """由 (x,y,到目标距离 d) 序列最小二乘三边定位反解目标 (gx,gy)。
    (x-gx)^2+(y-gy)^2 = d^2 ；相邻方程相减消去二次项得线性方程组。"""
    x, y, d = np.asarray(x), np.asarray(y), np.asarray(d)
    A, b = [], []
    x0, y0, d0 = x[0], y[0], d[0]
    for i in range(1, len(x)):
        A.append([2 * (x[i] - x0), 2 * (y[i] - y0)])
        b.append((x[i] ** 2 - x0 ** 2) + (y[i] ** 2 - y0 ** 2) - (d[i] ** 2 - d0 ** 2))
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return sol[0], sol[1]


def load(path):
    df = pd.read_csv(path)
    return df


# ---- 读取并校验代表 episode ----
base = load(VIZ / "exp01_eval/episode_000_log.csv")
gcgl = load(VIZ / "exp21_eval/episode_003_log.csv")

assert base["reached_target"].max() < 0.5, "基线代表 episode 应为未到达（停滞）"
assert base["collision"].max() < 0.5, "基线代表 episode 应无碰撞"
assert gcgl["reached_target"].max() > 0.5, "GCGL 代表 episode 应为到达"
assert gcgl["collision"].max() < 0.5, "GCGL 代表 episode 应无碰撞"

gb = solve_goal_xy(base["pos_x"], base["pos_y"], base["dist_to_target"])
gg = solve_goal_xy(gcgl["pos_x"], gcgl["pos_y"], gcgl["dist_to_target"])

fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))

# ===== (a)(b) 俯视轨迹 =====
for ax, df, g, color, tag, note in [
    (axes[0, 0], base, gb, C_BASELINE, "基线（Baseline-MSE）",
     "停滞：在障碍前减速悬停\n最大进度 58.6%，止于距目标≈5.0 m"),
    (axes[0, 1], gcgl, gg, C_HILIGHT, "GCGL（本文）",
     "绕行到达：进度 100%\n末端距目标 0.06 m，全程无碰撞"),
]:
    x, y = df["pos_x"].values, df["pos_y"].values
    ax.plot(x, y, color=color, lw=2.0, alpha=0.9, zorder=3)
    # 按时间稀疏标记方向
    idx = np.linspace(0, len(x) - 1, 18).astype(int)
    ax.scatter(x[idx], y[idx], c=np.arange(len(idx)), cmap="viridis",
               s=16, zorder=4, edgecolor="white", linewidth=0.3)
    ax.scatter([x[0]], [y[0]], marker="^", s=150, color=C_START,
               edgecolor="k", linewidth=0.6, zorder=5, label="起点")
    ax.scatter([g[0]], [g[1]], marker="*", s=320, color=C_GOAL,
               edgecolor="k", linewidth=0.6, zorder=5, label="目标点")
    ax.scatter([x[-1]], [y[-1]], marker="o", s=70, facecolor="none",
               edgecolor=color, linewidth=1.8, zorder=5, label="终止位置")
    ax.set_title(tag, color=color, fontsize=11, pad=6)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="best", framealpha=0.85)
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#333333")

# ===== (c) 到目标距离–时间 =====
ax = axes[1, 0]
ax.plot(base["time_s"], base["dist_to_target"], color=C_BASELINE, label="基线（停滞）")
ax.plot(gcgl["time_s"], gcgl["dist_to_target"], color=C_HILIGHT, label="GCGL（到达）")
ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="到达半径 0.5 m")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("到目标距离 (m)")
ax.grid(True); ax.legend(loc="best", framealpha=0.85)
ax.text(0.97, 0.62, "基线距离\n停在≈5 m 平台", transform=ax.transAxes,
        ha="right", va="center", fontsize=8.5, color=C_BASELINE)

# ===== (d) 飞行速度–时间 =====
ax = axes[1, 1]
ax.plot(base["time_s"], base["speed"], color=C_BASELINE, label="基线（停滞）")
ax.plot(gcgl["time_s"], gcgl["speed"], color=C_HILIGHT, label="GCGL（到达）")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("飞行速度 (m/s)")
ax.grid(True); ax.legend(loc="best", framealpha=0.85)
ax.text(0.97, 0.78, "基线进入障碍前\n速度跌至≈0", transform=ax.transAxes,
        ha="right", va="center", fontsize=8.5, color=C_BASELINE)

fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.13, hspace=0.55, wspace=0.28)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("已输出:", OUT)
print(f"基线目标(反解)≈({gb[0]:.2f},{gb[1]:.2f}) 末端实测距={base['dist_to_target'].iloc[-1]:.2f}")
print(f"GCGL目标(反解)≈({gg[0]:.2f},{gg[1]:.2f}) 末端实测距={gcgl['dist_to_target'].iloc[-1]:.2f}")
