#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw thesis Figure 1.1: differentiable end-to-end training framework.

This is a structural figure, not a data plot. The layout is intentionally
split into a forward rollout row, a loss row, and a backward/update row so the
reader can see the method loop at a glance.

Output:
  docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/system_framework.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/system_framework.png"
)

plt.rcParams.update(
    {
        "font.family": "WenQuanYi Micro Hei",
        "font.sans-serif": ["WenQuanYi Micro Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
    }
)

EDGE = "#30343B"
TEXT = "#20242A"
MUTED = "#5E6877"
FWD = "#30343B"
BWD = "#C44E52"

C_STATE = "#DDEAF7"
C_RENDER = "#55A868"
C_OBS = "#EAF4EC"
C_POLICY = "#8172B3"
C_DYN = "#4C72B0"
C_TRAJ = "#D9E6F6"
C_LOSS = "#C44E52"
C_GCGL = "#DD8452"
C_UPDATE = "#EEE8F6"
C_BAND = "#F6F7F9"

fig, ax = plt.subplots(figsize=(15.0, 7.0))
ax.set_xlim(0, 150)
ax.set_ylim(0, 70)
ax.axis("off")


def band(x, y, w, h, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=1.2",
            linewidth=0.8,
            edgecolor="#D7DCE2",
            facecolor=C_BAND,
        )
    )
    ax.text(
        x + 2.0,
        y + h + 1.2,
        label,
        ha="left",
        va="center",
        fontsize=9,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95),
    )


def block(x, y, w, h, color, lines, chap=None, fs=10.0, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.32,rounding_size=1.55",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        linespacing=1.45,
    )
    if chap:
        ax.text(
            x + w - 1.0,
            y + h - 1.0,
            chap,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc="#00000055", ec="none"),
        )


def arrow(p1, p2, label="", color=FWD, dashed=False, rad=0.0, fs=8.4, offset=(0, 1.35)):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.35,
            color=color,
            shrinkA=3,
            shrinkB=3,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (p1[0] + p2[0]) / 2 + offset[0],
            (p1[1] + p2[1]) / 2 + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
        )


def legend_line(x, y, color, dashed, label):
    ax.plot([x, x + 7], [y, y], color=color, lw=1.5, ls="--" if dashed else "-")
    ax.text(x + 8.5, y, label, ha="left", va="center", fontsize=9.2, color=TEXT)


# Background bands
band(3, 43, 144, 20, "1  前向：可微仿真展开")
band(78, 25.2, 69, 13.2, "2  损失：从轨迹和动力学量构造优化目标")
band(35, 6.2, 77, 13.2, "3  反向：物理梯度回传并更新策略参数")

# Forward row
y, h = 48.2, 9.8
block(6, y, 17, h, C_STATE, ["仿真状态", r"$s_t$ 与场景"], chap="第2/3章", fs=9.6, tc="#16233A")
block(28, y, 18, h, C_RENDER, ["可微深度渲染", "PyTorch3D"], chap="第3章", fs=9.5)
block(51, y, 18, h, C_OBS, ["观测输入", r"深度图 $D_t$", r"+ 状态向量 $s_t$"], chap="第3章", fs=9.0, tc="#14321F")
block(74, y, 18, h, C_POLICY, [r"策略网络 $\pi_\theta$", "CNN-GRU"], chap="第4章", fs=9.5)
block(97, y, 18, h, C_DYN, ["可微动力学积分", "执行器/阻力/位置"], chap="第2章", fs=9.1)
block(120, y, 21, h, C_TRAJ, ["展开轨迹", r"$s_{t+1:T}$", "距离/碰撞/速度"], chap="第2/3章", fs=9.0, tc="#16233A")

arrow((23, y + h / 2), (28, y + h / 2), "相机位姿/场景")
arrow((46, y + h / 2), (51, y + h / 2), "深度图")
arrow((69, y + h / 2), (74, y + h / 2), "观测")
arrow((92, y + h / 2), (97, y + h / 2), r"动作 $a_t$")
arrow((115, y + h / 2), (120, y + h / 2), "状态更新")

# Time rollout hint, kept separate from the main flow to avoid crossings.
arrow(
    (132, 61.2),
    (14.5, 61.2),
    "时间展开：下一时刻状态再次进入渲染与策略输入",
    color="#48505A",
    rad=0.0,
    fs=8.4,
    offset=(0, -1.9),
)

# Loss row
block(
    92,
    28.4,
    39,
    8.4,
    C_LOSS,
    [r"多目标物理驱动损失 $L_{GCGL}$", "速度跟踪 · 碰撞 · 避障 · 平滑 · 目标到达"],
    chap="第4章",
    fs=8.8,
)
arrow((106, y), (106, 36.8), "动力学量", offset=(6.0, 0.0))
arrow((130.5, y), (119, 36.8), "轨迹指标", offset=(4.8, -0.3))

# Backward/update row
block(82, 11.0, 26, 7.0, C_GCGL, ["GCGL 稳定化", "梯度范数裁剪"], chap="第4章", fs=8.9)
block(43, 11.0, 25, 7.0, C_UPDATE, [r"更新策略参数", r"$\theta \leftarrow \theta-\eta\nabla_\theta L$"], fs=8.8, tc="#2A1B35")
arrow((111, 28.4), (99, 18.0), "BPTT 梯度", color=BWD, dashed=True, rad=0.0, offset=(7.5, -0.8))
arrow((82, 14.5), (68, 14.5), "裁剪后的梯度", color=BWD, dashed=True, offset=(0, 2.0))
arrow((56, 18.0), (83, y), "作用于网络参数", color=BWD, dashed=True, rad=-0.12, offset=(-5.8, -3.2))

# Legend and note
legend_line(7, 4.2, FWD, False, "前向计算/仿真展开")
legend_line(50, 4.2, BWD, True, "损失梯度反向传播/参数更新")
ax.text(
    101,
    4.2,
    "章节标签表示论文中对应说明位置",
    ha="left",
    va="center",
    fontsize=8.8,
    color=MUTED,
)

fig.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved: {OUT}")
