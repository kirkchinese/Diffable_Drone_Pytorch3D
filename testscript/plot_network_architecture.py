#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw thesis Figure 4.1: Model_bigger policy network architecture.

The content follows model.py::Model_bigger and chapter4.tex:
  depth 1x48x64 -> 4 conv blocks -> 3072 -> FC256
  state R10 -> FC256
  elementwise addition + LeakyReLU -> GRUCell(256) -> FC6 -> reshape 3x2

Output:
  docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/image3.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/image3.png"

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
MUTED = "#606A78"

C_DEPTH = "#DDEAF7"
C_CNN = "#4C72B0"
C_STATE = "#E8EEF7"
C_PROJ = "#B07AA1"
C_FUSE = "#DD8452"
C_GRU = "#55A868"
C_DEC = "#8172B3"
C_OUT = "#C44E52"
C_GROUP = "#F6F7F9"


fig, ax = plt.subplots(figsize=(15.5, 7.2))
ax.set_xlim(0, 155)
ax.set_ylim(0, 72)
ax.axis("off")


def group(x, y, w, h, title, title_dx=2, title_dy=3.1, title_y=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=1.6",
            linewidth=0.9,
            edgecolor="#D7DCE2",
            facecolor=C_GROUP,
        )
    )
    ax.text(
        x + title_dx,
        y + h - title_dy if title_y is None else title_y,
        title,
        ha="left",
        va="center",
        fontsize=8.7,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.10", fc=C_GROUP, ec="none", alpha=0.94),
    )


def box(x, y, w, h, color, lines, fs=9.8, tc="white", lw=1.15):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.32,rounding_size=1.5",
            linewidth=lw,
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


def small_stage(x, y, w, h, title, detail, color=C_CNN):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.24,rounding_size=1.1",
            linewidth=1.0,
            edgecolor=EDGE,
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=8.8, color="white")
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=7.5, color="white")


def arrow(p1, p2, text="", color=EDGE, dashed=False, rad=0.0, fs=8.3, text_offset=(0, 1.5)):
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
    if text:
        ax.text(
            (p1[0] + p2[0]) / 2 + text_offset[0],
            (p1[1] + p2[1]) / 2 + text_offset[1],
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
        )


# Functional groups
group(3, 46, 89, 20, "① 深度视觉特征提取模块（§4.2.1）")
group(35, 14.2, 60, 30.8, "② 状态向量融合模块（§4.2.2）")
group(101, 23, 21, 27, "③ GRU 时序记忆模块（§4.2.3）", title_y=27.0)
group(126, 23, 26, 27, "④ 动作解码模块（§4.2.4）")

# Depth branch
box(6, 51.8, 13, 10, C_DEPTH, ["深度图", r"$1\times48\times64$"], fs=9.5, tc="#16233A")
small_stage(24, 53.2, 8.8, 8, "Conv1", r"$32\times24\times32$")
small_stage(35, 53.2, 8.8, 8, "Conv2", r"$64\times12\times16$")
small_stage(46, 53.2, 8.8, 8, "Conv3", r"$128\times6\times8$")
small_stage(57, 53.2, 8.8, 8, "Conv4", r"$256\times3\times4$")
box(24, 46.8, 42, 4.7, C_CNN, [r"每层 Conv $3{\times}3$, stride=2, padding=1, LeakyReLU"], fs=8.3)
box(57, 61.5, 9.2, 4.5, "#6A8FC1", ["Flatten", "3072"], fs=8.0)
box(72.8, 52.5, 11.8, 8, "#6A8FC1", ["FC", r"$3072\to256$"], fs=9)

arrow((19, 56.8), (24, 57.2))
arrow((32.8, 57.2), (35, 57.2))
arrow((43.8, 57.2), (46, 57.2))
arrow((54.8, 57.2), (57, 57.2))
arrow((61.6, 61.2), (61.6, 61.5))
arrow((66.2, 63.7), (72.8, 56.5), r"$f_{img}$", text_offset=(0.7, 2.6))

# State branch
box(
    6,
    16.5,
    24,
    10.5,
    C_STATE,
    ["状态向量", r"$s_t\in\mathbb{R}^{10}$", "速度/目标速度/重力投影/安全边距"],
    fs=8.6,
    tc="#16233A",
)
box(39, 17.8, 15, 8, C_PROJ, ["状态投影", r"FC $10\to256$"], fs=9.2)
arrow((30, 21.8), (39, 21.8), r"$s_t$")

# Fusion
box(
    77,
    34.8,
    16,
    10,
    C_FUSE,
    [r"$f_{fused}$", r"$=\mathrm{LeakyReLU}$", r"$(f_{img}+W_vs_t)$"],
    fs=9.0,
)
arrow((84.6, 52.5), (85, 44.8), "视觉特征", text_offset=(7.0, -2.4))
arrow((54, 21.8), (77, 36.6), "状态特征", text_offset=(-0.8, -3.0))

# GRU and hidden-state recurrence
box(103.5, 34.8, 16, 10, C_GRU, ["GRU Cell", r"$256\to256$", r"输出 $h_t$"], fs=9.4)
arrow((93, 39.8), (103.5, 39.8), r"$x_t=f_{fused}$")
arrow(
    (107.3, 45.0),
    (116.0, 45.0),
    r"$h_{t-1}$",
    color="#2F6B46",
    dashed=True,
    rad=-0.95,
    text_offset=(0, 6.3),
)

# Decoder and output
box(128.5, 35.2, 13.5, 9.2, C_DEC, ["FC", r"$256\to6$"], fs=9.5)
box(143.8, 35.2, 7.8, 9.2, "#9A83C4", ["reshape", r"$3\times2$"], fs=8.3)
box(129.5, 24.8, 21.0, 6.5, C_OUT, [r"输出：$a_{pred}$ 与 $v_{pred}$"], fs=9.2)
arrow((119.5, 39.8), (128.5, 39.8), r"$h_t$")
arrow((142.0, 39.8), (143.8, 39.8))
arrow((147.7, 35.2), (147.7, 31.3))

# Formula callout
ax.add_patch(
    FancyBboxPatch(
        (97.5, 11.0),
        54,
        7.0,
        boxstyle="round,pad=0.25,rounding_size=1.3",
        linewidth=1.0,
        edgecolor="#D8C6C8",
        facecolor="#FFF4F4",
    )
)
ax.text(
    124.5,
    14.5,
    r"控制命令：$a_{cmd}=a_{pred}-v_{pred}-g\,k_{thr}+g$",
    ha="center",
    va="center",
    fontsize=9.2,
    color="#7A2E33",
)
arrow((140, 24.8), (126, 18.0), "", color="#7A2E33", rad=0.1)

# Compact notes
ax.text(6, 6.2, "注：视觉特征与状态特征均映射为 256 维后相加，融合特征输入 GRU 进行时序建模。", ha="left", va="center", fontsize=8.6, color=MUTED)
ax.text(150.5, 65.3, "总参数量 ≈ 1.57M", ha="right", va="center", fontsize=9.0, color=MUTED)

fig.tight_layout(pad=0.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved: {OUT}")
