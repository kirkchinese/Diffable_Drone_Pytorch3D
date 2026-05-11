"""
Generate two architecture diagrams for Diffable_Drone project:
  1. pipeline.png  — Differentiable Training Pipeline (horizontal flowchart)
  2. modules.png   — System Module Map (two-panel block diagram)

Style: white background, dark gray strokes, academic-paper aesthetic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import os

# ---------------------------------------------------------------------------
# Font setup – prefer Noto Sans CJK JP for Chinese glyphs, fall back to sans
# ---------------------------------------------------------------------------
FONT_NAME = None
available = {f.name for f in fm.fontManager.ttflist}
for candidate in ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "AR PL UMing CN"]:
    if candidate in available:
        FONT_NAME = candidate
        break

if FONT_NAME is not None:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [FONT_NAME, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Colour palette – academic greyscale
# ---------------------------------------------------------------------------
BG = "white"
EDGE = "#2d2d2d"
FILL_LIGHT = "#f4f4f4"
FILL_MED = "#e8e8e8"
TEXT = "#1a1a1a"
ACCENT = "#555555"
DASHED = "#666666"
CORNER_RADIUS = 6

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===================================================================
# Helper drawing functions
# ===================================================================

def draw_box(ax, x, y, w, h, text, fontsize=9, fill=FILL_LIGHT, edge=EDGE,
             lw=1.2, text_color=TEXT, bold=False, fontsize_sub=None,
             text_sub=None):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=f"round,pad=0.3,rounding_size={CORNER_RADIUS}",
                         facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    lines = text.split("\n")
    n = len(lines)
    line_h = fontsize * 1.35
    for i, line in enumerate(lines):
        ly = y + (n - 1 - i) * line_h / 2
        ax.text(x, ly, line, ha="center", va="center", fontsize=fontsize,
                color=text_color, weight=weight)
    if text_sub is not None:
        fs = fontsize_sub if fontsize_sub else fontsize - 1.5
        ax.text(x, y - h / 2 + 2, text_sub, ha="center", va="bottom",
                fontsize=fs, color=ACCENT, style="italic")


def draw_arrow(ax, x0, y0, x1, y1, color="#3a3a3a", lw=1.2, style="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                ls=style, shrinkA=0, shrinkB=0))


def draw_label(ax, x, y, text, fontsize=9, color=TEXT, ha="center",
               va="center", weight="normal", style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            weight=weight, style=style)


# ===================================================================
# FIGURE 1: Differentiable Training Pipeline
# ===================================================================

def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 4.0))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    nodes = [
        ("Random Scene\nGenerator",           1.6, None),
        ("PyTorch3D\nMeshRasterizer",         1.7, None),
        ("Depth Image\n(48x64)",             1.4, None),
        ("Policy Network\n(CNN + GRU)",       1.7, None),
        ("Thrust Command\n(a_pred, v_pred)",  1.7, None),
        ("Differentiable\nDynamics",          1.5, "Verlet + GDecay"),
        ("Task Loss",                         1.3, "speed + collision\n+ smoothness"),
    ]

    n = len(nodes)
    total_w = sum(w for _, w, _ in nodes)
    gap = 0.32
    span = total_w + gap * (n - 1)
    margin = (12.0 - span) / 2

    xs = []
    xc = margin
    for _, w, _ in nodes:
        xs.append(xc + w / 2)
        xc += w + gap

    yc = 2.1

    # Draw boxes
    for i, ((label, w, sub), x) in enumerate(zip(nodes, xs)):
        highlight = (i == 3 or i == 6)
        draw_box(ax, x, yc, w, 1.4, label, fontsize=8.5,
                 fontsize_sub=7, text_sub=sub,
                 fill=FILL_MED if highlight else FILL_LIGHT,
                 bold=highlight)

    # Arrows between boxes
    for i in range(n - 1):
        x0 = xs[i] + nodes[i][1] / 2
        x1 = xs[i + 1] - nodes[i + 1][1] / 2
        draw_arrow(ax, x0, yc, x1, yc, lw=1.2)

    # Odometry State input above Policy Network (index 3)
    ox = xs[3]
    oy = yc + 0.7 + 0.5
    draw_label(ax, ox, oy, "Odometry State (v, R, margin)",
               fontsize=7.5, color=ACCENT, style="italic")
    ax.annotate("", xy=(ox, yc + 0.7), xytext=(ox, oy - 0.28),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    # Backpropagation dashed arrow below
    bx0, bx1 = xs[-1], xs[3]
    by = yc - 0.78
    ax.annotate("", xy=(bx1, by), xytext=(bx0, by),
                arrowprops=dict(arrowstyle="->", color=DASHED, lw=1.4,
                                ls="--", connectionstyle="arc3,rad=-0.38",
                                shrinkA=3, shrinkB=3))
    draw_label(ax, (bx0 + bx1) / 2 + 0.1, by - 0.68,
               "Backpropagation through entire pipeline",
               fontsize=8, color=DASHED, style="italic")

    # Title
    draw_label(ax, 6.0, 3.75, "Differentiable Training Pipeline",
               fontsize=13, weight="bold")

    path = os.path.join(OUT_DIR, "pipeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(os.path.join(OUT_DIR, "pipeline.svg"), bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved {path}")


# ===================================================================
# FIGURE 2: System Module Map
# ===================================================================

def draw_modules():
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Left panel
    lx, lw = 0.5, 4.0
    ly, lh = 0.4, 6.7
    ax.add_patch(FancyBboxPatch((lx, ly), lw, lh,
                 boxstyle=f"round,pad=0.5,rounding_size=8",
                 facecolor="none", edgecolor=EDGE, linewidth=1.5))
    draw_label(ax, lx + lw / 2, ly + lh - 0.4, "Core Pipeline",
               fontsize=11, weight="bold")

    core = [
        ("train.py\nDroneTrainer",
         "Training loop, checkpointing, fitness eval"),
        ("model.py\n10 architectures",
         "CNN+GRU policies: attention, residual, lightweight variants"),
        ("drone_env.py\nDroneSimulator",
         "Gym environment: spawn, safety, reset logic"),
        ("drone_renderer.py\nPyTorch3D wrapper",
         "Mesh rasterisation, camera model, safety radius"),
        ("drone_dynamics.py\nVerlet + GDecay",
         "Differentiable physics step, gravity-decay controller"),
        ("loss.py\n12 loss terms",
         "Speed, collision, smoothness, auxiliary objectives"),
    ]
    mh = (lh - 1.0) / len(core) - 0.06
    for i, (title, desc) in enumerate(core):
        my = ly + lh - 0.85 - (i + 0.5) * (mh + 0.06) - i * 0.06
        draw_box(ax, lx + lw / 2, my, lw - 0.5, mh, title,
                 fontsize=7.5, fill=FILL_LIGHT, fontsize_sub=6.2, text_sub=desc)

    # Right panel
    rx, rw = 5.6, 4.3
    ry, rh = 0.4, 6.7
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh,
                 boxstyle=f"round,pad=0.5,rounding_size=8",
                 facecolor="none", edgecolor=EDGE, linewidth=1.5))
    draw_label(ax, rx + rw / 2, ry + rh - 0.4,
               "Extensions (beyond DiffPhysDrone)",
               fontsize=10.5, weight="bold")

    ext = [
        ("scene_generator.py\nSceneGenerator",
         "Procedural obstacle layout, cross-map spawn"),
        ("lidar_sensor.py\nLiDARSensor",
         "Simulated LiDAR point cloud / range image"),
        ("training_monitor.py\nTrainingMonitor",
         "Real-time loss/fitness curves and logging"),
        ("visualize_eval.py\nEvalRunner",
         "Trajectory plots, video export, eval runner"),
    ]
    eh = (rh - 1.0) / len(ext) - 0.06
    for i, (title, desc) in enumerate(ext):
        my = ry + rh - 0.85 - (i + 0.5) * (eh + 0.06) - i * 0.06
        draw_box(ax, rx + rw / 2, my, rw - 0.5, eh, title,
                 fontsize=7.5, fill=FILL_MED, fontsize_sub=6.2, text_sub=desc)

    draw_label(ax, 5.25, 7.55, "System Module Map", fontsize=13, weight="bold")

    path = os.path.join(OUT_DIR, "modules.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(os.path.join(OUT_DIR, "modules.svg"), bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    draw_pipeline()
    draw_modules()
    print("Done.")
