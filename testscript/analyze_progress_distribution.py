#!/usr/bin/env python3
"""
Analyze the progress distribution of all failed episodes across 4 experiments.
Generates a histogram figure for thesis use.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz_results", "thesis_eval")
EXPERIMENTS = {
    "exp01_baseline_mse": "Exp01 Baseline MSE",
    "exp17_goal_reaching": "Exp17 Goal Reaching",
    "exp21_grad_clip_goal": "Exp21 GradClip+Goal",
    "exp22_grad_clip_only": "Exp22 GradClip Only",
}
SEEDS = [0, 42, 123, 456, 789]
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "论文相关", "thesis", "figures", "fig_progress_distribution.png")

# ---------------------------------------------------------------------------
# Font setup – prefer Noto Sans CJK SC for Chinese characters
# ---------------------------------------------------------------------------
zh_font = None
for candidate in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                   "Noto Serif CJK SC", "WenQuanYi Micro Hei"]:
    matches = font_manager.findSystemFonts()
    for fp in font_manager.findSystemFonts():
        try:
            prop = font_manager.FontProperties(fname=fp)
            if candidate.lower() in prop.get_name().lower():
                zh_font = prop.get_name()
                break
        except Exception:
            continue
    if zh_font:
        break

if zh_font:
    plt.rcParams['font.sans-serif'] = [zh_font, 'DejaVu Sans']
    print(f"[INFO] Using Chinese font: {zh_font}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("[WARN] No Chinese font found, falling back to DejaVu Sans")

plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
records = []  # list of dicts: {exp, seed, episode, progress_pct, failure_reason}

for exp_key, exp_label in EXPERIMENTS.items():
    for seed in SEEDS:
        seed_dir = os.path.join(BASE_DIR, exp_key, f"seed{seed}")
        if not os.path.isdir(seed_dir):
            print(f"[WARN] Missing directory: {seed_dir}")
            continue
        csv_files = sorted(glob.glob(os.path.join(seed_dir, "episode_*_log.csv")))
        for csv_path in csv_files:
            ep_name = os.path.basename(csv_path)
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[ERR] Failed to read {csv_path}: {e}")
                continue

            # 论文定义: 任意时刻到达即算reached (公式4.21)
            reached = (df["reached_target"] == 1).any()
            has_collision = df["collision"].any()
            success = reached and (not has_collision)

            if not success:
                # 论文定义: 使用轨迹中最佳进度 (公式4.25)
                progress = float(df["progress_pct"].max())
                reason = "collision" if has_collision else "timeout"
                records.append({
                    "exp_key": exp_key,
                    "exp_label": exp_label,
                    "seed": seed,
                    "episode": ep_name,
                    "progress_pct": progress,
                    "failure_reason": reason,
                })

fail_df = pd.DataFrame(records)
print(f"\n{'='*60}")
print(f"Total failed episodes: {len(fail_df)}")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# Print statistics
# ---------------------------------------------------------------------------
bins_stats = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
              (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

print("\n--- Progress distribution (all failed episodes) ---")
for lo, hi in bins_stats:
    if hi == 100:
        count = len(fail_df[(fail_df["progress_pct"] >= lo) & (fail_df["progress_pct"] <= hi)])
    else:
        count = len(fail_df[(fail_df["progress_pct"] >= lo) & (fail_df["progress_pct"] < hi)])
    print(f"  {lo:3d}% - {hi:3d}%: {count:4d} episodes")

print("\n--- Median progress of failed episodes per experiment ---")
for exp_key, exp_label in EXPERIMENTS.items():
    subset = fail_df[fail_df["exp_key"] == exp_key]
    if len(subset) > 0:
        med = subset["progress_pct"].median()
        print(f"  {exp_label:30s}: median = {med:6.2f}%  (n={len(subset)})")
    else:
        print(f"  {exp_label:30s}: no failed episodes")

print("\n--- Failure reason breakdown ---")
for exp_key, exp_label in EXPERIMENTS.items():
    subset = fail_df[fail_df["exp_key"] == exp_key]
    n_collision = len(subset[subset["failure_reason"] == "collision"])
    n_timeout = len(subset[subset["failure_reason"] == "timeout"])
    print(f"  {exp_label:30s}: collision={n_collision}, timeout={n_timeout}")

# ---------------------------------------------------------------------------
# Plot: beeswarm + box overlay with KDE density curve
# ---------------------------------------------------------------------------
from scipy.stats import gaussian_kde

rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.3]})

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
exp_keys = list(EXPERIMENTS.keys())
exp_labels_short = ["Baseline", "GoalLoss", "GCGL", "GradClip"]

# --- Left panel: aggregate beeswarm + box + KDE density ---
ax1 = axes[0]
all_progress = fail_df["progress_pct"].values
jitter_x = rng.uniform(-0.22, 0.22, size=len(all_progress))

# KDE density curve (rotated to vertical)
kde = gaussian_kde(all_progress, bw_method=0.15)
y_grid = np.linspace(-5, 105, 300)
density = kde(y_grid)
density_scaled = density / density.max() * 0.55
ax1.fill_betweenx(y_grid, -density_scaled, 0, alpha=0.25, color="#4C72B0", zorder=1)
ax1.plot(-density_scaled, y_grid, color="#4C72B0", lw=2.2, alpha=0.5, zorder=1)

# Scatter (beeswarm)
ax1.scatter(jitter_x, all_progress, alpha=0.50, s=16, color="#4C72B0",
            edgecolors="white", linewidth=0.3, zorder=2)

# Box overlay
bp1 = ax1.boxplot(all_progress, positions=[0.05], widths=0.30,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color="black", linewidth=2.0),
                  boxprops=dict(facecolor="#4C72B0", alpha=0.30),
                  zorder=3)

ax1.axhline(y=20, color="red", linestyle="--", linewidth=2.0, label="停滞阈值 (20%)")
ax1.set_xticks([])
ax1.set_ylabel("Best-Over-Trajectory 进度 (%)", fontsize=13)
ax1.set_title("合计失败episode分布 (N={})".format(len(fail_df)), fontsize=15, fontweight="bold")
ax1.legend(fontsize=11, loc="upper left", framealpha=0.85)
ax1.set_ylim(-5, 105)
ax1.set_xlim(-0.85, 0.85)

# Annotations: stagnation and high-progress bins
n_stagnant = len(fail_df[fail_df["progress_pct"] < 20])
pct_stagnant = 100.0 * n_stagnant / len(fail_df) if len(fail_df) > 0 else 0
n_high = len(fail_df[fail_df["progress_pct"] > 90])
pct_high = 100.0 * n_high / len(fail_df) if len(fail_df) > 0 else 0
ax1.text(0.97, 0.95,
         f"< 20%: {n_stagnant} ({pct_stagnant:.1f}%)\n> 90%: {n_high} ({pct_high:.1f}%)",
         transform=ax1.transAxes, fontsize=10.5, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.80))

# --- Right panel: per-experiment beeswarms + boxes ---
ax2 = axes[1]
positions = np.arange(len(exp_keys))

for i, (exp_key, exp_label) in enumerate(EXPERIMENTS.items()):
    vals = fail_df[fail_df["exp_key"] == exp_key]["progress_pct"].values
    if len(vals) == 0:
        continue
    jx = rng.uniform(-0.22, 0.22, size=len(vals))
    ax2.scatter(i + jx, vals, alpha=0.50, s=16, color=colors[i],
                edgecolors="white", linewidth=0.3, zorder=2)

    bp = ax2.boxplot(vals, positions=[i], widths=0.45,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=2.0),
                     boxprops=dict(facecolor=colors[i], alpha=0.30),
                     zorder=3)

    # Add n= label below each group
    ax2.text(i, -3, f"n={len(vals)}", ha="center", fontsize=9.5, color="#555")

ax2.axhline(y=20, color="red", linestyle="--", linewidth=2.0, label="停滞阈值 (20%)")
ax2.set_xticks(positions)
ax2.set_xticklabels(exp_labels_short, fontsize=12)
ax2.set_ylabel("Best-Over-Trajectory 进度 (%)", fontsize=13)
ax2.set_title("各实验失败episode进度对比", fontsize=15, fontweight="bold")
ax2.legend(fontsize=11, loc="upper left", framealpha=0.85)
ax2.set_ylim(-5, 105)
ax2.set_xlim(-0.8, len(exp_keys) - 1 + 0.8)

fig.tight_layout(pad=2.0)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"\n[OK] Figure saved to: {OUT_PATH}")
plt.close(fig)
