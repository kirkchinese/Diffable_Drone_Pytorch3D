#!/usr/bin/env python3
"""失败轨迹分类分析脚本

读取多种子评估的per-episode CSV日志, 将每个episode分类为:
- 成功 (Success): reached_target=1 且 collision=0 全程
- 碰撞 (Collision): 任一步 collision=1
- 停滞 (Stagnation): 无碰撞, 未到达, progress < 20%
- 超时 (Timeout): 无碰撞, 未到达, progress >= 20%

输出: 失败模式分布表(CSV) + 堆叠柱状图(PNG)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

# ── 全局设置 ────────────────────────────────────────────────────
BASE_DIR = Path("viz_results/thesis_eval")
OUTPUT_DIR = Path("docs/论文相关/thesis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 需要分析的核心实验
EXPERIMENTS = {
    "exp01_baseline_mse": "Baseline",
    "exp17_goal_reaching": "GoalLoss-Only",
    "exp22_grad_clip_only": "GradClip-Only",
    "exp21_grad_clip_goal": "GCGL",
}

SEEDS = [0, 42, 123, 456, 789]
STAGNATION_THRESHOLD = 20.0  # progress_pct < 20% => stagnation

# ── 中文字体 ────────────────────────────────────────────────────
import subprocess as _sp
_fc_out = _sp.run(["fc-list", ":lang=zh", "family"], capture_output=True, text=True).stdout
_found_font = None
for _candidate in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]:
    if _candidate in _fc_out:
        _found_font = _candidate
        break
if _found_font:
    plt.rcParams["font.family"] = _found_font
    print(f"使用字体: {_found_font}")
else:
    print("警告: 未找到中文字体, 使用默认字体")
plt.rcParams["axes.unicode_minus"] = False


def classify_episode(csv_path: str) -> dict:
    """对单个episode CSV进行分类

    指标口径与论文正式定义一致:
    - reached: episode内任意时刻 reached_target==1 (对应公式4.21/4.22)
    - progress: 基于整条轨迹最佳距离 max(progress_pct) (对应公式4.25)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None

    has_collision = df["collision"].any()
    # 论文定义: 任意时刻到达目标即算reached (公式4.21)
    reached = (df["reached_target"] == 1).any() if "reached_target" in df.columns else False
    # 论文定义: 使用轨迹中最佳进度 (公式4.25: 基于min_t d_i^t)
    best_progress = df["progress_pct"].max() if "progress_pct" in df.columns else 0.0
    final_dist = df["dist_to_target"].min() if "dist_to_target" in df.columns else 999.0
    avg_speed = df["speed"].mean() if "speed" in df.columns else 0.0
    min_obs_dist = df["dist_to_obs"].min() if "dist_to_obs" in df.columns else 999.0

    # 分类
    if reached and not has_collision:
        category = "成功"
    elif has_collision:
        category = "碰撞"
    elif best_progress < STAGNATION_THRESHOLD:
        category = "停滞"
    else:
        category = "超时"

    return {
        "category": category,
        "has_collision": has_collision,
        "reached": reached,
        "best_progress": best_progress,
        "final_dist": final_dist,
        "avg_speed": avg_speed,
        "min_obs_dist": min_obs_dist,
    }


def analyze_experiment(exp_name: str) -> pd.DataFrame:
    """分析一个实验的所有seed和episode"""
    records = []
    for seed in SEEDS:
        seed_dir = BASE_DIR / exp_name / f"seed{seed}"
        if not seed_dir.exists():
            continue
        csvs = sorted(glob.glob(str(seed_dir / "episode_*_log.csv")))
        for csv_path in csvs:
            ep_num = int(Path(csv_path).stem.split("_")[1])
            result = classify_episode(csv_path)
            if result is None:
                continue
            result["experiment"] = exp_name
            result["seed"] = seed
            result["episode"] = ep_num
            records.append(result)
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  失败轨迹分类分析")
    print("=" * 60)

    all_results = []
    for exp_name, display_name in EXPERIMENTS.items():
        print(f"\n[分析] {display_name} ({exp_name})")
        df = analyze_experiment(exp_name)
        if df.empty:
            print(f"  警告: 未找到评估数据")
            continue
        df["display_name"] = display_name
        all_results.append(df)

        # 打印每experiment的统计
        counts = df["category"].value_counts()
        total = len(df)
        for cat in ["成功", "碰撞", "停滞", "超时"]:
            n = counts.get(cat, 0)
            print(f"  {cat}: {n}/{total} ({n / total * 100:.1f}%)")

    if not all_results:
        print("ERROR: 没有找到任何评估数据")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # ── 汇总表 ────────────────────────────────────────────────
    summary_rows = []
    for exp_name, display_name in EXPERIMENTS.items():
        exp_df = all_df[all_df["experiment"] == exp_name]
        if exp_df.empty:
            continue
        total = len(exp_df)
        counts = exp_df["category"].value_counts()
        row = {
            "实验": display_name,
            "总episode数": total,
            "成功": counts.get("成功", 0),
            "碰撞": counts.get("碰撞", 0),
            "停滞": counts.get("停滞", 0),
            "超时": counts.get("超时", 0),
            "成功率(%)": counts.get("成功", 0) / total * 100,
            "碰撞率(%)": counts.get("碰撞", 0) / total * 100,
            "停滞率(%)": counts.get("停滞", 0) / total * 100,
            "超时率(%)": counts.get("超时", 0) / total * 100,
        }
        # 停滞episode的额外统计
        stag = exp_df[exp_df["category"] == "停滞"]
        if len(stag) > 0:
            row["停滞avg_dist"] = stag["final_dist"].mean()
            row["停滞avg_speed"] = stag["avg_speed"].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = OUTPUT_DIR / "failure_mode_analysis.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表已保存: {csv_path}")
    print(summary_df.to_string(index=False))

    # ── 分组柱状图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))

    categories = ["成功", "碰撞", "停滞", "超时"]
    colors = {"成功": "#4CAF50", "碰撞": "#F44336", "停滞": "#FF9800", "超时": "#42A5F5"}
    display_names = [r["实验"] for r in summary_rows]

    # Add n= labels to experiment names
    x_labels = []
    for r in summary_rows:
        total = r["总episode数"]
        x_labels.append(f"{r['实验']}\n(N={total})")

    x = np.arange(len(display_names))

    bar_width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    for i, cat in enumerate(categories):
        values = [r.get(f"{cat}率(%)", 0) for r in summary_rows]
        bars = ax.bar(x + offsets[i], values, bar_width, label=cat,
                      color=colors[cat], edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, values):
            if v >= 1.5:  # only label visible bars
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylabel("各类别比例 (%)", fontsize=13)
    ax.set_title("图4.10 消融实验失败模式分布", fontsize=15, fontweight="bold", pad=20)
    # Legend well above bars with enough clearance
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc="lower center",
              ncol=4, fontsize=11, frameon=True, edgecolor="#DDDDDD",
              fancybox=True)
    ax.set_ylim(0, 95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig_path = OUTPUT_DIR / "fig4_10_failure_modes.png"
    fig.tight_layout(pad=3.0)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n分组柱状图已保存: {fig_path}")

    # ── 交叉分析: 停滞 episode 的详细特征 ──────────────────────
    print("\n" + "=" * 60)
    print("  停滞 episode 详细特征分析")
    print("=" * 60)
    for exp_name, display_name in EXPERIMENTS.items():
        stag = all_df[(all_df["experiment"] == exp_name) & (all_df["category"] == "停滞")]
        if stag.empty:
            print(f"\n{display_name}: 无停滞 episode")
            continue
        print(f"\n{display_name}: {len(stag)} 个停滞 episode")
        print(f"  平均终端距离: {stag['final_dist'].mean():.2f} m "
              f"(std={stag['final_dist'].std():.2f})")
        print(f"  平均速度: {stag['avg_speed'].mean():.3f} m/s "
              f"(std={stag['avg_speed'].std():.3f})")
        print(f"  平均最近障碍物距离: {stag['min_obs_dist'].mean():.3f} m")
        print(f"  平均最佳进度: {stag['best_progress'].mean():.1f}%")


if __name__ == "__main__":
    main()
