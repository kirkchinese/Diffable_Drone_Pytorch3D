#!/usr/bin/env python3
"""梯度范数分析脚本

读取基线和GCGL实验的metrics.csv, 生成:
1. 梯度范数时间序列对比图 (基线 vs GCGL)
2. 梯度范数分布直方图 (对数坐标)
3. 时序对齐分析: 梯度范数尖峰 vs 后续AR/SR变化的描述性统计

输出: 图表(PNG) + 统计摘要(CSV)

关于因果性声明:
本脚本生成的是描述性统计数据, 仅支持"梯度裁剪确实截断了梯度尖峰"的结论,
不支持"梯度尖峰导致性能退化"的因果性结论。论文中应使用"描述性支持"而非"直接证据"。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────
BASE = Path("/home/misaka/Diffable_Drone_Pytorch3D-test1")
CKPT_DIR = BASE / "checkpoints/thesis"
OUT_DIR = BASE / "docs/论文相关/thesis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_EXP = "exp_gradnorm_baseline"
GCGL_EXP = "exp_gradnorm_gcgl"

# ── 字体设置 ──────────────────────────────────────────────────────
_CHINESE_FONT = None
for candidate in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                   "Noto Sans CJK JP", "WenQuanYi Micro Hei"]:
    try:
        resolved = findfont(FontProperties(family=candidate), fallback_to_default=False)
        _CHINESE_FONT = candidate
        break
    except ValueError:
        continue

if _CHINESE_FONT is None:
    _CHINESE_FONT = "DejaVu Sans"
    print(f"[WARN] No CJK font found, falling back to {_CHINESE_FONT}")
else:
    print(f"[INFO] Using font: {_CHINESE_FONT}")

plt.rcParams.update({
    "font.family": _CHINESE_FONT,
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.color": "#CCCCCC",
})

C_BLUE = "#2196F3"
C_RED = "#F44336"


def load_exp(name):
    path = CKPT_DIR / name / "metrics.csv"
    if not path.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    return pd.read_csv(path)


def smooth(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()


def print_grad_stats(df, label):
    """打印梯度范数统计摘要"""
    gn = df['grad_norm']
    print(f"\n{'='*50}")
    print(f"  {label} 梯度范数统计")
    print(f"{'='*50}")
    print(f"  iterations: {len(df)}")
    print(f"  mean:   {gn.mean():.2f}")
    print(f"  median: {gn.median():.2f}")
    print(f"  std:    {gn.std():.2f}")
    print(f"  min:    {gn.min():.2f}")
    print(f"  max:    {gn.max():.2f}")
    print(f"  p75:    {np.percentile(gn, 75):.2f}")
    print(f"  p90:    {np.percentile(gn, 90):.2f}")
    print(f"  p95:    {np.percentile(gn, 95):.2f}")
    print(f"  p99:    {np.percentile(gn, 99):.2f}")
    print(f"  >5:     {(gn > 5).sum()} ({(gn > 5).mean()*100:.1f}%)")
    print(f"  >10:    {(gn > 10).sum()} ({(gn > 10).mean()*100:.1f}%)")
    print(f"  >20:    {(gn > 20).sum()} ({(gn > 20).mean()*100:.1f}%)")
    print(f"  >50:    {(gn > 50).sum()} ({(gn > 50).mean()*100:.1f}%)")
    return gn


def fig_grad_norm_timeseries(df_base, df_gcgl=None):
    """图: 梯度范数时间序列对比"""
    has_gcgl = df_gcgl is not None

    fig, axes = plt.subplots(3 if has_gcgl else 2, 1,
                             figsize=(12, 9 if has_gcgl else 6),
                             gridspec_kw={'height_ratios': [2, 2, 1] if has_gcgl else [2, 1]})

    # ── 上面板: 梯度范数时间序列 (对数y轴) ──
    ax = axes[0]
    steps_b = df_base['step']
    gn_b = df_base['grad_norm']
    ax.semilogy(steps_b, gn_b, alpha=0.25, color=C_BLUE, linewidth=0.5)
    ax.semilogy(steps_b, smooth(gn_b), color=C_BLUE, linewidth=1.5, label='基线 (原始+平滑)')

    if has_gcgl:
        steps_g = df_gcgl['step']
        gn_g = df_gcgl['grad_norm']
        ax.semilogy(steps_g, gn_g, alpha=0.25, color=C_RED, linewidth=0.5)
        ax.semilogy(steps_g, smooth(gn_g), color=C_RED, linewidth=1.5, label='GCGL (原始+平滑)')
        # 裁剪阈值线
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='裁剪阈值 c=1.0')

    ax.set_ylabel('梯度范数 (对数尺度)', fontsize=11)
    ax.set_title('图4.3 梯度范数时间序列对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # ── 中面板: AR/SR曲线 ──
    ax2 = axes[1]
    ax2.plot(steps_b, smooth(df_base['ar'], 100), color=C_BLUE, linewidth=1.5, label='基线 AR')
    ax2.plot(steps_b, smooth(df_base['success_rate'], 100), color=C_BLUE, linewidth=1.5,
             linestyle='--', alpha=0.7, label='基线 SR')
    if has_gcgl:
        steps_g = df_gcgl['step']
        ax2.plot(steps_g, smooth(df_gcgl['ar'], 100), color=C_RED, linewidth=1.5, label='GCGL AR')
        ax2.plot(steps_g, smooth(df_gcgl['success_rate'], 100), color=C_RED, linewidth=1.5,
                 linestyle='--', alpha=0.7, label='GCGL SR')
    ax2.set_ylabel('AR / SR', fontsize=11)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(-0.05, 1.05)

    if has_gcgl:
        # ── 下面板: 梯度范数分布对比 (直方图) ──
        ax3 = axes[2]
        bins = np.logspace(np.log10(0.1), np.log10(max(gn_b.max(), gn_g.max()) * 1.1), 60)
        ax3.hist(gn_b, bins=bins, alpha=0.5, color=C_BLUE, label=f'基线 (median={gn_b.median():.1f})')
        ax3.hist(gn_g, bins=bins, alpha=0.5, color=C_RED, label=f'GCGL (median={gn_g.median():.1f})')
        ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax3.set_xscale('log')
        ax3.set_xlabel('梯度范数 (对数尺度)', fontsize=11)
        ax3.set_ylabel('频次', fontsize=11)
        ax3.legend(fontsize=9)
    else:
        ax3 = axes[-1]
        ax3.plot(steps_b, smooth(df_base['collision_free_rate'], 100), color='#4CAF50',
                 linewidth=1.5, label='无碰撞率')
        ax3.set_ylabel('CFR', fontsize=11)
        ax3.legend(fontsize=9)

    axes[-1].set_xlabel('训练迭代次数', fontsize=11)
    fig.tight_layout()
    return fig


def spike_analysis(df, label, threshold_percentile=95):
    """分析梯度尖峰后的性能变化 (描述性, 非因果)"""
    gn = df['grad_norm'].values
    ar = df['ar'].values
    sr = df['success_rate'].values

    threshold = np.percentile(gn, threshold_percentile)
    spike_mask = gn > threshold
    spike_indices = np.where(spike_mask)[0]

    print(f"\n{'='*50}")
    print(f"  {label}: 梯度尖峰后性能变化 (描述性统计)")
    print(f"  尖峰阈值: p{threshold_percentile} = {threshold:.2f}")
    print(f"  尖峰次数: {len(spike_indices)}")
    print(f"{'='*50}")

    if len(spike_indices) < 5:
        print("  尖峰样本不足, 跳过分析")
        return None

    # 尖峰后10步的AR变化
    ar_changes = []
    sr_changes = []
    for idx in spike_indices:
        if idx + 10 < len(ar) and idx >= 10:
            ar_before = np.mean(ar[max(0, idx-10):idx])
            ar_after = np.mean(ar[idx+1:idx+11])
            sr_before = np.mean(sr[max(0, idx-10):idx])
            sr_after = np.mean(sr[idx+1:idx+11])
            ar_changes.append(ar_after - ar_before)
            sr_changes.append(sr_after - sr_before)

    if ar_changes:
        ar_changes = np.array(ar_changes)
        sr_changes = np.array(sr_changes)
        print(f"\n  尖峰后10步 AR 变化:")
        print(f"    mean: {ar_changes.mean():.4f}")
        print(f"    std:  {ar_changes.std():.4f}")
        print(f"    负变化比例: {(ar_changes < 0).mean()*100:.1f}%")
        print(f"\n  尖峰后10步 SR 变化:")
        print(f"    mean: {sr_changes.mean():.4f}")
        print(f"    std:  {sr_changes.std():.4f}")
        print(f"    负变化比例: {(sr_changes < 0).mean()*100:.1f}%")

        # 与非尖峰步骤对比
        non_spike_mask = gn <= threshold
        non_spike_indices = np.where(non_spike_mask)[0]
        ar_changes_normal = []
        for idx in non_spike_indices[::5]:  # 采样
            if idx + 10 < len(ar) and idx >= 10:
                ar_before = np.mean(ar[max(0, idx-10):idx])
                ar_after = np.mean(ar[idx+1:idx+11])
                ar_changes_normal.append(ar_after - ar_before)
        if ar_changes_normal:
            ar_changes_normal = np.array(ar_changes_normal)
            print(f"\n  非尖峰步骤后10步 AR 变化 (对照):")
            print(f"    mean: {ar_changes_normal.mean():.4f}")
            print(f"    std:  {ar_changes_normal.std():.4f}")
            print(f"    负变化比例: {(ar_changes_normal < 0).mean()*100:.1f}%")

    return {
        'threshold': threshold,
        'n_spikes': len(spike_indices),
        'ar_change_mean': ar_changes.mean() if len(ar_changes) > 0 else None,
        'sr_change_mean': sr_changes.mean() if len(sr_changes) > 0 else None,
    }


def save_stats_csv(df_base, df_gcgl=None):
    """保存梯度范数统计到CSV"""
    rows = []
    for df, name in [(df_base, '基线'), (df_gcgl, 'GCGL')]:
        if df is None:
            continue
        gn = df['grad_norm']
        rows.append({
            '实验': name,
            '迭代数': len(df),
            '均值': f"{gn.mean():.2f}",
            '中位数': f"{gn.median():.2f}",
            '标准差': f"{gn.std():.2f}",
            '最大值': f"{gn.max():.2f}",
            'P95': f"{np.percentile(gn, 95):.2f}",
            'P99': f"{np.percentile(gn, 99):.2f}",
            '>10占比': f"{(gn > 10).mean()*100:.1f}%",
            '>20占比': f"{(gn > 20).mean()*100:.1f}%",
        })
    stats_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "grad_norm_stats.csv"
    stats_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n统计已保存: {out_path}")
    print(stats_df.to_string(index=False))


def main():
    print("=" * 60)
    print("  梯度范数分析")
    print("=" * 60)

    # 加载数据
    df_base = load_exp(BASELINE_EXP)
    df_gcgl = load_exp(GCGL_EXP)

    if df_base is None:
        print("ERROR: 基线实验数据不存在")
        return

    print_grad_stats(df_base, "基线 (无裁剪)")
    if df_gcgl is not None:
        print_grad_stats(df_gcgl, "GCGL (裁剪+目标损失)")

    # 生成时间序列图
    fig = fig_grad_norm_timeseries(df_base, df_gcgl)
    fig_path = OUT_DIR / "fig4_3_grad_norm.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n梯度范数图已保存: {fig_path}")

    # 尖峰分析 (描述性)
    spike_analysis(df_base, "基线")
    if df_gcgl is not None:
        spike_analysis(df_gcgl, "GCGL")

    # 统计CSV
    save_stats_csv(df_base, df_gcgl)


if __name__ == "__main__":
    main()
