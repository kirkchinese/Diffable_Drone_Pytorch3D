"""B6 梯度范数曲线图（论文用）
=================================
读取 exp_gradnorm_baseline / exp_gradnorm_gcgl 的 metrics.csv（含 grad_norm 列），
生成两面板图：
  (上) 梯度范数时间序列（对数y轴），基线 vs GCGL，原始+平滑，并标注裁剪阈值 c=1.0；
  (下) 梯度范数分布直方图（对数x轴），凸显 GCGL 选择性削减尾部尖峰。

仅复用已有训练记录，不重训。输出 PNG 直接写入论文 figures 目录。
统计口径与 §4.4.3 文字一致（前 3000 步）：峰度 24.7→15.1、最大 34.5→26.8、>20 尖峰 10→3。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CKPT = BASE / "checkpoints/thesis"
OUT = BASE / "docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/grad_norm.png"

N = 3000          # 与正文一致：各取前 3000 步
CLIP = 1.0        # 裁剪阈值 c=1.0
C_BASE, C_GCGL = "#2196F3", "#F44336"

plt.rcParams.update({
    "font.family": "WenQuanYi Micro Hei",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})


def load(name):
    df = pd.read_csv(CKPT / name / "metrics.csv").iloc[:N]
    return df["step"].values, df["grad_norm"].values


def smooth(y, w=50):
    return pd.Series(y).rolling(w, min_periods=1).mean().values


sb, gb = load("exp_gradnorm_baseline")
sg, gg = load("exp_gradnorm_gcgl")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2),
                               gridspec_kw={"height_ratios": [2, 1]})

# 上：时间序列
ax1.semilogy(sb, gb, color=C_BASE, alpha=0.18, lw=0.5)
ax1.semilogy(sb, smooth(gb), color=C_BASE, lw=1.6, label="基线（无裁剪）")
ax1.semilogy(sg, gg, color=C_GCGL, alpha=0.18, lw=0.5)
ax1.semilogy(sg, smooth(gg), color=C_GCGL, lw=1.6, label="GCGL（裁剪+目标损失）")
ax1.axhline(CLIP, color="gray", ls="--", lw=1.1, label=f"裁剪阈值 $c={CLIP}$")
ax1.set_ylabel("梯度范数（对数尺度）")
ax1.set_xlabel("训练迭代次数")
ax1.legend(fontsize=9, loc="upper right")

# 下：分布直方图
bins = np.logspace(np.log10(0.1), np.log10(max(gb.max(), gg.max()) * 1.1), 55)
ax2.hist(gb, bins=bins, alpha=0.5, color=C_BASE,
         label=f"基线（中位 {np.median(gb):.1f}，最大 {gb.max():.1f}）")
ax2.hist(gg, bins=bins, alpha=0.5, color=C_GCGL,
         label=f"GCGL（中位 {np.median(gg):.1f}，最大 {gg.max():.1f}）")
ax2.axvline(20, color="black", ls=":", lw=1.0, label="严重尖峰阈值 (>20)")
ax2.set_xscale("log")
ax2.set_xlabel("梯度范数（对数尺度）")
ax2.set_ylabel("频次")
ax2.legend(fontsize=8.5, loc="upper right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
print(f"baseline: median={np.median(gb):.2f} max={gb.max():.2f} >20={int((gb>20).sum())}")
print(f"gcgl:     median={np.median(gg):.2f} max={gg.max():.2f} >20={int((gg>20).sum())}")
