#!/usr/bin/env python3
"""生成全实验对比图表"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "viz_results/formal_eval_all"
with open(f"{OUT_DIR}/summary_seed42.json") as f:
    data = json.load(f)

# 按 SR 排序
sorted_items = sorted(data.items(), key=lambda x: x[1].get('SR', 0), reverse=True)
names = [k.replace('exp', 'E').replace('_', '\n', 1).replace('_', ' ') for k, _ in sorted_items]
short_names = []
for k, _ in sorted_items:
    parts = k.split('_', 1)
    short_names.append(parts[0].replace('exp', 'E') + '\n' + parts[1][:18])

sr  = [v.get('SR', 0) for _, v in sorted_items]
rr  = [v.get('RR', 0) for _, v in sorted_items]
cfr = [v.get('CFR', 0) for _, v in sorted_items]
fd  = [v.get('final_target_dist', 0) for _, v in sorted_items]
bd  = [v.get('best_target_dist', 0) for _, v in sorted_items]
prog= [v.get('progress', 0) for _, v in sorted_items]

n = len(sorted_items)
x = np.arange(n)

# 颜色区分
colors = []
for k, _ in sorted_items:
    if 'grad_clip' in k:
        colors.append('#2ecc71')   # 绿色 - 创新方法
    elif 'goal_reaching' in k:
        colors.append('#27ae60')   # 深绿 - 创新方法
    elif 'ema' in k:
        colors.append('#3498db')   # 蓝色 - 创新方法
    elif 'baseline_mse' in k:
        colors.append('#e74c3c')   # 红色 - 基线
    elif 'lossnet' in k:
        colors.append('#95a5a6')   # 灰色 - 失败
    else:
        colors.append('#f39c12')   # 橙色 - 其他

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('All Experiments Evaluation (seed=42, 32 episodes)', fontsize=16, fontweight='bold')

# 1) SR / RR / CFR
ax = axes[0, 0]
w = 0.25
ax.bar(x - w, sr, w, label='SR (Strict Success)', color=[c for c in colors], alpha=0.9, edgecolor='black', linewidth=0.5)
ax.bar(x, rr, w, label='RR (Reach Rate)', color=[c for c in colors], alpha=0.5, edgecolor='black', linewidth=0.5)
ax.bar(x + w, cfr, w, label='CFR (Collision-Free)', color=[c for c in colors], alpha=0.3, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Rate (%)')
ax.set_title('Success / Reach / Collision-Free Rates')
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
for i, v in enumerate(sr):
    ax.text(i - w, v + 1, f'{v:.0f}%', ha='center', fontsize=6, fontweight='bold')

# 2) Final distance
ax = axes[0, 1]
bars = ax.bar(x, fd, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Distance (m)')
ax.set_title('Final Distance to Target (lower is better)')
for i, v in enumerate(fd):
    ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=7)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Reach threshold (0.5m)')
ax.legend(fontsize=8)

# 3) Progress
ax = axes[1, 0]
bars = ax.bar(x, prog, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Progress (%)')
ax.set_title('Average Task Completion Progress')
ax.set_ylim(80, 101)
for i, v in enumerate(prog):
    ax.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=7)

# 4) Ranking summary
ax = axes[1, 1]
ax.axis('off')
header = ['Rank', 'Experiment', 'SR%', 'RR%', 'CFR%', 'FinalDist', 'Progress%']
table_data = []
for rank, (k, v) in enumerate(sorted_items, 1):
    table_data.append([
        str(rank),
        k,
        f"{v.get('SR',0):.1f}",
        f"{v.get('RR',0):.1f}",
        f"{v.get('CFR',0):.1f}",
        f"{v.get('final_target_dist',0):.2f}m",
        f"{v.get('progress',0):.1f}"
    ])

table = ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.3)

# 高亮前3行
for col in range(len(header)):
    table[1, col].set_facecolor('#d5f5e3')
    table[2, col].set_facecolor('#d5f5e3')
    table[3, col].set_facecolor('#d5f5e3')
    # 最后一行（失败）
    table[len(table_data), col].set_facecolor('#fadbd8')

ax.set_title('Ranking Summary (sorted by SR)', pad=20)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/all_experiments_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/all_experiments_comparison.pdf', bbox_inches='tight')
print(f"Saved: {OUT_DIR}/all_experiments_comparison.png")
print(f"Saved: {OUT_DIR}/all_experiments_comparison.pdf")
