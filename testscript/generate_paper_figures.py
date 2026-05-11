#!/usr/bin/env python3
"""生成论文用图表 - 全部实验对比、训练曲线、消融分析"""
import json, os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT_DIR = "docs/论文相关/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_BASE = "checkpoints/thesis"
EVAL_DIR = "viz_results/formal_eval_all"

# ================================================================
# 图1: 训练曲线对比 (基线 vs 梯度裁剪)
# ================================================================
def plot_training_curves():
    """Fig 4.2: 训练过程滑动平均AR曲线"""
    experiments = {
        'exp01_baseline_mse': {'label': 'exp01 MSE Baseline', 'color': '#e74c3c', 'ls': '-'},
        'exp02_loss_decomposed': {'label': 'exp02 Decomposed', 'color': '#f39c12', 'ls': '--'},
        'exp03_loss_adaptive': {'label': 'exp03 Adaptive', 'color': '#9b59b6', 'ls': '--'},
        'exp22_grad_clip_only': {'label': 'exp22 GradClip', 'color': '#3498db', 'ls': '-'},
        'exp21_grad_clip_goal': {'label': 'exp21 GradClip+Goal (Ours)', 'color': '#2ecc71', 'ls': '-', 'lw': 2.5},
    }
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    for exp_name, cfg in experiments.items():
        csv_path = os.path.join(CKPT_BASE, exp_name, 'metrics.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'ar' not in df.columns or len(df) < 100:
            continue
        
        rolling = df['ar'].rolling(50, min_periods=10).mean()
        lw = cfg.get('lw', 1.5)
        ax.plot(range(len(rolling)), rolling, label=cfg['label'], 
                color=cfg['color'], linestyle=cfg['ls'], linewidth=lw, alpha=0.9)
    
    # 标注退化区域
    ax.axvspan(3000, 5000, alpha=0.08, color='red', label='Regression zone (no clip)')
    ax.axhline(y=0.839, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(200, 0.845, 'Baseline best_ar=0.839', fontsize=8, color='gray')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Rolling AR (window=50)', fontsize=12)
    ax.set_title('Training Curves: Late-Stage Regression vs. Gradient Clipping', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_training_curves.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_training_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] fig_training_curves")


# ================================================================
# 图2: 全实验SR柱状图排名
# ================================================================
def plot_sr_ranking():
    """Fig 4.3: 全实验严格成功率排名"""
    with open(f"{EVAL_DIR}/summary_seed42.json") as f:
        data = json.load(f)
    
    sorted_items = sorted(data.items(), key=lambda x: x[1].get('SR', 0), reverse=True)
    
    names = []
    for k, _ in sorted_items:
        short = k.replace('exp', 'E').replace('_', ' ', 1)
        if len(short) > 25:
            short = short[:25]
        names.append(short)
    
    sr = [v.get('SR', 0) for _, v in sorted_items]
    rr = [v.get('RR', 0) for _, v in sorted_items]
    cfr = [v.get('CFR', 0) for _, v in sorted_items]
    
    colors = []
    for k, _ in sorted_items:
        if k == 'exp21_grad_clip_goal':
            colors.append('#2ecc71')
        elif k == 'exp22_grad_clip_only':
            colors.append('#27ae60')
        elif k == 'exp17_goal_reaching':
            colors.append('#3498db')
        elif k == 'exp01_baseline_mse':
            colors.append('#e74c3c')
        elif 'lossnet' in k:
            colors.append('#bdc3c7')
        else:
            colors.append('#f39c12')
    
    n = len(sorted_items)
    x = np.arange(n)
    
    fig, ax = plt.subplots(figsize=(14, 5.5))
    
    bars = ax.bar(x, sr, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # 在柱上标注数值
    for i, (v, name) in enumerate(zip(sr, names)):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    
    ax.axhline(y=75.0, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(n-1, 76, 'Baseline (75.0%)', fontsize=8, color='red', ha='right')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=45, ha='right')
    ax.set_ylabel('Strict Success Rate (%)', fontsize=12)
    ax.set_title('All Experiments: Strict Success Rate Ranking (seed=42, 32 episodes)', fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_sr_ranking.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_sr_ranking.pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] fig_sr_ranking")


# ================================================================
# 图3: 消融实验 2x2 矩阵
# ================================================================
def plot_ablation():
    """Fig 4.4: 2x2 消融矩阵"""
    with open(f"{EVAL_DIR}/summary_seed42.json") as f:
        data = json.load(f)
    
    # 消融矩阵数据
    exps = {
        'No Clip\nNo Goal': data.get('exp01_baseline_mse', {}),
        'No Clip\n+ Goal': data.get('exp17_goal_reaching', {}),
        'Clip\nNo Goal': data.get('exp22_grad_clip_only', {}),
        'Clip\n+ Goal': data.get('exp21_grad_clip_goal', {}),
    }
    
    names = list(exps.keys())
    sr = [exps[n].get('SR', 0) for n in names]
    fd = [exps[n].get('final_target_dist', 3) for n in names]
    rr = [exps[n].get('RR', 0) for n in names]
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # SR
    ax = axes[0]
    bars = ax.bar(range(4), sr, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(sr):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('SR (%)')
    ax.set_title('Strict Success Rate')
    ax.set_ylim(0, 100)
    
    # Final Distance
    ax = axes[1]
    bars = ax.bar(range(4), fd, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(fd):
        ax.text(i, v + 0.05, f'{v:.2f}m', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Final Distance (m)')
    ax.set_title('Terminal Distance to Target')
    
    # Reach Rate
    ax = axes[2]
    bars = ax.bar(range(4), rr, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(rr):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('RR (%)')
    ax.set_title('Reach Rate')
    ax.set_ylim(0, 105)
    
    fig.suptitle('Ablation Study: Gradient Clipping × Goal-Reaching Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_ablation_2x2.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_ablation_2x2.pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] fig_ablation_2x2")


# ================================================================
# 图4: 多维度雷达图
# ================================================================
def plot_radar():
    """Fig 4.5: 关键实验多维度雷达图"""
    with open(f"{EVAL_DIR}/summary_seed42.json") as f:
        data = json.load(f)
    
    exps = {
        'Baseline (exp01)': data.get('exp01_baseline_mse', {}),
        'GradClip (exp22)': data.get('exp22_grad_clip_only', {}),
        'Ours (exp21)': data.get('exp21_grad_clip_goal', {}),
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # 指标: SR, RR, CFR, Progress, 1-NormDist (越高越好)
    metrics_labels = ['SR', 'RR', 'CFR', 'Progress', 'Precision']
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    for (name, m), color in zip(exps.items(), colors):
        sr = m.get('SR', 0) / 100
        rr = m.get('RR', 0) / 100
        cfr = m.get('CFR', 0) / 100
        prog = m.get('progress', 0) / 100
        # Precision: 1 - final_dist/max_dist, capped at 0
        precision = max(0, 1 - m.get('final_target_dist', 3) / 3)
        
        values = [sr, rr, cfr, prog, precision]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Multi-Dimensional Performance Comparison', fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_radar.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_radar.pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] fig_radar")


# ================================================================
# 图5: 五维度分组对比图
# ================================================================
def plot_dimension_comparison():
    """Fig 4.6: 五个实验维度分组对比"""
    with open(f"{EVAL_DIR}/summary_seed42.json") as f:
        data = json.load(f)
    
    dimensions = {
        'Loss Function': ['exp01_baseline_mse', 'exp02_loss_decomposed', 'exp03_loss_adaptive'],
        'CMA-ES': ['exp04_cmaes_decay', 'exp05_cmaes_guide', 'exp06_cmaes_meta', 'exp07_cmaes_lossnet'],
        'Sensor': ['exp01_baseline_mse', 'exp08_sensor_lidar', 'exp09_sensor_fusion'],
        'Architecture': ['exp01_baseline_mse', 'exp10_model_attention', 'exp11_model_lightweight'],
        'Stability (Ours)': ['exp01_baseline_mse', 'exp19_ema_mse', 'exp17_goal_reaching', 'exp22_grad_clip_only', 'exp21_grad_clip_goal'],
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    
    for ax, (dim_name, exp_list) in zip(axes, dimensions.items()):
        short_names = []
        sr_vals = []
        colors = []
        for exp in exp_list:
            m = data.get(exp, {})
            sr_vals.append(m.get('SR', 0))
            
            short = exp.split('_', 1)[1][:14]
            short_names.append(short)
            
            if exp == 'exp21_grad_clip_goal':
                colors.append('#2ecc71')
            elif exp == 'exp01_baseline_mse':
                colors.append('#e74c3c')
            elif 'lossnet' in exp:
                colors.append('#bdc3c7')
            else:
                colors.append('#3498db')
        
        bars = ax.bar(range(len(exp_list)), sr_vals, color=colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(sr_vals):
            ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=7.5)
        
        ax.set_xticks(range(len(exp_list)))
        ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
        ax.set_title(dim_name, fontsize=10, fontweight='bold')
        ax.axhline(y=75.0, color='red', linestyle=':', alpha=0.4)
        ax.set_ylim(0, 100)
    
    axes[0].set_ylabel('SR (%)', fontsize=11)
    fig.suptitle('Success Rate by Experimental Dimension', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_dimension_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_dimension_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] fig_dimension_comparison")


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    plot_training_curves()
    plot_sr_ranking()
    plot_ablation()
    plot_radar()
    plot_dimension_comparison()
    print(f"\nAll figures saved to: {OUT_DIR}/")
