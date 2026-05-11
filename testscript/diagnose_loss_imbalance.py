"""
训练崩溃诊断: 损失分量权重分析

从 metrics.csv 中分析各损失分量的加权贡献比例,
验证 "ground_affinity 主导梯度信号" 假设。
"""

import csv
import sys
import os


def load_metrics(csv_path):
    with open(csv_path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def analyze(rows, coefs):
    """分析每个迭代的损失构成比例."""
    loss_names = [
        'v', 'lateral', 'v_pred', 'collide', 'obj_avoidance',
        'd_acc', 'd_jerk', 'ground_affinity', 'drone_collide'
    ]
    
    print("=" * 110)
    print(f"{'Iter':>4} {'L':>7} {'SR%':>5} | "
          f"{'v_wt':>6} {'v%':>5} | {'ga_wt':>6} {'ga%':>5} | "
          f"{'dc_wt':>6} {'dc%':>5} | {'col_wt':>6} {'oa_wt':>6}")
    print("=" * 110)
    
    for r in rows[:50]:
        step = int(r['step'])
        L = float(r['loss'])
        sr = float(r['success_rate'])
        
        wt = {}
        for name in loss_names:
            if name not in coefs or coefs[name] == 0:
                continue
            raw = float(r.get(f'loss_{name}', 0))
            wt[name] = coefs[name] * raw
        
        wt_v = wt.get('v', 0)
        wt_ga = wt.get('ground_affinity', 0)
        wt_dc = wt.get('drone_collide', 0)
        wt_col = wt.get('collide', 0)
        wt_oa = wt.get('obj_avoidance', 0)
        
        pv = 100 * wt_v / L if L > 0 else 0
        pga = 100 * wt_ga / L if L > 0 else 0
        pdc = 100 * wt_dc / L if L > 0 else 0
        
        print(f"{step:4d} {L:7.3f} {sr*100:5.1f} | "
              f"{wt_v:6.3f} {pv:4.1f}% | {wt_ga:6.3f} {pga:4.1f}% | "
              f"{wt_dc:6.3f} {pdc:4.1f}% | {wt_col:6.3f} {wt_oa:6.3f}")
    
    # 汇总
    N = min(len(rows), 20)
    print(f"\n{'='*60}")
    print(f"前 {N} 轮平均加权损失贡献:")
    print(f"{'='*60}")
    total_avg = sum(float(rows[i]['loss']) for i in range(N)) / N
    for name in loss_names:
        c = coefs.get(name, 0)
        if c == 0:
            continue
        vals = [c * float(rows[i].get(f'loss_{name}', 0)) for i in range(N)]
        avg = sum(vals) / len(vals)
        pct = 100 * avg / total_avg if total_avg > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {name:20s}  coef={c:5.3f}  avg={avg:7.4f}  {pct:5.1f}% {bar}")
    
    # 诊断结论
    print(f"\n{'='*60}")
    print("诊断结论:")
    print(f"{'='*60}")
    
    ga_pct = sum(coefs['ground_affinity'] * float(rows[i].get('loss_ground_affinity', 0))
                 for i in range(N)) / N / total_avg * 100
    dc_pct = sum(coefs['drone_collide'] * float(rows[i].get('loss_drone_collide', 0))
                 for i in range(N)) / N / total_avg * 100
    v_pct = sum(coefs['v'] * float(rows[i].get('loss_v', 0))
                for i in range(N)) / N / total_avg * 100
    
    nav_ratio = v_pct
    noise_ratio = ga_pct + dc_pct
    
    print(f"  导航信号 (loss_v): {nav_ratio:.1f}%")
    print(f"  干扰信号 (ga+dc):  {noise_ratio:.1f}%")
    print(f"  信噪比:            {nav_ratio / noise_ratio:.2f}" if noise_ratio > 0 else "")
    
    if noise_ratio > 2 * nav_ratio:
        print(f"\n  ⚠ 警告: 干扰信号是导航信号的 {noise_ratio/nav_ratio:.1f}x 倍!")
        print(f"  → ground_affinity 贡献 {ga_pct:.1f}%, 当前实现惩罚所有 z>0 的高度")
        print(f"    参考项目 coef_ground_affinity = 0.0 (未使用)")
        print(f"  → drone_collide 贡献 {dc_pct:.1f}%, 方差大导致极端梯度")
    
    # 模拟修复后的比例
    print(f"\n{'='*60}")
    print("模拟修复后的损失比例 (ground_affinity 改为天花板式):")
    print(f"{'='*60}")
    z_ceil = 5.0
    # 修复后 ga_raw ≈ (mean_z - z_ceil).relu()^2 ≈ 0 (z < z_ceil)
    ga_fixed_raw = 0.0  # 正常飞行时几乎为 0
    new_total = total_avg - coefs['ground_affinity'] * sum(
        float(rows[i].get('loss_ground_affinity', 0)) for i in range(N)
    ) / N + coefs['ground_affinity'] * ga_fixed_raw
    
    for name in loss_names:
        c = coefs.get(name, 0)
        if c == 0:
            continue
        if name == 'ground_affinity':
            avg = c * ga_fixed_raw
        else:
            avg = sum(c * float(rows[i].get(f'loss_{name}', 0)) for i in range(N)) / N
        pct = 100 * avg / new_total if new_total > 0 else 0
        print(f"  {name:20s}  avg={avg:7.4f}  {pct:5.1f}%")
    print(f"  总损失: {total_avg:.3f} → {new_total:.3f}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        'checkpoints/trainbase_grop8_run20260321-1/metrics.csv'
    
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path}")
        sys.exit(1)
    
    coefs = {
        'v': 1.0, 'lateral': 0.5, 'v_pred': 2.0,
        'collide': 3.0, 'obj_avoidance': 2.0,
        'd_acc': 0.01, 'd_jerk': 0.001,
        'ground_affinity': 0.5, 'drone_collide': 5.0,
    }
    
    rows = load_metrics(csv_path)
    print(f"加载 {len(rows)} 行数据\n")
    analyze(rows, coefs)
