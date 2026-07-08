#!/usr/bin/env python3
"""
多种子批量评估脚本 —— 对所有论文实验在多个seed下运行评估并汇总统计结果

用法:
    python testscript/multi_seed_eval.py [--gpu 0] [--episodes 32] [--seeds 0 42 123 456 789]

输出:
    viz_results/thesis_eval/
        <exp_name>/seed<N>/        每个实验×种子的详细结果
        summary_all.csv            全部实验×种子的指标
        summary_aggregated.csv     均值±标准差汇总表
        summary_aggregated.json    同上, JSON格式
"""

import subprocess, os, sys, re, json, csv, argparse
from pathlib import Path
from collections import defaultdict
import math

# ================================================================
# 实验定义
# ================================================================

CKPT_BASE = "checkpoints/thesis"
BASE_OUT = "viz_results/thesis_eval"

# 每个实验的特殊参数
SPECIAL_PARAMS = {
    "exp08_sensor_lidar":       {"model_type": "lidar",       "sensor_mode": "lidar"},
    "exp09_sensor_fusion":      {"model_type": "fusion",      "sensor_mode": "fusion"},
    "exp10_model_attention":    {"model_type": "attention"},
    "exp11_model_lightweight":  {"model_type": "lightweight"},
}

# 全部论文实验 (包括 clip sensitivity)
EXPERIMENTS = [
    "exp01_baseline_mse",
    "exp02_loss_decomposed",
    "exp03_loss_adaptive",
    "exp04_cmaes_decay",
    "exp05_cmaes_guide",
    "exp06_cmaes_meta",
    "exp07_cmaes_lossnet",
    "exp08_sensor_lidar",
    "exp09_sensor_fusion",
    "exp10_model_attention",
    "exp11_model_lightweight",
    "exp12_baseline_adaptive_b32",
    "exp17_goal_reaching",
    "exp19_ema_mse",
    "exp21_grad_clip_goal",
    "exp22_grad_clip_only",
    "exp_clip_0p5",
    "exp_clip_2p0",
    "exp_clip_5p0",
]


def parse_summary(log_text):
    """从评估输出中提取汇总指标"""
    metrics = {}
    patterns = {
        'SR':    r'(?m)^\s*严格成功率 SR:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'RR':    r'(?m)^\s*抵达率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'CFR':   r'(?m)^\s*全程无碰撞率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'collision_rate':    r'(?m)^\s*碰撞率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'avg_speed':         r'(?m)^\s*平均速度:\s*([\d.]+)\s*±\s*([\d.]+)',
        'min_obs_dist':      r'(?m)^\s*最小障碍距离:\s*([\d.]+)\s*m\s*\(均值\s*([\d.]+)',
        'best_target_dist':  r'(?m)^\s*最佳到目标距离:\s*([\d.]+)\s*±\s*([\d.]+)',
        'final_target_dist': r'(?m)^\s*终端到目标距离:\s*([\d.]+)\s*±\s*([\d.]+)',
        'progress':          r'(?m)^\s*平均完成进度:\s*([\d.]+)%',
    }

    for key, pat in patterns.items():
        m = re.search(pat, log_text)
        if m:
            if key in ('SR', 'RR', 'CFR', 'collision_rate'):
                metrics[key] = float(m.group(3))
                metrics[f'{key}_n'] = f"{m.group(1)}/{m.group(2)}"
            elif key in ('avg_speed', 'min_obs_dist', 'best_target_dist', 'final_target_dist'):
                metrics[key] = float(m.group(1))
                if m.lastindex >= 2:
                    metrics[f'{key}_std'] = float(m.group(2))
            elif key == 'progress':
                metrics[key] = float(m.group(1))
    return metrics


def load_metrics(out_dir, log_path):
    """优先读 metrics.json（结构化，去 stdout-regex 脆弱性）；回退到日志正则解析（兼容旧 run）。"""
    mjson = os.path.join(out_dir, 'metrics.json')
    if os.path.exists(mjson):
        try:
            with open(mjson) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return parse_summary(open(log_path).read())


def run_eval(exp_name, seed, gpu, episodes, timesteps):
    """运行单个实验的单个seed评估"""
    out_dir = f"{BASE_OUT}/{exp_name}/seed{seed}"
    done_file = f"{out_dir}/DONE"

    # 检查已完成
    if os.path.exists(done_file):
        log_file = f"{out_dir}/eval.log"
        if os.path.exists(log_file):
            result = load_metrics(out_dir, log_file)
            if result:
                return result
        # DONE文件存在但解析失败, 删除重跑
        os.remove(done_file)

    ckpt = f"{CKPT_BASE}/{exp_name}/best_ar.pth"
    if not os.path.exists(ckpt):
        print(f"  [SKIP] {exp_name} - no checkpoint at {ckpt}")
        return None

    sp = SPECIAL_PARAMS.get(exp_name, {})
    model_type = sp.get("model_type", "bigger")
    sensor_mode = sp.get("sensor_mode", "depth")

    cmd = [
        sys.executable, "visualize_eval.py",
        "--checkpoint", ckpt,
        "--output_dir", out_dir,
        "--num_episodes", str(episodes),
        "--timesteps", str(timesteps),
        "--gpu", str(gpu),
        "--random_scene",
        "--no_video",
        "--model_type", model_type,
        "--sensor_mode", sensor_mode,
        "--random_init_yaw",
        "--force_cross_map",
        "--enable_dynamic_obstacles",
        "--arena_range", "8.0",
        "--safe_clearance", "1.0",
        "--spawn_z_max", "3.0",
        "--seed", str(seed),
    ]

    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/eval.log"

    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            logf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"  [ERROR] {exp_name} seed={seed} failed (code {proc.returncode})")
        return None

    Path(done_file).touch()
    return load_metrics(out_dir, log_path)


def aggregate_results(all_results):
    """将多seed结果聚合为均值±标准差"""
    aggregated = {}

    for exp_name, seed_results in all_results.items():
        if not seed_results:
            continue

        metric_keys = ['SR', 'RR', 'CFR', 'collision_rate', 'avg_speed',
                        'final_target_dist', 'best_target_dist', 'progress']

        agg = {}
        for key in metric_keys:
            values = [r[key] for r in seed_results.values() if key in r]
            if values:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    std = math.sqrt(variance)
                else:
                    std = 0.0
                agg[f'{key}_mean'] = round(mean, 2)
                agg[f'{key}_std'] = round(std, 2)
                agg[f'{key}_n'] = len(values)
                agg[f'{key}_values'] = values

        agg['n_seeds'] = len(seed_results)
        aggregated[exp_name] = agg

    return aggregated


def main():
    global CKPT_BASE, BASE_OUT

    parser = argparse.ArgumentParser(description='多种子批量评估')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--episodes', type=int, default=32, help='每个seed的episode数')
    parser.add_argument('--timesteps', type=int, default=200, help='每个episode的时间步')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 42, 123, 456, 789],
                        help='评估种子列表')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='只评估指定的实验 (默认全部)')
    parser.add_argument('--ckpt_base', default=None,
                        help=f'checkpoint 根目录 (默认 {CKPT_BASE}; '
                             f'multiseed 战役用 checkpoints/thesis_multiseed)')
    parser.add_argument('--out_base', default=None,
                        help=f'评估输出根目录 (默认 {BASE_OUT}; multiseed 请另指目录, '
                             f'勿覆盖论文 5-seed 真值)')
    args = parser.parse_args()

    if args.ckpt_base:
        CKPT_BASE = args.ckpt_base
    if args.out_base:
        BASE_OUT = args.out_base

    experiments = args.experiments if args.experiments else EXPERIMENTS
    seeds = args.seeds

    print(f"评估配置: {len(experiments)} 实验 × {len(seeds)} seeds × {args.episodes} episodes")
    print(f"GPU: {args.gpu}, Timesteps: {args.timesteps}")
    print(f"Seeds: {seeds}")
    print(f"输出目录: {BASE_OUT}")
    print()

    os.makedirs(BASE_OUT, exist_ok=True)
    all_results = defaultdict(dict)  # {exp_name: {seed: metrics}}

    total = len(experiments) * len(seeds)
    done = 0

    for exp in experiments:
        for seed in seeds:
            done += 1
            progress = f"[{done}/{total}]"
            print(f"{progress} {exp}  seed={seed}  ...", flush=True)

            metrics = run_eval(exp, seed, args.gpu, args.episodes, args.timesteps)
            if metrics:
                all_results[exp][seed] = metrics
                sr = metrics.get('SR', 0)
                rr = metrics.get('RR', 0)
                cfr = metrics.get('CFR', 0)
                print(f"  -> SR={sr:.1f}%  RR={rr:.1f}%  CFR={cfr:.1f}%")
            else:
                print(f"  -> FAILED or SKIPPED")

    # 保存逐seed详细结果
    detail_path = f"{BASE_OUT}/summary_all.csv"
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'seed', 'SR%', 'RR%', 'CFR%',
                          'collision_rate%', 'final_dist', 'best_dist',
                          'avg_speed', 'progress%'])
        for exp in experiments:
            for seed in seeds:
                m = all_results.get(exp, {}).get(seed, {})
                if m:
                    writer.writerow([
                        exp, seed,
                        m.get('SR', ''), m.get('RR', ''), m.get('CFR', ''),
                        m.get('collision_rate', ''), m.get('final_target_dist', ''),
                        m.get('best_target_dist', ''), m.get('avg_speed', ''),
                        m.get('progress', '')
                    ])

    # 聚合结果
    aggregated = aggregate_results(all_results)

    # 打印汇总表
    print("\n" + "=" * 110)
    print(f"{'实验':40s} | {'SR%':>12s} | {'RR%':>12s} | {'CFR%':>12s} | {'终端距离':>12s} | {'进度%':>12s} | N")
    print("-" * 110)

    sorted_agg = sorted(aggregated.items(),
                         key=lambda x: x[1].get('SR_mean', 0), reverse=True)
    for exp, agg in sorted_agg:
        sr = f"{agg.get('SR_mean',0):.1f}±{agg.get('SR_std',0):.1f}"
        rr = f"{agg.get('RR_mean',0):.1f}±{agg.get('RR_std',0):.1f}"
        cfr = f"{agg.get('CFR_mean',0):.1f}±{agg.get('CFR_std',0):.1f}"
        fd = f"{agg.get('final_target_dist_mean',99):.2f}±{agg.get('final_target_dist_std',0):.2f}"
        prog = f"{agg.get('progress_mean',0):.1f}±{agg.get('progress_std',0):.1f}"
        n = agg.get('n_seeds', 0)
        print(f"{exp:40s} | {sr:>12s} | {rr:>12s} | {cfr:>12s} | {fd:>12s} | {prog:>12s} | {n}")
    print("=" * 110)

    # 保存聚合CSV
    agg_csv = f"{BASE_OUT}/summary_aggregated.csv"
    with open(agg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'n_seeds',
                          'SR_mean', 'SR_std', 'RR_mean', 'RR_std',
                          'CFR_mean', 'CFR_std',
                          'final_dist_mean', 'final_dist_std',
                          'best_dist_mean', 'best_dist_std',
                          'avg_speed_mean', 'avg_speed_std',
                          'progress_mean', 'progress_std'])
        for exp, agg in sorted_agg:
            writer.writerow([
                exp, agg.get('n_seeds', 0),
                agg.get('SR_mean', ''), agg.get('SR_std', ''),
                agg.get('RR_mean', ''), agg.get('RR_std', ''),
                agg.get('CFR_mean', ''), agg.get('CFR_std', ''),
                agg.get('final_target_dist_mean', ''), agg.get('final_target_dist_std', ''),
                agg.get('best_target_dist_mean', ''), agg.get('best_target_dist_std', ''),
                agg.get('avg_speed_mean', ''), agg.get('avg_speed_std', ''),
                agg.get('progress_mean', ''), agg.get('progress_std', '')
            ])

    # 保存聚合JSON
    agg_json = f"{BASE_OUT}/summary_aggregated.json"
    with open(agg_json, 'w') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"\n逐seed详细结果: {detail_path}")
    print(f"聚合统计结果: {agg_csv}")
    print(f"JSON详细结果: {agg_json}")
    print(f"\n评估完成! 共 {sum(len(v) for v in all_results.values())} 个有效评估点")


if __name__ == '__main__':
    main()
