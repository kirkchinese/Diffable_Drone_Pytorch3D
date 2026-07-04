#!/usr/bin/env python3
"""批量固定种子评估所有实验 - 自动运行并汇总结果"""
import subprocess, os, sys, re, json, csv
from pathlib import Path

GPU = 1
SEED = 42
EPISODES = 32
TIMESTEPS = 200
BASE_OUT = "viz_results/formal_eval_all"
CKPT_BASE = "checkpoints/thesis"

# 每个实验的特殊参数
SPECIAL_PARAMS = {
    "exp08_sensor_lidar":  {"model_type": "lidar",  "sensor_mode": "lidar"},
    "exp09_sensor_fusion": {"model_type": "fusion", "sensor_mode": "fusion"},
    "exp10_model_attention": {"model_type": "attention"},
    "exp11_model_lightweight": {"model_type": "lightweight"},
}

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
]

def parse_summary(log_text):
    """从评估输出中提取汇总指标"""
    metrics = {}
    patterns = {
        # 使用多行锚点，避免“碰撞率”误匹配到“全程无碰撞率”这一行
        'SR': r'(?m)^\s*严格成功率 SR:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'RR': r'(?m)^\s*抵达率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'CFR': r'(?m)^\s*全程无碰撞率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'collision_rate': r'(?m)^\s*碰撞率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)',
        'avg_speed': r'(?m)^\s*平均速度:\s*([\d.]+)\s*±\s*([\d.]+)',
        'min_obs_dist': r'(?m)^\s*最小障碍距离:\s*([\d.]+)\s*m\s*\(均值\s*([\d.]+)',
        'best_target_dist': r'(?m)^\s*最佳到目标距离:\s*([\d.]+)\s*±\s*([\d.]+)',
        'final_target_dist': r'(?m)^\s*终端到目标距离:\s*([\d.]+)\s*±\s*([\d.]+)',
        'progress': r'(?m)^\s*平均完成进度:\s*([\d.]+)%',
    }
    
    for key, pat in patterns.items():
        m = re.search(pat, log_text)
        if m:
            if key in ('SR', 'RR', 'CFR', 'collision_rate'):
                metrics[key] = float(m.group(3))
                metrics[f'{key}_n'] = f"{m.group(1)}/{m.group(2)}"
            elif key in ('avg_speed', 'min_obs_dist', 'best_target_dist', 'final_target_dist'):
                metrics[key] = float(m.group(1))
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


def run_eval(exp_name):
    out_dir = f"{BASE_OUT}/{exp_name}_seed{SEED}"
    done_file = f"{out_dir}/DONE"
    
    # 检查已完成
    if os.path.exists(done_file):
        log_file = f"{out_dir}/eval.log"
        if os.path.exists(log_file):
            return load_metrics(out_dir, log_file)
        return None
    
    ckpt = f"{CKPT_BASE}/{exp_name}/best_ar.pth"
    if not os.path.exists(ckpt):
        print(f"[SKIP] {exp_name} - no checkpoint")
        return None
    
    sp = SPECIAL_PARAMS.get(exp_name, {})
    model_type = sp.get("model_type", "bigger")
    sensor_mode = sp.get("sensor_mode", "depth")
    
    cmd = [
        sys.executable, "visualize_eval.py",
        "--checkpoint", ckpt,
        "--output_dir", out_dir,
        "--num_episodes", str(EPISODES),
        "--timesteps", str(TIMESTEPS),
        "--gpu", str(GPU),
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
        "--seed", str(SEED),
    ]
    
    print(f"\n{'='*60}")
    print(f"  [{exp_name}]  model={model_type}  sensor={sensor_mode}  seed={SEED}")
    print(f"{'='*60}")
    
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/eval.log"
    
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    
    if proc.returncode != 0:
        print(f"[ERROR] {exp_name} failed with code {proc.returncode}")
        return None
    
    Path(done_file).touch()
    return load_metrics(out_dir, log_path)


def main():
    os.makedirs(BASE_OUT, exist_ok=True)
    results = {}
    
    for exp in EXPERIMENTS:
        metrics = run_eval(exp)
        if metrics:
            results[exp] = metrics
    
    # 汇总表格
    print("\n" + "="*100)
    print(f"{'实验':40s} | {'SR%':>6s} | {'RR%':>6s} | {'CFR%':>6s} | {'终端距离':>8s} | {'最佳距离':>8s} | {'均速':>6s} | {'进度%':>6s}")
    print("-"*100)
    
    # 按SR排序
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('SR', 0), reverse=True)
    for exp, m in sorted_results:
        print(f"{exp:40s} | {m.get('SR',0):6.1f} | {m.get('RR',0):6.1f} | {m.get('CFR',0):6.1f} | {m.get('final_target_dist',99):8.2f} | {m.get('best_target_dist',99):8.2f} | {m.get('avg_speed',0):6.2f} | {m.get('progress',0):6.1f}")
    print("="*100)
    
    # 保存 CSV
    csv_path = f"{BASE_OUT}/summary_seed{SEED}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'SR%', 'RR%', 'CFR%', 'final_dist', 'best_dist', 'avg_speed', 'progress%', 'SR_n', 'RR_n', 'CFR_n'])
        for exp, m in sorted_results:
            writer.writerow([exp, m.get('SR',0), m.get('RR',0), m.get('CFR',0),
                           m.get('final_target_dist',''), m.get('best_target_dist',''),
                           m.get('avg_speed',''), m.get('progress',''),
                           m.get('SR_n',''), m.get('RR_n',''), m.get('CFR_n','')])
    print(f"\n结果已保存: {csv_path}")
    
    # 保存 JSON
    json_path = f"{BASE_OUT}/summary_seed{SEED}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"详细结果: {json_path}")


if __name__ == '__main__':
    main()
