"""
表4.9（检查点选择偏置分析）重评估脚本
======================================
目的：用与表4.8 完全相同的 5种子×32ep 离线协议，重新生成
      SR@AR-best 与 SR@SR-best，使表4.9 与表4.8 同量纲。

做法：
- SR@AR-best：复用论文目录内固化的 5 个评估场景种子汇总
  `docs/论文相关/会议论文/data/thesis_eval_5seed/summary_5seed_aggregated.csv`。
- SR@SR-best：从训练日志定位"在线SR 最高"对应的周期检查点（每200步保存一份，取最近者），
  对该检查点跑 5 种子离线评估，聚合 mean±std。
- AR-best 步 / SR-best 步：从训练日志解析，仅用于报告。

仅使用 RTX 3080(gpu0)，复用现有 checkpoint，不重训。
输出：viz_results/checkpoint_bias_rerun/results.json
"""
import os
import re
import sys
import csv
import json
import math
import argparse
import subprocess
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "testscript"))
from multi_seed_eval import parse_summary  # 复用评估日志解析

CKPT_BASE = os.path.join(ROOT, "checkpoints/thesis")
OUT_BASE = os.path.join(ROOT, "viz_results/checkpoint_bias_rerun")
SUMMARY_CSV = os.path.join(
    ROOT,
    "docs/论文相关/会议论文/data/thesis_eval_5seed/summary_5seed_aggregated.csv",
)

# 表4.9 的 4 组消融实验：显示名 -> (实验目录名, 训练日志相对路径)
CONFIGS = {
    "Baseline-MSE":  ("exp01_baseline_mse",   "logs/thesis/exp01_baseline_mse.log"),
    "GoalLoss-Only": ("exp17_goal_reaching",  "logs/thesis/exp17_goal_reaching/train.log"),
    "GradClip-Only": ("exp22_grad_clip_only", "logs/thesis/exp22_grad_clip_only/train.log"),
    "GCGL":          ("exp21_grad_clip_goal", "logs/thesis/exp21_grad_clip_goal/train.log"),
}

SEEDS = [0, 42, 123, 456, 789]
EPISODES = 32
TIMESTEPS = 200
GPU = 0  # RTX 3080

TQDM_SR = re.compile(r"(\d+)/5000\b.*?SR=(\d+)%")
AR_BEST = re.compile(r"Best AR model.*?@\s*iter\s*(\d+)")


def parse_online_sr(log_path):
    """从 tqdm 行解析 (step, SR%) 序列；同一 step 保留最后一次。"""
    txt = open(log_path, encoding="utf-8", errors="ignore").read()
    series = {}
    for m in TQDM_SR.finditer(txt):
        series[int(m.group(1))] = float(m.group(2))
    ar_best = AR_BEST.search(txt)
    ar_best_step = int(ar_best.group(1)) if ar_best else None
    return series, ar_best_step


def available_ckpt_steps(exp_dir):
    steps = []
    for f in os.listdir(os.path.join(CKPT_BASE, exp_dir)):
        m = re.fullmatch(r"checkpoint_(\d+)\.pth", f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def find_sr_best_ckpt(series, ckpt_steps, win=40):
    """对每个已保存的周期检查点，取其邻域窗口内在线SR的均值，选最高者。"""
    best_c, best_val = None, -1.0
    for c in ckpt_steps:
        vals = [sr for s, sr in series.items() if abs(s - c) <= win]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        if avg > best_val:
            best_val, best_c = avg, c
    return best_c, best_val


def run_eval(ckpt_path, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.log")
    done = os.path.join(out_dir, "DONE")
    if os.path.exists(done) and os.path.exists(log_path):
        r = parse_summary(open(log_path, encoding="utf-8", errors="ignore").read())
        if r:
            return r
    cmd = [
        sys.executable, "visualize_eval.py",
        "--checkpoint", ckpt_path,
        "--output_dir", out_dir,
        "--num_episodes", str(EPISODES),
        "--timesteps", str(TIMESTEPS),
        "--gpu", str(GPU),
        "--random_scene", "--no_video",
        "--model_type", "bigger", "--sensor_mode", "depth",
        "--random_init_yaw", "--force_cross_map", "--enable_dynamic_obstacles",
        "--arena_range", "8.0", "--safe_clearance", "1.0", "--spawn_z_max", "3.0",
        "--seed", str(seed),
    ]
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, cwd=ROOT)
        for line in p.stdout:
            lf.write(line)
        p.wait()
    if p.returncode != 0:
        print(f"  [ERROR] seed={seed} 评估失败 (code {p.returncode})", flush=True)
        return None
    Path(done).touch()
    return parse_summary(open(log_path, encoding="utf-8", errors="ignore").read())


def agg_sr(seed_results):
    vals = [r["SR"] for r in seed_results if r and "SR" in r]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 2), "std": round(std, 2), "n": len(vals), "values": vals}


def load_ar_best_sr():
    """读取已有 summary：实验目录名 -> (SR_mean, SR_std)。"""
    out = {}
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            out[row["experiment"]] = (float(row["SR_mean"]), float(row["SR_std"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="只定位检查点，不跑评估")
    args = ap.parse_args()

    ar_best_sr = load_ar_best_sr()
    results = {}
    for name, (exp_dir, log_rel) in CONFIGS.items():
        log_path = os.path.join(ROOT, log_rel)
        series, ar_best_step = parse_online_sr(log_path)
        ckpt_steps = available_ckpt_steps(exp_dir)
        sr_best_step, sr_best_online = find_sr_best_ckpt(series, ckpt_steps)
        sr_best_ckpt = os.path.join(CKPT_BASE, exp_dir, f"checkpoint_{sr_best_step:06d}.pth")
        ar_mean, ar_std = ar_best_sr.get(exp_dir, (None, None))
        print(f"[{name}] AR-best步={ar_best_step}  SR@AR-best(已有)={ar_mean}±{ar_std}  "
              f"SR-best检查点步={sr_best_step}(在线SR≈{sr_best_online:.0f}%)", flush=True)
        results[name] = {
            "exp_dir": exp_dir, "ar_best_step": ar_best_step,
            "sr_at_ar_best": {"mean": ar_mean, "std": ar_std},
            "sr_best_step": sr_best_step, "sr_best_ckpt": sr_best_ckpt,
        }
        if args.dry_run:
            continue
        seed_res = []
        for s in SEEDS:
            out_dir = os.path.join(OUT_BASE, exp_dir, f"srbest_seed{s}")
            r = run_eval(sr_best_ckpt, out_dir, s)
            print(f"    seed={s}: SR={r.get('SR') if r else 'FAIL'}", flush=True)
            seed_res.append(r)
        results[name]["sr_at_sr_best"] = agg_sr(seed_res)

    os.makedirs(OUT_BASE, exist_ok=True)
    with open(os.path.join(OUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n==== 表4.9 重生成结果（5种子×32ep 离线，同表4.8 协议）====", flush=True)
    print(f"{'方法':16s} {'AR-best步':>9s} {'SR@AR-best':>14s} {'SR-best步':>9s} {'SR@SR-best':>14s}")
    for name, r in results.items():
        a = r["sr_at_ar_best"]
        b = r.get("sr_at_sr_best") or {}
        sa = f"{a['mean']}±{a['std']}"
        sb = f"{b.get('mean')}±{b.get('std')}"
        print(f"{name:16s} {str(r['ar_best_step']):>9s} {sa:>14s} "
              f"{str(r['sr_best_step']):>9s} {sb:>14s}", flush=True)


if __name__ == "__main__":
    main()
