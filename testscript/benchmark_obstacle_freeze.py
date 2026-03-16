import argparse
import csv
import re
import subprocess
from pathlib import Path


EPISODE_LOG_GLOB = "episode_*_log.csv"


DEFAULT_CHECKPOINT_ROOT = Path("/home/misaka/Diffable_Drone_Pytorch3D/checkpoints")
FAMILIES = {
    "anti_freeze_v1",
    "anti_freeze_v2",
    "anti_freeze_v3",
    "anti_freeze_v3b",
    "decomposed_loss_v1",
    "velocity_decomp_stage1",
    "cmaes_run_01",
}


def discover_checkpoints(root: Path):
    rows = []
    for child in root.iterdir():
        if not child.is_dir() or child.name not in FAMILIES:
            continue
        for filename in ["best_ar.pth", "checkpoint_final.pth"]:
            candidate = child / filename
            if candidate.exists():
                rows.append({"family": child.name, "checkpoint": candidate})
                break
    return rows


def rank_rows(rows):
    return sorted(rows, key=lambda r: (
        -r.get("success_rate", 0),
        -r.get("mean_progress", 0),
        -r.get("min_clearance", 0),
        r.get("family", ""),
    ))


def build_eval_command(
    checkpoint: Path,
    output_dir: Path,
    *,
    visualize_script: Path = Path("visualize_eval.py"),
    gpu: int = 1,
    num_episodes: int = 8,
    timesteps: int = 200,
    goal_radius: float = 0.5,
    stagnation_window: int = 30,
    stagnation_progress: float = 0.1,
    stagnation_speed: float = 0.1,
    spin_near_goal_radius: float = 1.0,
    spin_yaw_thresh: float = 2 * 3.14159,
    conda_env: str = "pytorch",
):
    return [
        "conda", "run", "-n", conda_env, "python",
        str(visualize_script),
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--num_episodes",
        str(num_episodes),
        "--timesteps",
        str(timesteps),
        "--goal_radius",
        str(goal_radius),
        "--stagnation_window",
        str(stagnation_window),
        "--stagnation_progress",
        str(stagnation_progress),
        "--stagnation_speed",
        str(stagnation_speed),
        "--spin_near_goal_radius",
        str(spin_near_goal_radius),
        "--spin_yaw_thresh",
        str(spin_yaw_thresh),
        "--random_scene",
        "--force_cross_map",
        "--no_video",
        "--gpu",
        str(gpu),
    ]


def summarize_episode_logs(output_dir: Path, *, stagnation_speed: float):
    log_paths = sorted(output_dir.glob(EPISODE_LOG_GLOB))
    if not log_paths:
        raise FileNotFoundError(f"No episode logs found under {output_dir}")

    num_trajectories = 0
    successes = 0
    stagnations = 0
    spins = 0
    progresses = []
    clearances = []
    idle_ratios = []

    for log_path in log_paths:
        with log_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        num_trajectories += 1
        last = rows[-1]
        progresses.append(float(last["progress_pct"]) / 100.0)
        successes += int(last["verdict_success"])
        stagnations += int(last["verdict_reason"] == "stagnation")
        spins += int(last["verdict_reason"] == "spinning")
        clearances.extend(float(r["dist_to_obs"]) - float(r["margin"]) for r in rows)
        idle_steps = sum(float(r["speed"]) < stagnation_speed for r in rows)
        idle_ratios.append(idle_steps / len(rows))

    if num_trajectories == 0:
        raise ValueError(f"No non-empty episode logs found under {output_dir}")

    return {
        "num_trajectories": num_trajectories,
        "success_rate": successes / num_trajectories,
        "mean_progress": sum(progresses) / num_trajectories,
        "stagnation_rate": stagnations / num_trajectories,
        "spin_rate": spins / num_trajectories,
        "min_clearance": min(clearances),
        "idle_ratio": sum(idle_ratios) / num_trajectories,
    }


def parse_summary_text(stdout: str):
    metrics = {}
    match = re.search(
        r"Center/Edge 开口比:\s*([0-9.]+) \(均值\),\s*([0-9.]+) \(最小\)",
        stdout,
    )
    if match:
        metrics["mean_center_edge_ratio"] = float(match.group(1))
        metrics["min_center_edge_ratio"] = float(match.group(2))
    return metrics


def evaluate_checkpoint(row, *, visualize_script: Path = Path("visualize_eval.py"), **kwargs):
    output_dir = Path(kwargs.pop("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_eval_command(
        row["checkpoint"],
        output_dir,
        visualize_script=visualize_script,
        **kwargs,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    metrics = summarize_episode_logs(output_dir, stagnation_speed=kwargs["stagnation_speed"])
    metrics.update(parse_summary_text(result.stdout))
    return {
        **row,
        **metrics,
        "output_dir": output_dir,
        "command": command,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--run-eval", action="store_true", default=False)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_runs"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--stagnation-window", type=int, default=30)
    parser.add_argument("--stagnation-progress", type=float, default=0.1)
    parser.add_argument("--stagnation-speed", type=float, default=0.1)
    parser.add_argument("--spin-near-goal-radius", type=float, default=1.0)
    parser.add_argument("--spin-yaw-thresh", type=float, default=2 * 3.14159)
    args = parser.parse_args()
    rows = discover_checkpoints(args.checkpoint_root)
    if args.run_eval:
        evaluated = []
        for row in rows:
            run_dir = args.output_root / row["family"]
            evaluated.append(
                evaluate_checkpoint(
                    row,
                    output_dir=run_dir,
                    gpu=args.gpu,
                    num_episodes=args.num_episodes,
                    timesteps=args.timesteps,
                    goal_radius=args.goal_radius,
                    stagnation_window=args.stagnation_window,
                    stagnation_progress=args.stagnation_progress,
                    stagnation_speed=args.stagnation_speed,
                    spin_near_goal_radius=args.spin_near_goal_radius,
                    spin_yaw_thresh=args.spin_yaw_thresh,
                )
            )
        rows = rank_rows(evaluated)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = ["family", "checkpoint"]
        if args.run_eval:
            fieldnames += [
                "success_rate",
                "mean_progress",
                "stagnation_rate",
                "spin_rate",
                "min_clearance",
                "idle_ratio",
                "mean_center_edge_ratio",
                "min_center_edge_ratio",
                "output_dir",
            ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"family": row["family"], "checkpoint": str(row["checkpoint"])}
            if args.run_eval:
                for key in fieldnames[2:]:
                    if key in row:
                        out[key] = str(row[key]) if isinstance(row[key], Path) else row[key]
            writer.writerow(out)


if __name__ == "__main__":
    main()
