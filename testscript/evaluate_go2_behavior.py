"""Three-arm, multi-seed Go2 locomotion and visual-foot-placement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.envs.go2 import Go2Env  # noqa: E402
from diffsim.go2.policy import Go2Policy  # noqa: E402
from diffsim.go2.types import Go2EnvConfig  # noqa: E402


def _load_policy(path: Path, mode: str, device: torch.device) -> Go2Policy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("sensor_mode", mode) != mode:
        raise ValueError(f"{path} is not a {mode} checkpoint")
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    hidden_size = state_dict["gru.weight_hh"].shape[1]
    policy = Go2Policy(mode, hidden_size=hidden_size).to(device)
    policy.load_state_dict(state_dict)
    return policy.eval()


@torch.no_grad()
def _episode(policy, mode, seed, speed, obstacles, steps, device):
    config = Go2EnvConfig(batch_size=1)
    env = Go2Env(config, device, seed=seed, randomize=obstacles, sensor_mode=mode)
    env.command[:] = env.command.new_tensor((speed, 0.0, 0.0))
    observation = env.observe()
    hidden = policy.initial_hidden(1, device)
    fell = False
    obstacle_collision = False
    foot_forward, obstacle_forward = [], []
    survival_steps = 0
    for step in range(steps):
        survival_steps = step + 1
        action, hidden = policy(observation, hidden)
        output = env.step(action)
        contact = env.last_contact
        nonfoot_obstacle = (
            (~contact.sample_is_foot)[None]
            & (contact.source >= 0)
            & (contact.probability > 0.5)
        )
        obstacle_collision |= bool(nonfoot_obstacle.any())
        foot_contact = contact.probability[:, contact.sample_is_foot] > 0.5
        if foot_contact.any():
            foot_x = contact.sample_world[:, contact.sample_is_foot, 0] - env.state.base_pos[:, None, 0]
            foot_forward.extend(foot_x[foot_contact].cpu().tolist())
            active = env.scene.obstacle_mask[0]
            if active.any():
                nearest = env.scene.obstacle_position[0, active, 0].amin() - env.state.base_pos[0, 0]
                obstacle_forward.extend([float(nearest)] * int(foot_contact.sum()))
        if bool(output.terminated[0]):
            fell = True
            break
        observation = output.observation
    return {
        "fell": fell,
        "obstacle_collision": obstacle_collision,
        "survival_steps": survival_steps,
        "foot_forward_m": foot_forward,
        "nearest_obstacle_forward_m": obstacle_forward,
    }


def _aggregate(rows):
    count = len(rows)
    foot_rows = [row["foot_forward_m"] for row in rows if row["foot_forward_m"]]
    obstacle_rows = [
        row["nearest_obstacle_forward_m"]
        for row in rows
        if row["nearest_obstacle_forward_m"]
    ]
    foot = np.concatenate(foot_rows) if foot_rows else np.empty(0)
    obstacle = np.concatenate(obstacle_rows) if obstacle_rows else np.empty(0)
    correlation = None
    if foot.size > 2 and foot.size == obstacle.size and np.std(obstacle) > 1e-6:
        correlation = float(np.corrcoef(foot, obstacle)[0, 1])
    return {
        "episodes": count,
        "fall_rate": sum(row["fell"] for row in rows) / count,
        "obstacle_collision_rate": sum(row["obstacle_collision"] for row in rows) / count,
        "mean_survival_steps": float(np.mean([row["survival_steps"] for row in rows])),
        "foot_obstacle_correlation": correlation,
        "foot_samples": int(foot.size),
    }


def main():
    parser = argparse.ArgumentParser()
    for mode in ("blind", "heightmap", "depth"):
        parser.add_argument(f"--{mode}", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-seeds", default="101,102,103,104,105")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    seeds = [int(value) for value in args.eval_seeds.split(",")]
    paths = {mode: getattr(args, mode) for mode in ("blind", "heightmap", "depth")}
    report = {"evaluation_seeds": seeds, "checkpoint_selection": {}, "arms": {}}
    for mode, checkpoints in paths.items():
        report["checkpoint_selection"][mode] = [str(path.resolve()) for path in checkpoints]
        tasks = {"walk_flat": [], "run_flat": [], "obstacles": []}
        for checkpoint in checkpoints:
            policy = _load_policy(checkpoint, mode, device)
            for seed in seeds:
                tasks["walk_flat"].append(
                    _episode(policy, mode, seed, 0.5, False, args.steps, device)
                )
                tasks["run_flat"].append(
                    _episode(policy, mode, seed, 1.5, False, args.steps, device)
                )
                tasks["obstacles"].append(
                    _episode(policy, mode, seed, 0.8, True, args.steps, device)
                )
        report["arms"][mode] = {name: _aggregate(rows) for name, rows in tasks.items()}

    blind_collision = report["arms"]["blind"]["obstacles"]["obstacle_collision_rate"]
    depth_collision = report["arms"]["depth"]["obstacles"]["obstacle_collision_rate"]
    improvement = 0.0 if blind_collision == 0 else 1.0 - depth_collision / blind_collision
    report["gates"] = {
        "depth_flat_walk_fall_le_10pct": report["arms"]["depth"]["walk_flat"]["fall_rate"] <= 0.10,
        "depth_flat_run_fall_le_10pct": report["arms"]["depth"]["run_flat"]["fall_rate"] <= 0.10,
        "depth_collision_reduction_vs_blind": improvement,
        "depth_collision_reduction_ge_20pct": improvement >= 0.20,
        "depth_foot_obstacle_correlation": report["arms"]["depth"]["obstacles"]["foot_obstacle_correlation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], indent=2, ensure_ascii=False))
    if args.enforce:
        if any(len(checkpoints) != 3 for checkpoints in paths.values()) or len(seeds) != 5:
            raise AssertionError("enforced gate requires 3 training seeds per arm and 5 evaluation seeds")
        required = [value for key, value in report["gates"].items() if key.endswith(("le_10pct", "ge_20pct"))]
        if not all(required):
            raise AssertionError("one or more Go2 behavior gates failed; inspect the JSON report")


if __name__ == "__main__":
    main()
