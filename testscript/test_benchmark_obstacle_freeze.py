from pathlib import Path
import csv

from testscript.benchmark_obstacle_freeze import (
    build_eval_command,
    discover_checkpoints,
    parse_summary_text,
    rank_rows,
    summarize_episode_logs,
)


def test_discover_checkpoints_filters_known_families(tmp_path: Path):
    for name in ["anti_freeze_v1", "decomposed_loss_v1", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "anti_freeze_v1" / "best_ar.pth").write_text("x")
    (tmp_path / "decomposed_loss_v1" / "checkpoint_final.pth").write_text("x")

    rows = discover_checkpoints(tmp_path)
    names = {row["family"] for row in rows}
    assert "anti_freeze_v1" in names
    assert "decomposed_loss_v1" in names
    assert "notes" not in names


def test_rank_rows_prefers_success_then_progress_then_clearance():
    rows = [
        {"family": "a", "success_rate": 0.8, "mean_progress": 0.4, "min_clearance": 0.2},
        {"family": "b", "success_rate": 0.6, "mean_progress": 0.9, "min_clearance": 0.5},
    ]
    ranked = rank_rows(rows)
    assert ranked[0]["family"] == "a"


def test_rank_rows_handles_discovery_output(tmp_path: Path):
    """rank_rows must not KeyError on raw discover_checkpoints() output."""
    # Simulate raw discovery output - only family and checkpoint keys
    rows = [
        {"family": "decomposed_loss_v1", "checkpoint": tmp_path / "a.pth"},
        {"family": "anti_freeze_v1", "checkpoint": tmp_path / "b.pth"},
        {"family": "anti_freeze_v3", "checkpoint": tmp_path / "c.pth"},
    ]
    # Should not raise KeyError
    ranked = rank_rows(rows)
    # Ordering should be deterministic (by family name as tiebreak)
    families = [r["family"] for r in ranked]
    assert families == sorted(families)


def test_rank_rows_deterministic_tiebreak():
    """rank_rows must produce deterministic ordering even with same scores."""
    # Rows with same scores - should tiebreak by family name
    rows = [
        {"family": "zoo", "success_rate": 0.5, "mean_progress": 0.5, "min_clearance": 0.5},
        {"family": "alpha", "success_rate": 0.5, "mean_progress": 0.5, "min_clearance": 0.5},
        {"family": "mike", "success_rate": 0.5, "mean_progress": 0.5, "min_clearance": 0.5},
    ]
    ranked = rank_rows(rows)
    families = [r["family"] for r in ranked]
    assert families == sorted(families), f"Expected sorted, got {families}"


def test_build_eval_command_includes_corrected_metric_flags(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    output_dir = tmp_path / "out"
    visualize_script = tmp_path / "visualize_eval.py"

    command = build_eval_command(
        checkpoint,
        output_dir,
        visualize_script=visualize_script,
        gpu=1,
        num_episodes=8,
        timesteps=200,
        goal_radius=0.5,
        stagnation_window=30,
        stagnation_progress=0.1,
        stagnation_speed=0.1,
        spin_near_goal_radius=1.0,
        spin_yaw_thresh=6.28318,
    )

    joined = " ".join(command)
    assert str(checkpoint) in joined
    assert str(output_dir) in joined
    for flag in [
        "--checkpoint",
        "--output_dir",
        "--num_episodes",
        "--timesteps",
        "--goal_radius",
        "--stagnation_window",
        "--stagnation_progress",
        "--stagnation_speed",
        "--spin_near_goal_radius",
        "--spin_yaw_thresh",
        "--random_scene",
        "--force_cross_map",
        "--no_video",
        "--gpu",
    ]:
        assert flag in command


def test_summarize_episode_logs_aggregates_checkpoint_metrics(tmp_path: Path):
    fieldnames = [
        "step",
        "speed",
        "dist_to_obs",
        "margin",
        "progress_pct",
        "verdict_success",
        "verdict_reason",
    ]
    rows_by_file = {
        "episode_000_log.csv": [
            {"step": "0", "speed": "0.05", "dist_to_obs": "1.2", "margin": "0.3", "progress_pct": "20.0", "verdict_success": "1", "verdict_reason": "None"},
            {"step": "1", "speed": "0.05", "dist_to_obs": "1.0", "margin": "0.3", "progress_pct": "60.0", "verdict_success": "1", "verdict_reason": "None"},
        ],
        "episode_001_log.csv": [
            {"step": "0", "speed": "0.02", "dist_to_obs": "0.8", "margin": "0.3", "progress_pct": "10.0", "verdict_success": "0", "verdict_reason": "stagnation"},
            {"step": "1", "speed": "0.12", "dist_to_obs": "0.7", "margin": "0.3", "progress_pct": "15.0", "verdict_success": "0", "verdict_reason": "stagnation"},
            {"step": "2", "speed": "0.08", "dist_to_obs": "0.6", "margin": "0.3", "progress_pct": "15.0", "verdict_success": "0", "verdict_reason": "stagnation"},
        ],
    }

    for filename, rows in rows_by_file.items():
        path = tmp_path / filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metrics = summarize_episode_logs(tmp_path, stagnation_speed=0.1)

    assert metrics["num_trajectories"] == 2
    assert metrics["success_rate"] == 0.5
    assert abs(metrics["mean_progress"] - 0.375) < 1e-6
    assert metrics["stagnation_rate"] == 0.5
    assert metrics["spin_rate"] == 0.0
    assert abs(metrics["min_clearance"] - 0.3) < 1e-6
    assert abs(metrics["idle_ratio"] - ((1.0 + (2 / 3)) / 2)) < 1e-6


def test_parse_summary_text_extracts_center_edge_metrics():
    summary = """
    Freeze 诊断:
      Idle 比率 (speed < 0.1): 83.33% (均值)
      Center/Edge 开口比: 0.650 (均值), 0.400 (最小)
    """

    metrics = parse_summary_text(summary)

    assert metrics["mean_center_edge_ratio"] == 0.65
    assert metrics["min_center_edge_ratio"] == 0.4


def test_build_eval_command_uses_conda_runtime(tmp_path: Path):
    """build_eval_command must use conda runtime to access pytorch3d."""
    checkpoint = tmp_path / "model.pth"
    output_dir = tmp_path / "out"

    command = build_eval_command(
        checkpoint,
        output_dir,
        gpu=1,
        num_episodes=1,
        timesteps=20,
    )

    # Command must use conda run to access pytorch3d in the pytorch environment
    assert "conda" in command, f"Expected 'conda' in command, got {command}"
    assert "run" in command, f"Expected 'run' in command, got {command}"
    assert "-n" in command, f"Expected '-n' in command, got {command}"
    # Verify 'pytorch' environment is specified
    idx = command.index("-n")
    assert command[idx + 1] == "pytorch", f"Expected 'pytorch' env, got {command}"
    # python should come after conda run
    python_idx = command.index("python")
    conda_idx = command.index("conda")
    assert conda_idx < python_idx, f"conda should come before python in {command}"
