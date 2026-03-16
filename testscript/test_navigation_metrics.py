import math

from testscript.navigation_metrics import (
    classify_episode,
    compute_progress,
    detect_spinning,
    detect_stagnation,
)


def test_compute_progress_uses_initial_distance_ratio():
    assert abs(compute_progress(10.0, 4.0) - 0.6) < 1e-6


def test_compute_progress_clamps_to_valid_range():
    # Negative final_dist should not yield progress > 1.0
    assert compute_progress(10.0, -5.0) == 1.0
    # Large positive final_dist should not yield progress < 0.0
    assert compute_progress(5.0, 100.0) == 0.0


def test_safe_idle_is_not_success():
    verdict = classify_episode(
        init_dist=8.0,
        final_dist=7.9,
        min_clearance=0.8,
        collided=False,
        timed_out=True,
        stagnated=True,
        spun=False,
        goal_radius=0.5,
    )
    assert verdict["success"] is False
    assert verdict["failure_reason"] == "stagnation"


def test_goal_radius_marks_success():
    verdict = classify_episode(
        init_dist=8.0,
        final_dist=0.3,
        min_clearance=0.6,
        collided=False,
        timed_out=False,
        stagnated=False,
        spun=False,
        goal_radius=0.5,
    )
    assert verdict["success"] is True
    assert verdict["failure_reason"] is None


def test_detect_stagnation_requires_low_speed_and_low_progress():
    distances = [6.0, 5.98, 5.97, 5.96, 5.95]
    speeds = [0.04, 0.03, 0.02, 0.03, 0.04]
    assert detect_stagnation(distances, speeds, min_progress=0.1, speed_thresh=0.1, window=5)


def test_detect_spinning_flags_large_yaw_change_near_goal():
    headings = [0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0]
    distances = [0.9, 0.8, 0.7, 0.6, 0.55]
    assert detect_spinning(headings, distances, near_goal_radius=1.0, yaw_thresh=2 * math.pi - 0.1)