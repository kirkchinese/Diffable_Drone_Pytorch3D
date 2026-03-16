import math


def compute_progress(init_dist: float, final_dist: float) -> float:
    return max(0.0, min(1.0, 1.0 - final_dist / max(init_dist, 1e-6)))


def detect_stagnation(distances, speeds, *, min_progress, speed_thresh, window):
    if len(distances) < window or len(speeds) < window:
        return False
    d0 = distances[-window]
    d1 = distances[-1]
    progress = d0 - d1
    avg_speed = sum(speeds[-window:]) / window
    return progress < min_progress and avg_speed < speed_thresh


def detect_spinning(headings, distances, *, near_goal_radius, yaw_thresh):
    total = 0.0
    active = [h for h, d in zip(headings, distances) if d <= near_goal_radius]
    if len(active) < 2:
        return False
    for a, b in zip(active[:-1], active[1:]):
        diff = math.atan2(math.sin(b - a), math.cos(b - a))
        total += abs(diff)
    return total >= yaw_thresh


def classify_episode(*, init_dist, final_dist, min_clearance, collided, timed_out, stagnated, spun, goal_radius):
    success = final_dist <= goal_radius and not collided
    if success:
        reason = None
    elif collided:
        reason = "collision"
    elif stagnated:
        reason = "stagnation"
    elif spun:
        reason = "spinning"
    elif timed_out:
        reason = "timeout"
    else:
        reason = "unknown"
    return {
        "success": success,
        "failure_reason": reason,
        "progress": compute_progress(init_dist, final_dist),
        "min_clearance": min_clearance,
    }