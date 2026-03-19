import numpy as np
import torch

from navigation_utils import (
    compute_navigation_metrics_np,
    compute_navigation_metrics_torch,
    preprocess_depth_for_model,
)


def test_strict_success_metrics():
    target_dist = np.array([3.0, 1.2, 0.4, 0.3], dtype=np.float32)
    collision = np.array([0, 0, 0, 0], dtype=np.int32)
    metrics = compute_navigation_metrics_np(target_dist, collision, reach_radius=0.5)
    assert metrics['collision_free'] is True
    assert metrics['reached_target'] is True
    assert metrics['success'] is True
    assert abs(metrics['best_dist'] - 0.3) < 1e-6
    assert metrics['progress'] > 0.8

    collision_bad = np.array([0, 0, 1, 0], dtype=np.int32)
    metrics_bad = compute_navigation_metrics_np(target_dist, collision_bad, reach_radius=0.5)
    assert metrics_bad['collision_free'] is False
    assert metrics_bad['reached_target'] is True
    assert metrics_bad['success'] is False


def test_torch_metrics_batch():
    target_dist = torch.tensor([
        [3.0, 4.0],
        [1.0, 3.5],
        [0.4, 3.0],
    ])
    collision = torch.tensor([
        [False, False],
        [False, False],
        [False, True],
    ])
    speed = torch.tensor([
        [1.0, 0.5],
        [1.0, 0.5],
        [1.0, 0.5],
    ])
    metrics = compute_navigation_metrics_torch(target_dist, collision, speed, reach_radius=0.5)
    assert abs(float(metrics['success_rate']) - 0.5) < 1e-6
    assert abs(float(metrics['reach_rate']) - 0.5) < 1e-6
    assert abs(float(metrics['collision_free_rate']) - 0.5) < 1e-6
    assert float(metrics['goal_progress']) > 0.3
    assert float(metrics['ar']) > 0.4


def test_depth_preprocess_masks_far_pixels():
    depth = torch.tensor([
        [[-1.0, 0.5, 3.5],
         [0.3, 2.0, 10.0]]
    ])
    x = preprocess_depth_for_model(depth, depth_min=0.3, depth_max=3.0, noise_std=0.0)
    assert x.shape == (1, 1, 2, 3)
    assert float(x[0, 0, 0, 0]) == 0.0
    assert float(x[0, 0, 0, 2]) == 0.0
    assert float(x[0, 0, 1, 2]) == 0.0
    assert float(x[0, 0, 1, 0]) > float(x[0, 0, 1, 1])


if __name__ == '__main__':
    test_strict_success_metrics()
    test_torch_metrics_batch()
    test_depth_preprocess_masks_far_pixels()
    print('[PASS] navigation_utils tests passed')
