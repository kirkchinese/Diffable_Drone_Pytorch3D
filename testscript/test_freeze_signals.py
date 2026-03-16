import numpy as np

from testscript.freeze_signals import center_edge_clearance_ratio, idle_ratio


def test_center_edge_clearance_ratio_detects_edge_opening():
    """Test that center edge ratio is low when edges are more open than center."""
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    depth[:, 24:40] = 0.35
    ratio = center_edge_clearance_ratio(depth)
    assert ratio < 0.5


def test_idle_ratio_counts_low_speed_steps():
    """Test that idle_ratio computes proportion of steps below threshold."""
    speeds = np.array([0.02, 0.04, 0.30, 0.05], dtype=np.float32)
    assert abs(idle_ratio(speeds, 0.1) - 0.75) < 1e-6


def test_center_edge_clearance_ratio_returns_zero_when_edges_have_no_valid_depth():
    """Edge-only invalid input should not blow up diagnostics."""
    depth = np.zeros((48, 64), dtype=np.float32)
    depth[:, 24:40] = 2.0

    assert center_edge_clearance_ratio(depth) == 0.0


def test_center_edge_clearance_ratio_handles_tiny_images_without_empty_slice_artifacts():
    """Very small images should still yield a finite diagnostic value."""
    depth = np.full((3, 3), 1.0, dtype=np.float32)

    ratio = center_edge_clearance_ratio(depth)

    assert np.isfinite(ratio)
    assert ratio == 1.0


def test_idle_ratio_returns_zero_for_empty_speed_sequence():
    """Empty speed history should not produce NaN diagnostics."""
    ratio = idle_ratio(np.array([], dtype=np.float32), 0.1)

    assert ratio == 0.0
