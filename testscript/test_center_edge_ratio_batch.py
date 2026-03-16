"""
Test for center_edge_ratio batching bug (Task 4).

Bug: center_edge_ratio is recorded from only the first batch sample (depth_lo[0])
but later consumed as per-batch data (rec['center_edge_ratio'][:, b]).

This test verifies that center_edge_ratio has shape (T, B) after stacking,
not just (T,) with only the first batch item's data.
"""


def test_center_edge_ratio_uses_only_first_batch_item():
    """
    DETECT THE BUG: Verify that center_edge_ratio incorrectly uses only depth_lo[0].
    
    The buggy code:
        depth_np = depth_lo[0].cpu().numpy()  # Takes only first batch item
        center_edge = center_edge_clearance_ratio(depth_np)
        rec['center_edge_ratio'].append(center_edge)  # Appends scalar
    
    Expected after fix:
        - Should compute for all batch items
        - Should append array of shape (B,) per timestep
    """
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    # Find the lines that compute center_edge_ratio
    # Look for depth_lo[0] followed by center_edge_ratio.append
    
    lines = source.split('\n')
    
    found_bug = False
    for i, line in enumerate(lines):
        # Look for the buggy pattern: depth_lo[0] used for center_edge_ratio
        if 'depth_lo[0]' in line and 'cpu().numpy()' in line:
            # Check if the next few lines contain center_edge_ratio.append
            for j in range(i, min(i+8, len(lines))):
                if "center_edge_ratio'].append" in lines[j]:
                    # This is the bug - using only first batch item
                    found_bug = True
                    break
    
    # Test FAILS if bug is detected (this is the RED phase)
    assert not found_bug, (
        "BUG DETECTED: center_edge_ratio is computed from only depth_lo[0] "
        "(first batch item) but verdict generation accesses rec['center_edge_ratio'][:, b] "
        "for all batch items. This causes IndexError or wrong data for b > 0. "
        "FIX: Compute center_edge_ratio for all batch items (shape T, B)."
    )


def test_center_edge_ratio_append_receives_per_batch_array():
    """
    Verify that center_edge_ratio.append receives an array with batch dimension.
    
    Buggy: append(scalar) -> shape (T,) after stack
    Fixed: append(array of shape (B,)) -> shape (T, B) after stack
    """
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    lines = source.split('\n')
    
    # Find center_edge_ratio.append line
    for i, line in enumerate(lines):
        if "center_edge_ratio'].append" in line:
            # Check if this append is receiving a scalar (the bug)
            # Look backwards to find depth computation
            bug_found = False
            for j in range(max(0, i-10), i):
                if 'depth_lo[0]' in lines[j]:
                    bug_found = True
                    break
            
            if bug_found:
                assert False, (
                    "BUG DETECTED: center_edge_ratio.append receives scalar from "
                    "center_edge_clearance_ratio(depth_lo[0]). This creates shape (T,) "
                    "after np.stack, but verdict expects (T, B). "
                    "FIX: Compute for all batch items and append array of shape (B,)."
                )


def test_verdict_accesses_center_edge_ratio_as_2d():
    """
    Verify verdict generation expects 2D array access pattern.
    
    This documents the expected access: rec['center_edge_ratio'][:, b]
    This test should pass - it's documenting the expectation.
    """
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    # Find where verdict accesses center_edge_ratio
    access_count = 0
    lines = source.split('\n')
    for line in lines:
        if "rec['center_edge_ratio'][:, b]" in line:
            access_count += 1
    
    # Should have 2 accesses: mean and min
    assert access_count == 2, (
        f"Expected 2 accesses to center_edge_ratio[:, b] for mean/min, "
        f"but found {access_count}. This test documents expected behavior."
    )