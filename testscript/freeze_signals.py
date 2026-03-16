import numpy as np


def center_edge_clearance_ratio(depth: np.ndarray) -> float:
    """
    Compute ratio of center region depth to edge region depth.
    
    Lower ratio means edges are more open than center (obstacle blocking center FOV).
    
    Args:
        depth: 2D numpy array of depth values (H x W)
        
    Returns:
        Ratio of center mean depth to edge mean depth
    """
    _, w = depth.shape
    center_half_width = max(1, w // 8)
    center_start = max(0, w // 2 - center_half_width)
    center_end = min(w, w // 2 + center_half_width)
    center = depth[:, center_start:center_end]
    edges = np.concatenate([depth[:, : w // 4], depth[:, -w // 4 :]], axis=1)
    center_valid = center[center > 0]
    edge_valid = edges[edges > 0]

    if center_valid.size == 0 or edge_valid.size == 0:
        return 0.0

    center_mean = float(center_valid.mean())
    edge_mean = float(edge_valid.mean())
    return center_mean / edge_mean


def idle_ratio(speeds, threshold: float) -> float:
    """
    Compute proportion of steps where speed is below threshold.
    
    Args:
        speeds: 1D array of speed values
        threshold: speed threshold below which steps are considered idle
        
    Returns:
        Ratio of idle steps (0.0 to 1.0)
    """
    speeds = np.asarray(speeds)
    if speeds.size == 0:
        return 0.0
    return float((speeds < threshold).mean())
