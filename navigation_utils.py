import numpy as np
import torch

_FALLBACK_FWD = None  # 缓存的 fallback 向量（延迟初始化，第一次调用时创建）


def compute_local_frame(R: torch.Tensor) -> torch.Tensor:
    """根据机体姿态构造与训练一致的局部水平坐标系。"""
    global _FALLBACK_FWD
    fwd = R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    if _FALLBACK_FWD is None or _FALLBACK_FWD.device != fwd.device or _FALLBACK_FWD.dtype != fwd.dtype:
        _FALLBACK_FWD = torch.tensor([1.0, 0.0, 0.0], device=fwd.device, dtype=fwd.dtype)
    fallback = _FALLBACK_FWD.expand_as(fwd)
    fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
    fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8), fallback)
    left = torch.linalg.cross(up, fwd)
    return torch.stack([fwd, left, up], -1)


def preprocess_depth_for_model(
    depth: torch.Tensor,
    depth_min: float,
    depth_max: float,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """将渲染深度图转换为训练/推理共用的模型输入。"""
    bg_mask = (depth < 0) | (depth > depth_max)
    x = depth.clamp(depth_min, depth_max)
    x = 3.0 / x - 0.6
    x = x.masked_fill(bg_mask, 0.0)
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x.unsqueeze(1)


def compute_navigation_metrics_torch(
    target_dist_history: torch.Tensor,
    collision_history: torch.Tensor,
    speed_history: torch.Tensor,
    reach_radius: float = 0.5,
) -> dict:
    """基于整条轨迹计算严格导航指标。"""
    if isinstance(target_dist_history, list):
        target_dist_history = torch.stack(target_dist_history)
    if isinstance(collision_history, list):
        collision_history = torch.stack(collision_history)
    if isinstance(speed_history, list):
        speed_history = torch.stack(speed_history)

    target_dist_history = target_dist_history.float()
    collision_history = collision_history.bool()
    speed_history = speed_history.float()

    collision_free = ~collision_history.any(dim=0)
    reached_target = (target_dist_history <= reach_radius).any(dim=0)
    success = collision_free & reached_target

    initial_dist = target_dist_history[0]
    best_dist = target_dist_history.min(dim=0).values
    final_dist = target_dist_history[-1]
    progress = ((initial_dist - best_dist) / initial_dist.clamp_min(1e-6)).clamp(0.0, 1.0)
    avg_speed = speed_history.mean(dim=0)
    max_speed = speed_history.max(dim=0).values

    return {
        'success_rate': success.float().mean(),
        'reach_rate': reached_target.float().mean(),
        'collision_free_rate': collision_free.float().mean(),
        'goal_progress': progress.mean(),
        'goal_distance_best': best_dist.mean(),
        'goal_distance_final': final_dist.mean(),
        'avg_speed': avg_speed.mean(),
        'max_speed': max_speed.mean(),
        'ar': (success.float() * avg_speed).mean(),
        'task_score': (collision_free.float() * progress * avg_speed).mean(),
    }


def compute_navigation_metrics_np(
    target_dist_history,
    collision_history,
    reach_radius: float = 0.5,
) -> dict:
    """NumPy 版本，用于评估与可视化摘要。"""
    target_dist_history = np.asarray(target_dist_history, dtype=np.float64)
    collision_history = np.asarray(collision_history, dtype=bool)

    collision_free = not collision_history.any()
    reached_target = bool((target_dist_history <= reach_radius).any())
    success = collision_free and reached_target

    initial_dist = float(target_dist_history[0])
    best_dist = float(target_dist_history.min())
    final_dist = float(target_dist_history[-1])
    progress = max(0.0, min(1.0, (initial_dist - best_dist) / max(initial_dist, 1e-6)))

    return {
        'collision_free': collision_free,
        'reached_target': reached_target,
        'success': success,
        'initial_dist': initial_dist,
        'best_dist': best_dist,
        'final_dist': final_dist,
        'progress': progress,
    }
