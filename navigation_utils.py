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

    # collision_history 可能是 (T, B) 或 (T, S, B) (子步细分);
    # 必须先折叠到 (T*S, B) 再沿 dim=0 取 any，否则结果形状错误
    if collision_history.dim() == 3:
        collision_history = collision_history.flatten(0, 1)
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
    import numpy as np

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


# ============================================================
# DronePolicy: 模型推理管线适配器（解耦模型与训练循环）
# ============================================================

class DronePolicy:
    """
    将 obs 构造 → 模型前向 → 动作后处理 封装为统一接口。

    任意满足 ``forward(x, v, hx) -> (act, aux, hx)`` 的模型均可即插即用。
    训练/评估循环只需调用 ``policy.infer(...)`` 即可获得推力指令。
    """

    def __init__(self, model, g_std, depth_min, depth_max, no_odom=False):
        """
        Args:
            model: nn.Module，接口 forward(x, v, hx) -> (act_raw, aux, hx)
            g_std: (3,) 重力向量
            depth_min, depth_max: 深度图裁剪范围
            no_odom: 是否省略里程计速度观测
        """
        self.model = model
        self.g_std = g_std
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.no_odom = no_odom

    # ---- 主接口 ----

    def infer(self, depth, R, v, target_v_raw, margin, max_speed,
              thr_est_error, hx, depth_noise_std=0.0):
        """
        单步完整推理管线。

        Args:
            depth:  (B, H, W) 渲染深度图
            R:      (B, 3, 3) 机体姿态旋转矩阵 (body→world)
            v:      (B, 3) 世界坐标系速度
            target_v_raw: (B, 3) 到目标点的世界坐标系方向向量
            margin: (B,) 安全半径
            max_speed: (B,1) 目标截断速度
            thr_est_error: (B,1) 推力估计误差系数
            hx:     GRU 隐状态 或 None
            depth_noise_std: 深度图噪声标准差

        Returns:
            act_cmd:  (B, 3) 推力指令
            v_pred:   (B, 3) 速度预测
            target_v: (B, 3) 截断后目标速度
            hx:       更新后的 GRU 隐状态
        """
        B = depth.shape[0]

        # 局部水平坐标系
        R_local = compute_local_frame(R)

        # 目标速度截断
        target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
        target_v_unit = target_v_raw / (target_v_norm + 1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, max_speed)

        # 转换到局部坐标系
        target_v_local = (target_v[:, None] @ R_local).squeeze(1)
        local_v = (v[:, None] @ R_local).squeeze(1)

        # 状态向量
        parts = [target_v_local, R[:, 2], margin[:, None]]
        if not self.no_odom:
            parts.insert(0, local_v)
        state = torch.cat(parts, dim=-1)

        # 深度图预处理
        x = preprocess_depth_for_model(
            depth, self.depth_min, self.depth_max, depth_noise_std)

        # 模型前向
        act_raw, img_feat, hx = self.model(x, state, hx)
        self._last_img_feat = img_feat  # 供 CMA-ES DecayController 使用

        # 动作后处理：local→world + 推力换算
        act_world = R_local @ act_raw.reshape(B, 3, 2)
        a_pred, v_pred = act_world.unbind(-1)
        act_cmd = (a_pred - v_pred - self.g_std) * thr_est_error + self.g_std

        return act_cmd, v_pred, target_v, hx

    def reset(self):
        """重置模型隐状态（新 episode 时调用）。"""
        self.model.reset()
