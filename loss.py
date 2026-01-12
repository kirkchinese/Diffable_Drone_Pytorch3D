"""
无人机训练损失函数模块。

该模块定义了用于无人机训练的损失函数类，包括速度跟踪、障碍物回避、控制平滑性等各项损失。
"""

import torch
import torch.nn.functional as F


class DroneLoss:
    """
    无人机训练损失函数类。

    该类负责计算无人机训练过程中的各项损失，包括速度跟踪、障碍物回避、控制平滑性等。
    支持多代理场景下的损失计算。

    Attributes:
        coefs (dict): 各项损失的权重系数。
        ctl_dt (float): 控制时间步长，用于缩放导数计算。
        window_size (int): 速度平均窗口大小。
    """

    def __init__(self,
                 coef_v=1.0,
                 coef_speed=0.0,
                 coef_v_pred=2.0,
                 coef_collide=2.0,
                 coef_obj_avoidance=1.5,
                 coef_d_acc=0.01,
                 coef_d_jerk=0.001,
                 coef_d_snap=0.0,
                 coef_ground_affinity=0.0,
                 coef_bias=0.0,
                 ctl_dt=0.02,
                 window_size=30):
        """
        初始化无人机损失函数。

        Args:
            coef_v (float): 速度跟踪损失权重。
            coef_speed (float): 速度损失权重。
            coef_v_pred (float): 预测速度损失权重。
            coef_collide (float): 碰撞损失权重。
            coef_obj_avoidance (float): 障碍物回避损失权重。
            coef_d_acc (float): 加速度平滑损失权重。
            coef_d_jerk (float): 加加速度（jerk）平滑损失权重。
            coef_d_snap (float): 快照（snap）平滑损失权重。
            coef_ground_affinity (float): 地面亲和损失权重。
            coef_bias (float): 偏向损失权重。
            ctl_dt (float): 控制时间步长，用于缩放导数计算。
            window_size (int): 速度平均窗口大小。
        """
        self.coefs = {
            'v': coef_v,
            'speed': coef_speed,
            'v_pred': coef_v_pred,
            'collide': coef_collide,
            'obj_avoidance': coef_obj_avoidance,
            'd_acc': coef_d_acc,
            'd_jerk': coef_d_jerk,
            'd_snap': coef_d_snap,
            'ground_affinity': coef_ground_affinity,
            'bias': coef_bias
        }
        self.ctl_dt = ctl_dt
        self.window_size = window_size

    def barrier(self, x: torch.Tensor, v_to_pt: torch.Tensor) -> torch.Tensor:
        """
        障碍物回避的屏障函数。

        计算基于距离和相对速度的屏障损失，用于鼓励无人机远离障碍物。

        Args:
            x (torch.Tensor): 距离障碍物的归一化距离。
            v_to_pt (torch.Tensor): 相对速度。

        Returns:
            torch.Tensor: 屏障损失值。
        """
        return (v_to_pt * (1 - x).relu().pow(2)).mean()

    def forward(self,
                p_history,
                v_history,
                target_vel_history,
                act_history,
                vec_to_obj_history,
                v_preds,
                env_margin,
                env_g_std=None):
        """
        计算总损失和各项指标。

        Args:
            p_history: (T, B, 3) 位置历史。
            v_history: (T, B, 3) 速度历史。
            target_vel_history: (T, B, 3) 目标速度历史。
            act_history: (T_act, B, 3) 动作历史。
            vec_to_obj_history: (T, B, 3) 到最近障碍物的向量历史。
            v_preds: (T, B, 3) 模型预测的速度。
            env_margin: (B,) 或标量。安全边距。
            env_g_std: (3,) 重力向量。默认为 [0, 0, -9.80665]。

        Returns:
            tuple: (总损失, 指标字典)
                - loss (torch.Tensor): 标量总损失。
                - metrics (dict): 包含各项损失和统计信息的字典。
        """
        # 辅助函数：转换列表为张量
        def to_tensor(x):
            if isinstance(x, list):
                return torch.stack(x)
            return x

        p_history = to_tensor(p_history)
        v_history = to_tensor(v_history)
        target_vel_history = to_tensor(target_vel_history)
        act_history = to_tensor(act_history)
        vec_to_obj_history = to_tensor(vec_to_obj_history)
        v_preds = to_tensor(v_preds)

        if env_g_std is None:
            env_g_std = torch.tensor([0.0, 0.0, -9.80665], device=p_history.device)

        T, B, _ = v_history.shape
        metrics = {}

        # 损失计算

        # 地面亲和损失：惩罚Z > 0的位置
        loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
        metrics['loss_ground_affinity'] = loss_ground_affinity

        # 速度跟踪损失：平均速度与目标速度的平滑L1损失
        if T > self.window_size:
            v_history_cum = v_history.cumsum(0)
            v_history_avg = (v_history_cum[self.window_size:] - v_history_cum[:-self.window_size]) / self.window_size

            target_slice = target_vel_history[1 : 1 + (T - self.window_size)]

            if v_history_avg.shape[0] != target_slice.shape[0]:
                min_len = min(v_history_avg.shape[0], target_slice.shape[0])
                v_history_avg = v_history_avg[:min_len]
                target_slice = target_slice[:min_len]

            delta_v = torch.norm(v_history_avg - target_slice, 2, -1)
            loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
        else:
            loss_v = torch.tensor(0.0, device=p_history.device)
        metrics['loss_v'] = loss_v

        # 预测速度损失：模型预测与实际速度的MSE损失
        loss_v_pred = F.mse_loss(v_preds, v_history.detach())
        metrics['loss_v_pred'] = loss_v_pred

        # 偏向损失：方向正确性
        target_v_norm = torch.norm(target_vel_history, 2, -1)
        target_v_normalized = target_vel_history / (target_v_norm[..., None] + 1e-6)
        fwd_v = torch.sum(v_history * target_v_normalized, -1)
        loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_normalized) * 3
        metrics['loss_bias'] = loss_bias

        # 控制正则化：加速度、jerk、snap
        loss_d_acc = act_history.pow(2).sum(-1).mean()

        jerk_history = act_history.diff(1, 0) / self.ctl_dt
        loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

        # 推力方向归一化（添加数值稳定性）
        thrust_vec = act_history - env_g_std
        thrust_norm = torch.norm(thrust_vec, dim=-1, keepdim=True).clamp(min=1e-6)
        thrust_dir = thrust_vec / thrust_norm
        snap_history = thrust_dir.diff(1, 0).diff(1, 0) / (self.ctl_dt ** 2)
        loss_d_snap = snap_history.pow(2).sum(-1).mean()

        metrics['loss_d_acc'] = loss_d_acc
        metrics['loss_d_jerk'] = loss_d_jerk
        metrics['loss_d_snap'] = loss_d_snap

        # 障碍物回避计算
        distance = torch.norm(vec_to_obj_history, 2, -1)
        distance = distance - env_margin

        # 参考项目使用 * 135 (即 9 / ctl_dt，ctl_dt=1/15 时为 135)
        # 计算接近障碍物的速度，用于加权损失
        with torch.no_grad():
            v_to_pt = (-torch.diff(distance, 1, 0) * (1.0 / self.ctl_dt) * 9.0).clamp_min(1)

        dist_slice = distance[1:]

        loss_obj_avoidance = self.barrier(dist_slice, v_to_pt)
        metrics['loss_obj_avoidance'] = loss_obj_avoidance

        loss_collide = F.softplus(dist_slice.mul(-32)).mul(v_to_pt).mean()
        metrics['loss_collide'] = loss_collide

        # 整体速度损失
        loss_speed = F.smooth_l1_loss(fwd_v, target_v_norm)
        metrics['loss_speed'] = loss_speed

        # 总损失
        total_loss = (
            self.coefs['v'] * loss_v +
            self.coefs['speed'] * loss_speed +
            self.coefs['v_pred'] * loss_v_pred +
            self.coefs['collide'] * loss_collide +
            self.coefs['obj_avoidance'] * loss_obj_avoidance +
            self.coefs['d_acc'] * loss_d_acc +
            self.coefs['d_jerk'] * loss_d_jerk +
            self.coefs['d_snap'] * loss_d_snap +
            self.coefs['ground_affinity'] * loss_ground_affinity +
            self.coefs['bias'] * loss_bias
        )

        # 附加指标
        with torch.no_grad():
            speed_history = v_history.norm(2, -1)
            success = torch.all(distance.flatten(0, 1) > 0)
            metrics['success_rate'] = float(success)
            metrics['avg_speed'] = speed_history.mean().item()
            metrics['max_speed'] = speed_history.max().item()

        return total_loss, metrics
