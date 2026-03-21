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
                 coef_lateral=0.0,
                 coef_drone_collide=5.0,
                 ctl_dt=0.02,
                 window_size=30,
                 loss_v_mode='mse',
                 adaptive_decay_rate=2.0):
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
            coef_lateral (float): 横向运动惩罚权重。0.0=横向完全免罚（推荐），
                >0时轻微惩罚横向分量以防止绕圈。
            coef_drone_collide (float): 无人机间碰撞损失权重。多机场景默认使用 5.0。
                仅在 forward 传入 inter_drone_dist_history 时生效。
            ctl_dt (float): 控制时间步长，用于缩放导数计算。
            window_size (int): 速度平均窗口大小。
            loss_v_mode (str): 速度损失模式。
                'mse'        — 参考项目式，SmoothL1(||v̄−target_v||₂)，惩罚所有方向（默认）
                'decomposed' — 分解式，只罚前向速度不足，横向完全免罚
                'adaptive'   — 自适应式，前向跟踪＋指数衰减制动（终点强制悬停）
            adaptive_decay_rate (float): adaptive 模式的指数衰减率 λ。
                alpha = exp(-λ * V_target)。λ 越大制动越快开始，默认 2.0。
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
            'bias': coef_bias,
            'lateral': coef_lateral,
            'drone_collide': coef_drone_collide,
        }
        self.ctl_dt = ctl_dt
        self.window_size = window_size
        self.loss_v_mode = loss_v_mode
        self.adaptive_decay_rate = adaptive_decay_rate

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
                env_g_std=None,
                inter_drone_dist_history=None):
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
            inter_drone_dist_history: 可选，(T, B) 或 (T, S, B) 无人机间最近椭球距离历史。
                仅在 n_drones_per_group > 1 时由训练循环传入。

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
            env_g_std = p_history.new_tensor([0.0, 0.0, -9.80665])

        T, B, _ = v_history.shape
        metrics = {}

        # 损失计算

        # 地面亲和损失：惩罚Z > 0的位置
        loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
        metrics['loss_ground_affinity'] = loss_ground_affinity

        # ==================== 速度跟踪损失 ====================
        # 三种模式 (loss_v_mode):
        #   'mse'        — 参考项目式: SmoothL1(||v̄ − target_v||₂)
        #                  惩罚所有方向偏差（含横向），容易在障碍物前悬停
        #   'decomposed' — 分解式: SmoothL1(relu(V_target − v_fwd))
        #                  横向完全免罚，打破悬停局部最优，但终点会绕圈
        #   'adaptive'   — 自适应式: 前向跟踪 + exp(-λ·V_target) 制动
        #                  路途横向免罚 + 终点强制悬停，兼顾避障与制动
        if T > self.window_size:
            v_history_cum = v_history.cumsum(0)
            v_history_avg = (v_history_cum[self.window_size:] - v_history_cum[:-self.window_size]) / self.window_size

            target_slice = target_vel_history[1 : 1 + (T - self.window_size)]

            if v_history_avg.shape[0] != target_slice.shape[0]:
                min_len = min(v_history_avg.shape[0], target_slice.shape[0])
                v_history_avg = v_history_avg[:min_len]
                target_slice = target_slice[:min_len]

            if self.loss_v_mode == 'mse':
                # ---- 参考项目式 ----
                # SmoothL1(||v̄₃₀ − target_v||₂)
                # 等价于惩罚速度向量误差的模长，横向偏移也受罚
                delta_v = (v_history_avg - target_slice).norm(p=2, dim=-1)
                loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
                loss_lateral = p_history.new_tensor(0.0)

            elif self.loss_v_mode == 'decomposed':
                # ---- 分解式 ----
                target_speed = target_slice.norm(p=2, dim=-1)
                target_unit = target_slice / (target_speed.unsqueeze(-1) + 1e-6)
                v_fwd = (v_history_avg * target_unit).sum(dim=-1)

                shortfall = F.relu(target_speed - v_fwd)
                loss_v = F.smooth_l1_loss(shortfall, torch.zeros_like(shortfall))

                if self.coefs['lateral'] > 0:
                    v_perp = v_history_avg - v_fwd.unsqueeze(-1) * target_unit
                    loss_lateral = F.smooth_l1_loss(
                        v_perp.norm(p=2, dim=-1),
                        torch.zeros(v_perp.shape[:-1], device=v_perp.device))
                else:
                    loss_lateral = p_history.new_tensor(0.0)

            else:  # adaptive
                # ---- 自适应式 ----
                # Part 1: 前向速度不足惩罚
                target_speed = target_slice.norm(p=2, dim=-1)          # (T', B)
                target_unit = target_slice / (target_speed.unsqueeze(-1) + 1e-6)
                v_fwd = (v_history_avg * target_unit).sum(dim=-1)      # (T', B)

                shortfall = F.relu(target_speed - v_fwd)
                loss_v_fwd = F.smooth_l1_loss(shortfall, torch.zeros_like(shortfall))

                # Part 2: 指数衰减制动 — 目标速度越低，制动惩罚越强
                # alpha = exp(-λ * V_target)
                #   V_target≈3 m/s (路途) → alpha≈0.002 → 几乎不制动
                #   V_target≈0   (终点) → alpha≈1.0   → 惩罚全部速度分量
                alpha = torch.exp(-self.adaptive_decay_rate * target_speed)  # (T', B)
                v_total = v_history_avg.norm(p=2, dim=-1)                   # (T', B)
                brake_elem = F.smooth_l1_loss(
                    v_total, torch.zeros_like(v_total), reduction='none')    # (T', B)
                loss_v_brake = (alpha * brake_elem).mean()

                loss_v = loss_v_fwd + loss_v_brake

                # Part 3: 横向速度惩罚 — 约束垂直于目标方向的漂移
                # 修复: adaptive 模式之前横向损失恒为 0 导致无人机飞高逃避障碍物
                if self.coefs['lateral'] > 0:
                    v_perp = v_history_avg - v_fwd.unsqueeze(-1) * target_unit
                    loss_lateral = F.smooth_l1_loss(
                        v_perp.norm(p=2, dim=-1),
                        torch.zeros(v_perp.shape[:-1], device=v_perp.device))
                else:
                    loss_lateral = p_history.new_tensor(0.0)

                metrics['loss_v_fwd'] = loss_v_fwd
                metrics['loss_v_brake'] = loss_v_brake
        else:
            loss_v = p_history.new_tensor(0.0)
            loss_lateral = p_history.new_tensor(0.0)
        metrics['loss_v'] = loss_v
        metrics['loss_lateral'] = loss_lateral

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
        # 参考项目：jerk_history = act_buffer.diff(1, 0).mul(15) 
        # 其中 15 = 1/ctl_dt，ctl_dt = 1/15
        loss_d_acc = act_history.pow(2).sum(-1).mean()

        # jerk = d(acc)/dt，乘以 1/ctl_dt 得到正确的物理量
        jerk_history = act_history.diff(1, 0).mul(1.0 / self.ctl_dt)
        loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

        # snap 计算：参考项目用 F.normalize 而不是手动归一化
        # snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
        thrust_vec = act_history - env_g_std
        thrust_dir = F.normalize(thrust_vec, dim=-1, eps=1e-6)  # 避免零向量导致 NaN
        snap_history = thrust_dir.diff(1, 0).diff(1, 0).mul((1.0 / self.ctl_dt) ** 2)
        loss_d_snap = snap_history.pow(2).sum(-1).mean()

        metrics['loss_d_acc'] = loss_d_acc
        metrics['loss_d_jerk'] = loss_d_jerk
        metrics['loss_d_snap'] = loss_d_snap

        # 障碍物回避计算
        # vec_to_obj_history 支持两种格式：
        #   - 单点:   (T, B, 3)    → distance (T, B)
        #   - 子步细分: (T, S, B, 3) → distance (T, S, B)
        distance = torch.norm(vec_to_obj_history, 2, -1)
        distance = distance - env_margin  # env_margin (B,) 自动 broadcast

        has_subdiv = (distance.dim() == 3)  # (T, S, B) 表示有子步细分

        with torch.no_grad():
            if has_subdiv:
                # 子步细分：沿子步维度 (dim=1) 差分，与参考项目一致
                # Δt_sub = ctl_dt / (S-1)，所以 1/Δt_sub = (S-1)/ctl_dt
                S = distance.shape[1]
                v_to_pt = (-torch.diff(distance, 1, 1) * ((S - 1) / self.ctl_dt)).clamp_min(1)
            else:
                # 单点：沿时间轴 (dim=0) 差分，乘以 9/ctl_dt (= 135 @ 15Hz)
                v_to_pt = (-torch.diff(distance, 1, 0) * (1.0 / self.ctl_dt) * 9.0).clamp_min(1)

        if has_subdiv:
            dist_slice = distance[:, 1:]  # (T, S-1, B)
        else:
            dist_slice = distance[1:]  # (T-1, B)

        loss_obj_avoidance = self.barrier(dist_slice, v_to_pt)
        metrics['loss_obj_avoidance'] = loss_obj_avoidance

        loss_collide = F.softplus(dist_slice.mul(-32)).mul(v_to_pt).mean()
        metrics['loss_collide'] = loss_collide

        # 无人机间碰撞损失 (多机场景)
        if inter_drone_dist_history is not None:
            drone_dist = to_tensor(inter_drone_dist_history)
            # softplus(-32 * d) 在 d<0 时急剧增长, 惩罚碰撞
            loss_drone_collide = F.softplus(drone_dist.mul(-32)).mean()
        else:
            loss_drone_collide = p_history.new_tensor(0.0)
        metrics['loss_drone_collide'] = loss_drone_collide

        # 整体速度损失
        loss_speed = F.smooth_l1_loss(fwd_v, target_v_norm)
        metrics['loss_speed'] = loss_speed

        # 总损失
        total_loss = (
            self.coefs['v'] * loss_v +
            self.coefs['lateral'] * loss_lateral +
            self.coefs['speed'] * loss_speed +
            self.coefs['v_pred'] * loss_v_pred +
            self.coefs['collide'] * loss_collide +
            self.coefs['obj_avoidance'] * loss_obj_avoidance +
            self.coefs['d_acc'] * loss_d_acc +
            self.coefs['d_jerk'] * loss_d_jerk +
            self.coefs['d_snap'] * loss_d_snap +
            self.coefs['ground_affinity'] * loss_ground_affinity +
            self.coefs['bias'] * loss_bias +
            self.coefs['drone_collide'] * loss_drone_collide
        )

        # 附加指标
        with torch.no_grad():
            speed_history = v_history.norm(2, -1)
            success = torch.all(distance.flatten(0, 1) > 0)
            metrics['success_rate'] = float(success)
            metrics['avg_speed'] = speed_history.mean().item()
            metrics['max_speed'] = speed_history.max().item()

        return total_loss, metrics
