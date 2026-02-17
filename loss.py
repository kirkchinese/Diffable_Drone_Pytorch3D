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
                 coef_stall=0.0,
                 coef_yaw_explore=0.0,
                 coef_progress=0.0,
                 yaw_penalty_start_rad=0.5,
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
            coef_stall (float): 停滞惩罚权重。惩罚速度过低以打破
                "正对障碍物原地不动"的局部极小值。
            coef_yaw_explore (float): 偏航探索损失权重。结合速度与偏航角的复合损失：
                - 低速时小角度偏航提供负损失（奖励探索），超过阈值变为惩罚
                - 高速时任何偏航都为惩罚（抑制危险转向）
            coef_progress (float): 路径进度损失权重。奖励任何缩短与目标距离的运动，
                允许模型绕行而不被强制沿直线飞行。负梯度鼓励模型在面对障碍物时
                选择绕行路径而非原地停滞。
            yaw_penalty_start_rad (float): 偏航探索损失中"开始由奖励转惩罚"的角度阈值（弧度）。
                即 |yaw| < alpha 时为奖励区间，|yaw| > alpha 时为惩罚区间。
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
            'bias': coef_bias,
            'stall': coef_stall,
            'yaw_explore': coef_yaw_explore,
            'progress': coef_progress
        }
        self.ctl_dt = ctl_dt
        self.window_size = window_size
        self.yaw_penalty_start_rad = max(float(yaw_penalty_start_rad), 1e-6)

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
                yaw_history=None,
                p_target=None):
        """
        计算总损失和各项指标。

        Args:
            p_history: (T, B, 3) 位置历史。
            v_history: (T, B, 3) 速度历史。
            target_vel_history: (T, B, 3) 指向目标的向量历史。
            act_history: (T_act, B, 3) 动作历史。
            vec_to_obj_history: (T, B, 3) 到最近障碍物的向量历史。
            v_preds: (T, B, 3) 模型预测的速度。
            env_margin: (B,) 或标量。安全边距。
            env_g_std: (3,) 重力向量。默认为 [0, 0, -9.80665]。
            yaw_history: (T, B) 偏航偏移累积历史（弧度），仅在启用偏航控制时提供，
                否则为 None。用于计算 yaw_explore 损失。
            p_target: (B, 3) 目标位置（ROS 坐标系）。用于计算 progress 损失，
                为 None 则跳过 progress 损失计算。

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
        # 参考项目：jerk_history = act_buffer.diff(1, 0).mul(15) 
        # 其中 15 = 1/ctl_dt，ctl_dt = 1/15
        loss_d_acc = act_history.pow(2).sum(-1).mean()

        # jerk = d(acc)/dt，乘以 1/ctl_dt 得到正确的物理量
        jerk_history = act_history.diff(1, 0).mul(1.0 / self.ctl_dt)
        loss_d_jerk = jerk_history.pow(2).sum(-1).mean()

        # snap 计算：参考项目用 F.normalize 而不是手动归一化
        # snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
        thrust_vec = act_history - env_g_std
        thrust_dir = F.normalize(thrust_vec, dim=-1)  # 使用 F.normalize 更稳定
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

        # 整体速度损失
        loss_speed = F.smooth_l1_loss(fwd_v, target_v_norm)
        metrics['loss_speed'] = loss_speed

        # 停滞惩罚: 当速度低于 0.3 m/s 时产生递增惩罚
        # 打破"正对障碍物原地不动"的局部极小值
        # 使用 softplus 实现平滑阈值: speed ↑ → penalty ↓
        speed = v_history.norm(2, -1)  # (T, B)
        # softplus(-k*(speed - threshold)) 在 speed < threshold 时产生惩罚
        loss_stall = F.softplus(-10.0 * (speed - 0.3)).mean()
        metrics['loss_stall'] = loss_stall

        # 路径进度损失: 奖励任何缩短与目标点距离的运动
        # p_target: (B, 3) 目标位置
        if p_target is not None and self.coefs.get('progress', 0.0) > 0:
            # dist_to_target: (T, B) 每个时刻到目标的距离
            dist_to_target = (p_target.unsqueeze(0) - p_history).norm(2, -1)
            # step_progress: (T-1, B) 正值 = 靠近目标
            step_progress = -torch.diff(dist_to_target, dim=0)
            # loss = 负的平均进度 → 最小化 loss = 最大化进度
            # clamp(-0.3, ...): 地板限制, 防止无限负损失——进度奖励有2层含义:
            #   1. 崩塔防护: 没有地板时, 优化器会无限压低 progress loss,
            #      等价于无限提高飞行速度 → 碰撞损失被进度奖励淡化 → 莽撞
            #   2. 梯度平衡: 达到地板后, progress 的梯度为 0,
            #      让其他损失项 (碰撞/避障) 主导行为, 而不是和 progress 比大小
            loss_progress = -step_progress.mean().clamp(-0.3, 10.0)
        else:
            loss_progress = torch.tensor(0.0, device=p_history.device)
        metrics['loss_progress'] = loss_progress

        # 偏航探索损失: 速度门控的奖励-惩罚复合项
        # yaw_history: (T, B) 累积偏航偏移角 (弧度)
        if yaw_history is not None:
            # 使用平滑绝对值 sqrt(y² + ε²) 代替 abs(y)
            # 原因: abs(0) 的子梯度为 0，导致 y≈0 时无梯度信号（死区）
            # 平滑版本在 y≈0 附近提供强梯度: ∂/∂y ≈ y(2 - α/ε)，驱动探索启动
            eps_sq = 0.01 ** 2  # ε = 0.01
            smooth_abs_yaw = (yaw_history.pow(2) + eps_sq).sqrt()  # (T, B)
            
            # ------ 偏航形状函数 ------
            # f(y) = y² - α·smooth_abs(y)
            # 等价于 smooth_abs · (smooth_abs - α)，但数值更稳定
            # |y| < α: 负值 (奖励探索), |y| > α: 正值 (惩罚过度旋转)
            # α 默认为 0.5 rad ≈ 29°, 最大奖励在 α/2
            alpha = self.yaw_penalty_start_rad
            yaw_reward = yaw_history.pow(2) - alpha * smooth_abs_yaw  # (T, B)
            
            # ------ 速度门控 ------
            # 低速 (< 1 m/s): gate ≈ 1 → 使用 yaw_reward (含负值奖励)
            # 高速 (> 1 m/s): gate ≈ 0 → 使用纯二次惩罚
            speed = v_history.norm(2, -1)  # (T, B)
            speed_gate = torch.sigmoid(-5.0 * (speed - 1.0))
            
            # ------ 混合 ------
            # 低速: loss = yaw_reward (小角度负值奖励，大角度正值惩罚)
            # 高速: loss = y² (任何偏航都惩罚)
            yaw_penalty = yaw_history.pow(2)
            loss_yaw_explore = (speed_gate * yaw_reward + (1 - speed_gate) * yaw_penalty).mean()
        else:
            loss_yaw_explore = torch.tensor(0.0, device=p_history.device)
        metrics['loss_yaw_explore'] = loss_yaw_explore

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
            self.coefs['bias'] * loss_bias +
            self.coefs['stall'] * loss_stall +
            self.coefs['yaw_explore'] * loss_yaw_explore +
            self.coefs['progress'] * loss_progress
        )

        # 附加指标
        with torch.no_grad():
            speed_history = v_history.norm(2, -1)
            success = torch.all(distance.flatten(0, 1) > 0)
            metrics['success_rate'] = float(success)
            metrics['avg_speed'] = speed_history.mean().item()
            metrics['max_speed'] = speed_history.max().item()

        return total_loss, metrics
