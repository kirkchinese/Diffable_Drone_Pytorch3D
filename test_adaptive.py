"""
Model_adaptive 专用测试脚本

针对 train_adaptive.py 训练的 Model_adaptive 模型进行测试和可视化。

功能特性：
- 自动检测模型参数 (dim_obs, no_odom)
- 实时深度图和 RGB 渲染显示
- 3D 轨迹可视化与保存
- 碰撞检测与统计
- 多无人机并行测试支持

使用方法:
    python test_adaptive.py --model_path ./checkpoints/checkpoint_final.pth
    python test_adaptive.py --model_path ./checkpoints/checkpoint_final.pth --batch_size 4 --no_display

按键控制:
    ESC / Q: 退出
    R: 重置环境
    SPACE: 暂停/继续
    +/-: 调整仿真速度
    S: 手动保存当前轨迹图
    V: 切换显示模式 (RGB/Depth/Both)
"""

import os
import math
import argparse
from random import normalvariate
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_env import DroneSimulator
from model import Model_adaptive


def parse_args():
    parser = argparse.ArgumentParser(description='Model_adaptive 专用测试脚本')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    
    # 环境参数
    parser.add_argument('--batch_size', type=int, default=1, help='同时测试的无人机数量 (1-4 推荐)')
    parser.add_argument('--timesteps', type=int, default=500, help='每个 episode 的最大步数')
    parser.add_argument('--ctl_dt', type=float, default=1/15, help='控制时间步长')
    
    # 渲染参数
    parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角')
    parser.add_argument('--image_height', type=int, default=240, help='渲染图像高度')
    parser.add_argument('--image_width', type=int, default=320, help='渲染图像宽度')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj', help='障碍物网格路径')
    parser.add_argument('--num_samples', type=int, default=100000, help='障碍物点云采样数')
    
    # 无人机参数 - 与 train_adaptive.py 保持一致
    parser.add_argument('--margin_min', type=float, default=0.1, help='安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.7, help='安全半径最大值')
    parser.add_argument('--init_p_range', type=float, default=8.0, help='初始位置范围')
    parser.add_argument('--noise_std', type=float, default=0.04, help='环境扰动噪声标准差')
    parser.add_argument('--yaw_inertia', type=float, default=5.0, help='偏航惯性')
    parser.add_argument('--yaw_ctl_delay', type=float, default=12.0, help='偏航控制延迟')
    parser.add_argument('--pitch_ctl_delay', type=float, default=12.0, help='俯仰控制延迟')
    parser.add_argument('--airmode_coef', type=float, default=0.5, help='Airmode 系数')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减系数')
    
    # 显示参数
    parser.add_argument('--display_scale', type=int, default=2, help='深度图显示放大倍数')
    parser.add_argument('--fps_limit', type=int, default=30, help='显示帧率限制')
    parser.add_argument('--no_display', action='store_true', help='禁用实时显示 (仅运行测试)')
    parser.add_argument('--display_mode', type=str, default='both', 
                        choices=['rgb', 'depth', 'both'], help='显示模式')
    
    # 轨迹图参数
    parser.add_argument('--save_trajectory', action='store_true', default=True, help='保存轨迹图')
    parser.add_argument('--trajectory_dir', type=str, default='./trajectory_plots/adaptive', help='轨迹图保存目录')
    parser.add_argument('--scene_points', type=int, default=5000, help='场景点云显示数量')
    
    # 测试参数
    parser.add_argument('--num_episodes', type=int, default=0, help='测试 episode 数量 (0 表示无限)')
    parser.add_argument('--target_dist_min', type=float, default=3.0, help='目标距离最小值')
    parser.add_argument('--target_dist_max', type=float, default=16.0, help='目标距离最大值')
    parser.add_argument('--max_speed_min', type=float, default=0.75, help='最大速度下限')
    parser.add_argument('--max_speed_range', type=float, default=5.0, help='最大速度随机范围')
    
    # 模型参数 (自动检测，但可手动覆盖)
    parser.add_argument('--no_odom', default=None, action='store_true', 
                        help='不使用里程计速度 (默认自动检测)')
    parser.add_argument('--force_odom', action='store_true', help='强制使用里程计')
    
    # 硬件
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    # 日志
    parser.add_argument('--log_metrics', action='store_true', default=True, help='记录详细指标')
    parser.add_argument('--metrics_file', type=str, default=None, help='指标保存文件路径')
    
    return parser.parse_args()


class AdaptiveModelTester:
    """Model_adaptive 专用测试器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.ctl_dt = args.ctl_dt
        self.paused = False
        self.speed_multiplier = 1.0
        self.display_mode = args.display_mode
        
        # 初始化环境 - 与 train_adaptive.py 保持一致
        self.env = DroneSimulator(
            batch_size=args.batch_size,
            dt=self.ctl_dt,
            mesh_path=args.mesh_path,
            image_size=(args.image_height, args.image_width),
            device=self.device,
            enable_airmode=True,
            enable_induced_drag=False,
            noise_std=args.noise_std,
            grad_decay=args.grad_decay,
            yaw_inertia=args.yaw_inertia,
            yaw_ctl_delay=args.yaw_ctl_delay,
            pitch_ctl_delay=args.pitch_ctl_delay,
            airmode_coef=args.airmode_coef,
            init_p_range=args.init_p_range,
            init_margin_range=(args.margin_min, args.margin_max),
            num_samples=args.num_samples
        )
        
        # 初始化模型 - 将在 _load_model 中设置
        self.model = None
        self.no_odom = None
        
        # 加载模型权重
        self._load_model(args.model_path)
        self.model.eval()
        
        # 重力向量
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)
        
        # 显示窗口设置
        if not args.no_display:
            self.window_name = "Adaptive Model Test"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 统计信息
        self.episode_count = 0
        self.step_count = 0
        self.collision_count = 0
        self.success_count = 0
        self.timeout_count = 0
        
        # 详细指标记录
        self.metrics_history = defaultdict(list)
        self.episode_metrics = []
        
        # 轨迹记录
        self.trajectory_history = []
        self.velocity_history = []
        self.clearance_history = []
        self.p_start = None
        
        # 创建轨迹图保存目录
        if args.save_trajectory:
            os.makedirs(args.trajectory_dir, exist_ok=True)
            print(f"轨迹图将保存到: {args.trajectory_dir}")
        
        # 预采样场景点云用于可视化
        self.scene_points_for_plot = self._sample_scene_points(args.scene_points)
        
    def _load_model(self, path):
        """加载模型权重，自动检测模型配置"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        
        # 自动检测 dim_obs
        # Model_adaptive 的 v_proj 层输入维度即为 dim_obs
        if 'v_proj.weight' in state_dict:
            v_proj_shape = state_dict['v_proj.weight'].shape
            dim_obs = v_proj_shape[1]
        else:
            raise ValueError("Cannot detect dim_obs from checkpoint")
        
        # 检测 dim_action
        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            dim_action = fc_shape[0]
        else:
            dim_action = 6  # 默认值
        
        print(f"Detected model config: dim_obs={dim_obs}, dim_action={dim_action}")
        
        # 确定 no_odom 设置
        if self.args.force_odom:
            self.no_odom = False
        elif self.args.no_odom is not None:
            self.no_odom = self.args.no_odom
        else:
            # 自动检测：dim_obs=7 表示 no_odom, dim_obs=10 表示有 odom
            self.no_odom = (dim_obs == 7)
        
        print(f"no_odom mode: {self.no_odom}")
        
        # 创建模型
        self.model = Model_adaptive(dim_obs=dim_obs, dim_action=dim_action).to(self.device)
        
        # 加载权重
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning - Missing keys: {missing}")
        if unexpected:
            print(f"Warning - Unexpected keys: {unexpected}")
        print(f"Loaded model from {path}")
    
    def _sample_scene_points(self, num_points):
        """从场景网格采样点云用于可视化"""
        full_pcd = self.env.renderer.obstacle_pcd[0].cpu().numpy()
        if len(full_pcd) > num_points:
            indices = np.random.choice(len(full_pcd), num_points, replace=False)
            return full_pcd[indices]
        return full_pcd
    
    def _compute_local_R(self):
        """计算局部坐标系旋转矩阵 - 与 train_adaptive.py 一致"""
        fwd = self.env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
        fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8),
                          torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(fwd))
        R = torch.stack([fwd, torch.linalg.cross(up, fwd), up], -1)
        return R
    
    def plot_trajectory(self, save_path=None, show=False):
        """绘制 3D 轨迹图，包含场景点云和详细信息"""
        if len(self.trajectory_history) < 2:
            print("轨迹点太少，跳过绘图")
            return
        
        trajectory = np.array(self.trajectory_history)
        velocity = np.array(self.velocity_history) if self.velocity_history else None
        clearance = np.array(self.clearance_history) if self.clearance_history else None
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 主 3D 轨迹图
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        
        # 绘制场景点云
        scene_pts = self.scene_points_for_plot
        ax_3d.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                      c='gray', s=1, alpha=0.3, label='Scene')
        
        # 绘制轨迹线（颜色渐变）
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        for i in range(len(trajectory) - 1):
            ax_3d.plot3D(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                         color=colors[i], linewidth=2)
        
        # 标记点
        ax_3d.scatter(*self.p_start, c='lime', s=200, marker='o',
                      edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
        ax_3d.scatter(*trajectory[-1], c='red', s=200, marker='o',
                      edgecolors='darkred', linewidths=2, label='End', zorder=5)
        
        target = self.p_target[0].cpu().numpy()
        ax_3d.scatter(*target, c='yellow', s=300, marker='*',
                      edgecolors='orange', linewidths=2, label='Target', zorder=5)
        
        # 连接终点和目标
        ax_3d.plot3D([trajectory[-1, 0], target[0]],
                     [trajectory[-1, 1], target[1]],
                     [trajectory[-1, 2], target[2]],
                     'y--', linewidth=1, alpha=0.7)
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.legend(loc='upper left')
        
        # 设置等比例坐标轴
        all_points = np.vstack([trajectory, scene_pts, target.reshape(1, 3)])
        max_range = np.max(np.ptp(all_points, axis=0)) / 2
        mid = np.mean(all_points, axis=0)
        ax_3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax_3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax_3d.set_zlim(max(0, mid[2] - max_range), mid[2] + max_range)
        
        # 速度曲线
        if velocity is not None:
            ax_vel = fig.add_subplot(2, 2, 2)
            speed = np.linalg.norm(velocity, axis=1)
            time_steps = np.arange(len(speed)) * self.ctl_dt
            ax_vel.plot(time_steps, speed, 'b-', linewidth=1.5, label='Speed')
            ax_vel.axhline(y=self.max_speed[0, 0].cpu().item(), color='r', 
                          linestyle='--', label=f'Max ({self.max_speed[0, 0].item():.2f} m/s)')
            ax_vel.set_xlabel('Time (s)')
            ax_vel.set_ylabel('Speed (m/s)')
            ax_vel.set_title('Speed Profile')
            ax_vel.legend()
            ax_vel.grid(True, alpha=0.3)
        
        # 间隙曲线
        if clearance is not None:
            ax_clr = fig.add_subplot(2, 2, 3)
            time_steps = np.arange(len(clearance)) * self.ctl_dt
            margin = self.env.margin[0].item()
            ax_clr.fill_between(time_steps, 0, clearance, 
                               where=(clearance > 0), color='green', alpha=0.3, label='Safe')
            ax_clr.fill_between(time_steps, clearance, 0,
                               where=(clearance < 0), color='red', alpha=0.3, label='Collision')
            ax_clr.plot(time_steps, clearance, 'k-', linewidth=1.5)
            ax_clr.axhline(y=0, color='r', linestyle='-', linewidth=2)
            ax_clr.axhline(y=margin, color='orange', linestyle='--', 
                          label=f'Margin ({margin:.3f}m)')
            ax_clr.set_xlabel('Time (s)')
            ax_clr.set_ylabel('Clearance (m)')
            ax_clr.set_title('Obstacle Clearance')
            ax_clr.legend()
            ax_clr.grid(True, alpha=0.3)
        
        # 信息面板
        ax_info = fig.add_subplot(2, 2, 4)
        ax_info.axis('off')
        
        dist_to_target = np.linalg.norm(trajectory[-1] - target)
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        min_clearance = np.min(clearance) if clearance is not None else 0
        avg_speed = np.mean(np.linalg.norm(velocity, axis=1)) if velocity is not None else 0
        
        status = "✓ SUCCESS" if dist_to_target < 0.5 else (
            "✗ COLLISION" if self.episode_collided else "○ TIMEOUT")
        
        info_text = f"""
Episode {self.episode_count} - {status}

═══════════════════════════════════════
Performance Metrics
═══════════════════════════════════════
Distance to Target: {dist_to_target:.3f} m
Path Length:        {total_distance:.3f} m
Steps:              {len(trajectory)}
Duration:           {len(trajectory) * self.ctl_dt:.2f} s

═══════════════════════════════════════
Safety Metrics
═══════════════════════════════════════
Safety Margin:      {self.env.margin[0].item():.3f} m
Min Clearance:      {min_clearance:.3f} m
Collision:          {'Yes' if self.episode_collided else 'No'}

═══════════════════════════════════════
Speed Metrics
═══════════════════════════════════════
Average Speed:      {avg_speed:.2f} m/s
Max Speed Setting:  {self.max_speed[0, 0].item():.2f} m/s

═══════════════════════════════════════
Model Config
═══════════════════════════════════════
no_odom:            {self.no_odom}
"""
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def reset_episode(self, save_previous=True):
        """重置一个 episode"""
        # 保存上一个 episode 的轨迹图
        if save_previous and self.args.save_trajectory and len(self.trajectory_history) > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"adaptive_ep{self.episode_count:04d}_{timestamp}.png"
            save_path = os.path.join(self.args.trajectory_dir, filename)
            self.plot_trajectory(save_path)
            
            # 保存 episode 指标
            if self.args.log_metrics:
                self._save_episode_metrics()
        
        self.env.reset()
        self.model.reset()
        
        B = self.args.batch_size
        
        # 生成目标点 - 与 train_adaptive.py 一致
        angle = torch.rand(B, device=self.device) * 2 * math.pi
        dist = torch.rand(B, device=self.device) * (
            self.args.target_dist_max - self.args.target_dist_min
        ) + self.args.target_dist_min
        
        self.p_target = self.env.p.clone()
        self.p_target[:, 0] += torch.cos(angle) * dist
        self.p_target[:, 1] += torch.sin(angle) * dist
        self.p_target[:, 2] += torch.randn(B, device=self.device) * 2.0
        self.p_target[:, 2] = self.p_target[:, 2].clamp(1.5, 6.0)
        
        # 随机化参数 - 与 train_adaptive.py 一致
        self.max_speed = self.args.max_speed_min + self.args.max_speed_range * torch.rand(
            (B, 1), device=self.device)
        self.thr_est_error = 1.0 + 0.01 * torch.randn((B, 1), device=self.device)
        
        # 动作缓冲 - 与 train_adaptive.py 一致
        act_lag = 1
        self.act_buffer = [self.env.act_curr.clone() for _ in range(act_lag + 1)]
        
        # GRU 隐藏状态
        self.h = None
        
        # 步数计数
        self.step_count = 0
        self.episode_count += 1
        self.episode_collided = False
        self.episode_reached = False
        
        # 重置轨迹记录
        self.trajectory_history = []
        self.velocity_history = []
        self.clearance_history = []
        self.p_start = self.env.p[0].cpu().numpy().copy()
        
        print(f"\n{'='*50}")
        print(f"Episode {self.episode_count}")
        print(f"{'='*50}")
        print(f"初始位置: [{self.env.p[0, 0].item():.2f}, {self.env.p[0, 1].item():.2f}, {self.env.p[0, 2].item():.2f}]")
        print(f"目标位置: [{self.p_target[0, 0].item():.2f}, {self.p_target[0, 1].item():.2f}, {self.p_target[0, 2].item():.2f}]")
        print(f"初始距离: {torch.norm(self.p_target[0] - self.env.p[0]).item():.2f} m")
        print(f"安全半径: {self.env.margin[0].item():.3f} m")
        print(f"最大速度: {self.max_speed[0, 0].item():.2f} m/s")
    
    def _save_episode_metrics(self):
        """保存单个 episode 的指标"""
        if len(self.trajectory_history) < 2:
            return
        
        trajectory = np.array(self.trajectory_history)
        target = self.p_target[0].cpu().numpy()
        
        metrics = {
            'episode': self.episode_count,
            'steps': len(trajectory),
            'distance_to_target': float(np.linalg.norm(trajectory[-1] - target)),
            'path_length': float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))),
            'min_clearance': float(np.min(self.clearance_history)) if self.clearance_history else 0,
            'avg_speed': float(np.mean(np.linalg.norm(self.velocity_history, axis=1))) if self.velocity_history else 0,
            'collision': self.episode_collided,
            'reached': self.episode_reached,
            'margin': float(self.env.margin[0].item()),
            'max_speed_setting': float(self.max_speed[0, 0].item()),
        }
        self.episode_metrics.append(metrics)
    
    @torch.no_grad()
    def step(self):
        """执行一步模拟 - 与 train_adaptive.py 逻辑对齐"""
        if self.paused:
            return None, None, {}
        
        args = self.args
        B = args.batch_size
        
        current_dt = normalvariate(self.ctl_dt, self.ctl_dt * 0.1) / self.speed_multiplier
        
        # 渲染
        rgb, depth = self.env.render(
            camera_pitch=args.cam_angle,
            return_tensor=True,
            return_rgb=True,
            return_depth=True,
            dt=current_dt
        )
        
        # 目标向量
        target_v_raw = self.p_target - self.env.p
        
        # 执行延迟动作 - 与 train_adaptive.py 一致
        t = self.step_count
        if t < len(self.act_buffer):
            self.env.step(act_cmd=self.act_buffer[t], target_pos_vector=target_v_raw, dt=current_dt)
        else:
            self.env.step(act_cmd=self.act_buffer[-1], target_pos_vector=target_v_raw, dt=current_dt)
        
        # 计算局部坐标系
        R_local = self._compute_local_R()
        
        # 目标速度计算 - 与 train_adaptive.py 一致
        target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
        target_v_unit = target_v_raw / (target_v_norm + 1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, self.max_speed)
        
        # 构建状态 - 与 train_adaptive.py 一致
        target_v_local = torch.squeeze(target_v[:, None] @ R_local, 1)
        local_v = torch.squeeze(self.env.v[:, None] @ R_local, 1)
        
        state_parts = [
            target_v_local,
            self.env.R[:, 2],
            self.env.margin[:, None]
        ]
        if not self.no_odom:
            state_parts.insert(0, local_v)
        state = torch.cat(state_parts, dim=-1)
        
        # 深度图预处理 - 与 train_adaptive.py 一致
        x = depth.clamp(0.3, 24.0)
        x = 3.0 / x - 0.6
        x = x + torch.randn_like(x) * 0.02
        x = x.unsqueeze(1)  # Model_adaptive 不需要池化
        
        # 模型推理
        act_raw, _, self.h = self.model(x, state, self.h)
        
        # 解析输出 - 与 train_adaptive.py 一致
        act_reshaped = act_raw.reshape(B, 3, 2)
        act_world = R_local @ act_reshaped
        a_pred, v_pred = act_world.unbind(-1)
        
        # 计算动作
        act = (a_pred - v_pred - self.g_std) * self.thr_est_error + self.g_std
        self.act_buffer.append(act)
        
        # 计算指标
        distance_to_target = torch.norm(target_v_raw, dim=-1)
        distance_to_obstacle = self.env.distance_to_obj()
        margin = self.env.margin
        clearance = distance_to_obstacle - margin
        speed = torch.norm(self.env.v, dim=-1)
        
        # 碰撞检测
        if (clearance < 0).any() and not self.episode_collided:
            self.collision_count += 1
            self.episode_collided = True
            print(f"⚠️  碰撞发生于步骤 {self.step_count}!")
        
        # 到达目标检测
        if (distance_to_target < 0.5).any() and not self.episode_reached:
            self.episode_reached = True
            self.success_count += 1
            print(f"✓ 到达目标! 步骤: {self.step_count}")
        
        self.step_count += 1
        
        # 记录轨迹
        self.trajectory_history.append(self.env.p[0].cpu().numpy().copy())
        self.velocity_history.append(self.env.v[0].cpu().numpy().copy())
        self.clearance_history.append(clearance[0].item())
        
        metrics = {
            'position': self.env.p[0].cpu().numpy(),
            'velocity': self.env.v[0].cpu().numpy(),
            'target': self.p_target[0].cpu().numpy(),
            'distance_to_target': distance_to_target[0].item(),
            'distance_to_obstacle': distance_to_obstacle[0].item(),
            'margin': margin[0].item(),
            'clearance': clearance[0].item(),
            'speed': speed[0].item(),
            'reached': self.episode_reached,
            'collided': self.episode_collided,
            'step': self.step_count,
        }
        
        return rgb, depth, metrics
    
    def render_display(self, rgb, depth, metrics):
        """渲染可视化显示"""
        args = self.args
        scale = args.display_scale
        img_h = args.image_height * scale
        img_w = args.image_width * scale
        
        images = []
        
        # RGB 图像
        if self.display_mode in ['rgb', 'both'] and rgb is not None:
            rgb_np = rgb[0].cpu().numpy()
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            rgb_display = cv2.resize(rgb_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            cv2.putText(rgb_display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
            images.append(rgb_display)
        
        # 深度图
        if self.display_mode in ['depth', 'both'] and depth is not None:
            depth_np = depth[0].cpu().numpy()
            depth_vis = np.clip(depth_np, 0.3, 10.0)
            depth_vis = ((depth_vis - 0.3) / (10.0 - 0.3) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            depth_color = cv2.resize(depth_color, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            cv2.putText(depth_color, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)
            images.append(depth_color)
        
        # 合并图像
        if images:
            images_combined = np.hstack(images)
        else:
            images_combined = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # 信息面板
        panel_width = 450
        panel_height = img_h
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        line_h = 24
        
        def put(text, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(panel, text, (15, y), font, 0.55, color, 1, cv2.LINE_AA)
            y += line_h
        
        # 标题
        put(f"=== Episode {self.episode_count} | Step {metrics.get('step', 0)} ===", (0, 255, 255))
        y += 5
        
        # 状态
        if metrics.get('reached'):
            put("STATUS: REACHED TARGET", (0, 255, 0))
        elif metrics.get('collided'):
            put("STATUS: COLLISION", (0, 0, 255))
        else:
            put("STATUS: IN PROGRESS", (255, 255, 0))
        y += 10
        
        # 位置
        pos = metrics.get('position', [0, 0, 0])
        put(f"Position: ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")
        
        # 速度
        vel = metrics.get('velocity', [0, 0, 0])
        speed = metrics.get('speed', 0)
        put(f"Velocity: ({vel[0]:7.2f}, {vel[1]:7.2f}, {vel[2]:7.2f})")
        color = (0, 255, 0) if speed < self.max_speed[0, 0].item() else (0, 165, 255)
        put(f"Speed: {speed:.2f} / {self.max_speed[0, 0].item():.2f} m/s", color)
        y += 10
        
        # 目标
        target = metrics.get('target', [0, 0, 0])
        dist_target = metrics.get('distance_to_target', 0)
        put(f"Target: ({target[0]:7.2f}, {target[1]:7.2f}, {target[2]:7.2f})", (255, 255, 0))
        color = (0, 255, 0) if dist_target < 1 else (255, 255, 255)
        put(f"Distance to target: {dist_target:.3f} m", color)
        y += 10
        
        # 安全信息
        margin = metrics.get('margin', 0)
        clearance = metrics.get('clearance', 0)
        dist_obs = metrics.get('distance_to_obstacle', 0)
        
        put(f"Safety margin: {margin:.3f} m")
        put(f"Distance to obstacle: {dist_obs:.3f} m")
        
        if clearance > margin:
            color, status = (0, 255, 0), "SAFE"
        elif clearance > 0:
            color, status = (0, 165, 255), "WARNING"
        else:
            color, status = (0, 0, 255), "DANGER"
        put(f"Clearance: {clearance:.3f} m [{status}]", color)
        y += 15
        
        # 统计
        put("--- Session Statistics ---", (200, 200, 200))
        put(f"Episodes: {self.episode_count}")
        put(f"Successes: {self.success_count} ({100*self.success_count/max(1,self.episode_count):.1f}%)",
            (0, 255, 0) if self.success_count > 0 else (255, 255, 255))
        put(f"Collisions: {self.collision_count} ({100*self.collision_count/max(1,self.episode_count):.1f}%)",
            (0, 0, 255) if self.collision_count > 0 else (255, 255, 255))
        put(f"Timeouts: {self.timeout_count}")
        y += 15
        
        # 模型信息
        put("--- Model Config ---", (200, 200, 200))
        put(f"no_odom: {self.no_odom}")
        put(f"Speed multiplier: x{self.speed_multiplier:.1f}")
        y += 15
        
        # 控制提示
        put("--- Controls ---", (150, 150, 150))
        put("R: Reset | SPACE: Pause | Q: Quit", (120, 120, 120))
        put("+/-: Speed | S: Save | V: View mode", (120, 120, 120))
        
        if self.paused:
            cv2.putText(panel, "PAUSED", (panel_width // 2 - 60, panel_height // 2),
                       font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
        
        display = np.hstack([images_combined, panel])
        return display
    
    def run(self):
        """主运行循环"""
        print("\n" + "=" * 60)
        print("Model_adaptive Test Session")
        print("=" * 60)
        print(f"Model: {self.args.model_path}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"no_odom: {self.no_odom}")
        print(f"Display: {'Disabled' if self.args.no_display else 'Enabled'}")
        print("=" * 60)
        
        self.reset_episode(save_previous=False)
        
        frame_delay = int(1000 / self.args.fps_limit)
        
        try:
            while True:
                # 检查 episode 限制
                if self.args.num_episodes > 0 and self.episode_count > self.args.num_episodes:
                    print(f"\n达到测试 episode 上限 ({self.args.num_episodes})")
                    break
                
                # 执行一步
                rgb, depth, metrics = self.step()
                
                # 显示
                if not self.args.no_display:
                    display = self.render_display(rgb, depth, metrics)
                    cv2.imshow(self.window_name, display)
                
                # 检查是否需要重置
                episode_done = False
                if self.step_count >= self.args.timesteps:
                    if not self.episode_reached and not self.episode_collided:
                        self.timeout_count += 1
                        print(f"○ Episode 超时 ({self.args.timesteps} 步)")
                    episode_done = True
                elif self.episode_reached:
                    episode_done = True
                
                if episode_done:
                    self.reset_episode()
                
                # 按键处理
                if not self.args.no_display:
                    key = cv2.waitKey(frame_delay) & 0xFF
                    
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('r'):
                        print("手动重置")
                        self.reset_episode()
                    elif key == ord(' '):
                        self.paused = not self.paused
                        print("暂停" if self.paused else "继续")
                    elif key == ord('+') or key == ord('='):
                        self.speed_multiplier = min(self.speed_multiplier * 1.5, 5.0)
                        print(f"速度: x{self.speed_multiplier:.1f}")
                    elif key == ord('-'):
                        self.speed_multiplier = max(self.speed_multiplier / 1.5, 0.2)
                        print(f"速度: x{self.speed_multiplier:.1f}")
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"manual_ep{self.episode_count:04d}_{timestamp}.png"
                        save_path = os.path.join(self.args.trajectory_dir, filename)
                        self.plot_trajectory(save_path)
                    elif key == ord('v'):
                        modes = ['rgb', 'depth', 'both']
                        idx = modes.index(self.display_mode)
                        self.display_mode = modes[(idx + 1) % 3]
                        print(f"显示模式: {self.display_mode}")
                else:
                    # 无显示模式下的简单延迟
                    pass
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            # 保存最后一个 episode
            if self.args.save_trajectory and len(self.trajectory_history) > 1:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"adaptive_ep{self.episode_count:04d}_{timestamp}_final.png"
                save_path = os.path.join(self.args.trajectory_dir, filename)
                self.plot_trajectory(save_path)
            
            # 保存指标汇总
            if self.args.log_metrics and self.episode_metrics:
                self._save_metrics_summary()
            
            if not self.args.no_display:
                cv2.destroyAllWindows()
        
        # 打印最终统计
        self._print_summary()
    
    def _save_metrics_summary(self):
        """保存指标汇总"""
        import json
        
        if self.args.metrics_file:
            filepath = self.args.metrics_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.args.trajectory_dir, f'metrics_{timestamp}.json')
        
        summary = {
            'model_path': self.args.model_path,
            'no_odom': self.no_odom,
            'total_episodes': self.episode_count,
            'successes': self.success_count,
            'collisions': self.collision_count,
            'timeouts': self.timeout_count,
            'success_rate': self.success_count / max(1, self.episode_count),
            'collision_rate': self.collision_count / max(1, self.episode_count),
            'episodes': self.episode_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"指标汇总已保存: {filepath}")
    
    def _print_summary(self):
        """打印最终统计"""
        print("\n" + "=" * 60)
        print("Test Session Summary")
        print("=" * 60)
        print(f"Total episodes:  {self.episode_count}")
        print(f"Successes:       {self.success_count} ({100*self.success_count/max(1,self.episode_count):.1f}%)")
        print(f"Collisions:      {self.collision_count} ({100*self.collision_count/max(1,self.episode_count):.1f}%)")
        print(f"Timeouts:        {self.timeout_count} ({100*self.timeout_count/max(1,self.episode_count):.1f}%)")
        
        if self.episode_metrics:
            avg_path = np.mean([m['path_length'] for m in self.episode_metrics])
            avg_speed = np.mean([m['avg_speed'] for m in self.episode_metrics])
            avg_clearance = np.mean([m['min_clearance'] for m in self.episode_metrics])
            print(f"\nAverage path length:   {avg_path:.2f} m")
            print(f"Average speed:         {avg_speed:.2f} m/s")
            print(f"Average min clearance: {avg_clearance:.3f} m")
        
        print("=" * 60)


def main():
    args = parse_args()
    tester = AdaptiveModelTester(args)
    tester.run()


if __name__ == '__main__':
    main()
