"""
无人机模型可视化测试脚本

实时显示无人机在训练环境中的行为，包括：
- 深度图渲染
- 无人机位置/速度/姿态
- 目标点和安全边距
- 碰撞检测信息

使用方法:
    python test_visualize.py --model_path ./checkpoints/checkpoint_final.pth
    python test_visualize.py --model_path ./checkpoints/checkpoint_final.pth --batch_size 4

按键控制:
    ESC / Q: 退出
    R: 重置环境
    SPACE: 暂停/继续
    +/-: 调整速度
"""

import os
import math
import argparse
from random import normalvariate
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drone_env import DroneSimulator
from model import Model, Model_bigger


def parse_args():
    parser = argparse.ArgumentParser(description='无人机模型可视化测试')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model_type', type=str, default='bigger', choices=['small', 'bigger'],
                        help='模型类型: small 或 bigger')
    
    # 环境参数
    parser.add_argument('--batch_size', type=int, default=1, help='同时测试的无人机数量 (1-4 推荐)')
    parser.add_argument('--timesteps', type=int, default=500, help='每个episode的最大步数')
    parser.add_argument('--ctl_dt', type=float, default=1/15, help='控制时间步长')
    
    # 渲染参数
    parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角')
    parser.add_argument('--image_height', type=int, default=48, help='渲染图像高度')
    parser.add_argument('--image_width', type=int, default=64, help='渲染图像宽度')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj', help='障碍物网格路径')
    parser.add_argument('--num_samples', type=int, default=100000, help='障碍物点云采样数')
    
    # 无人机参数
    parser.add_argument('--margin_min', type=float, default=0.1, help='安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.7, help='安全半径最大值')
    parser.add_argument('--init_p_range', type=float, default=8.0, help='初始位置范围')
    
    # 显示参数
    parser.add_argument('--display_scale', type=int, default=6, help='深度图显示放大倍数')
    parser.add_argument('--fps_limit', type=int, default=30, help='显示帧率限制')
    
    # 轨迹图参数
    parser.add_argument('--save_trajectory', action='store_true', default=True, help='保存轨迹图')
    parser.add_argument('--trajectory_dir', type=str, default='./trajectory_plots', help='轨迹图保存目录')
    parser.add_argument('--scene_points', type=int, default=5000, help='场景点云显示数量')
    
    # 模型参数
    parser.add_argument('--no_odom', default=False, action='store_true', help='不使用里程计速度')
    
    # 硬件
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    return parser.parse_args()


class DroneVisualizer:
    """无人机可视化测试器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.ctl_dt = args.ctl_dt
        self.paused = False
        self.speed_multiplier = 1.0
        
        # 初始化环境
        self.env = DroneSimulator(
            batch_size=args.batch_size,
            dt=self.ctl_dt,
            mesh_path=args.mesh_path,
            image_size=(args.image_height, args.image_width),
            device=self.device,
            enable_airmode=True,
            enable_induced_drag=False,
            noise_std=0.04,
            grad_decay=0.4,
            yaw_inertia=5.0,
            yaw_ctl_delay=12.0,
            pitch_ctl_delay=12.0,
            airmode_coef=0.5,
            init_p_range=args.init_p_range,
            init_margin_range=(args.margin_min, args.margin_max),
            num_samples=args.num_samples
        )
        
        # 初始化模型 - 将在 _load_model 中根据检查点自动设置
        self.model = None
        
        # 加载模型权重（会自动检测模型类型）
        self._load_model(args.model_path)
        self.model.eval()
        
        # 重力向量
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)
        
        # 显示窗口设置
        self.window_name = "Drone Visualization"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 统计信息
        self.episode_count = 0
        self.step_count = 0
        self.collision_count = 0
        self.success_count = 0
        
        # 轨迹记录
        self.trajectory_history = []
        self.p_start = None
        
        # 创建轨迹图保存目录
        if args.save_trajectory:
            os.makedirs(args.trajectory_dir, exist_ok=True)
            print(f"轨迹图将保存到: {args.trajectory_dir}")
        
        # 预采样场景点云用于可视化 (降采样以加快绘制)
        self.scene_points_for_plot = self._sample_scene_points(args.scene_points)
        
    def _load_model(self, path):
        """加载模型权重，自动检测模型类型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        
        # 自动检测模型类型
        # 通过检查 stem.0.weight 的形状来判断
        # Model: stem.0 是 Conv2d(1, 32, 2, 2) -> weight shape: (32, 1, 2, 2)
        # Model_bigger: stem.0 是 Conv2d(1, 32, 3, 2, 1) -> weight shape: (32, 1, 3, 3)
        stem_weight_shape = state_dict['stem.0.weight'].shape
        
        if stem_weight_shape == torch.Size([32, 1, 2, 2]):
            print("Detected model type: Model (small)")
            self.model_type = 'small'
            # 检测 dim_obs
            v_proj_shape = state_dict['v_proj.weight'].shape
            dim_obs = v_proj_shape[1]
            self.model = Model(dim_obs=dim_obs, dim_action=6).to(self.device)
        elif stem_weight_shape == torch.Size([32, 1, 3, 3]):
            print("Detected model type: Model_bigger")
            self.model_type = 'bigger'
            v_proj_shape = state_dict['v_proj.weight'].shape
            dim_obs = v_proj_shape[1]
            self.model = Model_bigger(dim_obs=dim_obs, dim_action=6).to(self.device)
        else:
            raise ValueError(f"Unknown model architecture, stem weight shape: {stem_weight_shape}")
        
        # 更新 no_odom 标志
        if dim_obs == 7:
            self.args.no_odom = True
            print(f"Detected dim_obs={dim_obs}, setting no_odom=True")
        else:
            self.args.no_odom = False
            print(f"Detected dim_obs={dim_obs}, setting no_odom=False")
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning - Missing keys: {missing}")
        if unexpected:
            print(f"Warning - Unexpected keys: {unexpected}")
        print(f"Loaded model from {path}")
    
    def _sample_scene_points(self, num_points):
        """从场景网格采样点云用于可视化"""
        # 使用渲染器中已有的障碍物点云，降采样
        full_pcd = self.env.renderer.obstacle_pcd[0].cpu().numpy()  # (N, 3)
        if len(full_pcd) > num_points:
            indices = np.random.choice(len(full_pcd), num_points, replace=False)
            return full_pcd[indices]
        return full_pcd
    
    def plot_trajectory(self, save_path=None):
        """绘制 3D 轨迹图，包含场景点云"""
        if len(self.trajectory_history) < 2:
            print("轨迹点太少，跳过绘图")
            return
        
        # 转换轨迹数据
        trajectory = np.array(self.trajectory_history)  # (T, 3)
        
        # 创建 3D 图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制场景点云 (灰色，半透明)
        scene_pts = self.scene_points_for_plot
        ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2], 
                   c='gray', s=1, alpha=0.3, label='Scene')
        
        # 绘制轨迹线 (根据时间渐变颜色)
        # 使用颜色映射显示时间进程
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        for i in range(len(trajectory) - 1):
            ax.plot3D(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                     color=colors[i], linewidth=2)
        
        # 绘制起点 (绿色大圆点)
        ax.scatter(*self.p_start, c='lime', s=200, marker='o', 
                   edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
        
        # 绘制终点 (当前位置，红色)
        ax.scatter(*trajectory[-1], c='red', s=200, marker='o',
                   edgecolors='darkred', linewidths=2, label='End', zorder=5)
        
        # 绘制目标点 (黄色星形)
        target = self.p_target[0].cpu().numpy()
        ax.scatter(*target, c='yellow', s=300, marker='*',
                   edgecolors='orange', linewidths=2, label='Target', zorder=5)
        
        # 绘制安全半径球 (在起点和终点)
        margin = self.env.margin[0].item()
        
        # 用虚线连接终点和目标
        ax.plot3D([trajectory[-1, 0], target[0]], 
                  [trajectory[-1, 1], target[1]], 
                  [trajectory[-1, 2], target[2]],
                  'y--', linewidth=1, alpha=0.7)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        # 计算统计信息
        dist_to_target = np.linalg.norm(trajectory[-1] - target)
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        # 设置标题
        status = "✓ SUCCESS" if dist_to_target < 0.5 else ("✗ COLLISION" if self.episode_collided else "○ TIMEOUT")
        title = f"Episode {self.episode_count} - {status}\n"
        title += f"Distance to target: {dist_to_target:.2f}m | Path length: {total_distance:.2f}m | Steps: {len(trajectory)}"
        ax.set_title(title, fontsize=14)
        
        # 图例
        ax.legend(loc='upper left', fontsize=10)
        
        # 设置等比例坐标轴
        # 计算边界
        all_points = np.vstack([trajectory, scene_pts, target.reshape(1, 3)])
        max_range = np.max(np.ptp(all_points, axis=0)) / 2
        mid = np.mean(all_points, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(max(0, mid[2] - max_range), mid[2] + max_range)
        
        # 添加颜色条显示时间
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(trajectory)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label('Time Step', fontsize=10)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存: {save_path}")
        
        plt.close(fig)
    
    def _compute_local_R(self):
        """计算局部坐标系旋转矩阵"""
        fwd = self.env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
        fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8),
                          torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(fwd))
        R = torch.stack([fwd, torch.linalg.cross(up, fwd), up], -1)
        return R
    
    def reset_episode(self, save_previous=True):
        """重置一个episode"""
        # 保存上一个 episode 的轨迹图
        if save_previous and self.args.save_trajectory and len(self.trajectory_history) > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"episode_{self.episode_count:04d}_{timestamp}.png"
            save_path = os.path.join(self.args.trajectory_dir, filename)
            self.plot_trajectory(save_path)
        
        self.env.reset()
        self.model.reset()
        
        B = self.args.batch_size
        
        # 生成目标点
        angle = torch.rand(B, device=self.device) * 2 * math.pi
        dist = torch.rand(B, device=self.device) * 5.0 + 3.0
        
        self.p_target = self.env.p.clone()
        self.p_target[:, 0] += torch.cos(angle) * dist
        self.p_target[:, 1] += torch.sin(angle) * dist
        self.p_target[:, 2] += torch.randn(B, device=self.device) * 2.0
        self.p_target[:, 2] = self.p_target[:, 2].clamp(0.5, 6.0)
        
        # 固定参数
        self.max_speed = torch.full((B, 1), 5.0, device=self.device)  # 固定最大速度为 2.0 m/s
        self.thr_est_error = 1.0 + 0.01 * torch.randn((B, 1), device=self.device)
        
        # 动作缓冲
        self.act_buffer = [self.env.act_curr.clone() for _ in range(2)]
        
        # GRU 隐藏状态
        self.h = None
        
        # 步数计数
        self.step_count = 0
        self.episode_count += 1
        self.episode_collided = False
        
        # 重置轨迹记录
        self.trajectory_history = []
        self.p_start = self.env.p[0].cpu().numpy().copy()
        
        print(f"\n=== Episode {self.episode_count} ===")
        print(f"初始位置: {self.env.p[0].cpu().numpy()}")
        print(f"目标位置:  {self.p_target[0].cpu().numpy()}")
        print(f"安全半径: {self.env.margin[0].item():.3f}")
        print(f"最大速度: {self.max_speed[0, 0].item():.2f}")
    
    @torch.no_grad()
    def step(self):
        """执行一步模拟"""
        if self.paused:
            return None, None, {}
        
        args = self.args
        B = args.batch_size
        
        current_dt = normalvariate(self.ctl_dt, self.ctl_dt * 0.1) / self.speed_multiplier
        
        # 渲染 RGB 和深度图
        rgb, depth = self.env.render(
            camera_pitch=args.cam_angle,
            return_tensor=True,
            return_rgb=True,
            return_depth=True,
            dt=current_dt
        )
        
        # 目标向量
        target_v_raw = self.p_target - self.env.p
        
        # 执行延迟动作
        t = self.step_count
        if t < len(self.act_buffer):
            self.env.step(act_cmd=self.act_buffer[t], target_pos_vector=target_v_raw, dt=current_dt)
        else:
            self.env.step(act_cmd=self.act_buffer[-1], target_pos_vector=target_v_raw, dt=current_dt)
        
        # 计算局部坐标系
        R_local = self._compute_local_R()
        
        # 目标速度
        target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
        target_v_unit = target_v_raw / (target_v_norm + 1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, self.max_speed)
        
        # 构建状态
        target_v_local = torch.squeeze(target_v[:, None] @ R_local, 1)
        local_v = torch.squeeze(self.env.v[:, None] @ R_local, 1)
        
        state_parts = [
            target_v_local,
            self.env.R[:, 2],
            self.env.margin[:, None]
        ]
        if not args.no_odom:
            state_parts.insert(0, local_v)
        state = torch.cat(state_parts, dim=-1)
        
        # 深度图预处理
        x = depth.clamp(0.3, 24.0)
        x = 3.0 / x - 0.6
        x = x + torch.randn_like(x) * 0.02  # 添加噪声（与训练一致）
        
        # 根据模型类型决定是否下采样
        if self.model_type == 'small':
            # 小模型需要 4x 下采样: 48x64 -> 12x16
            x = F.max_pool2d(x[:, None], kernel_size=4, stride=4)
        else:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # 模型推理
        act_raw, _, self.h = self.model(x, state, self.h)
        
        # 解析输出
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
            print(f"⚠️  丫在 {self.step_count}步撞了球了!")
        
        # 到达目标检测
        reached = distance_to_target < 0.5
        
        self.step_count += 1
        
        # 记录轨迹点
        self.trajectory_history.append(self.env.p[0].cpu().numpy().copy())
        
        metrics = {
            'position': self.env.p[0].cpu().numpy(),
            'velocity': self.env.v[0].cpu().numpy(),
            'target': self.p_target[0].cpu().numpy(),
            'distance_to_target': distance_to_target[0].item(),
            'distance_to_obstacle': distance_to_obstacle[0].item(),
            'margin': margin[0].item(),
            'clearance': clearance[0].item(),
            'speed': speed[0].item(),
            'reached': reached[0].item(),
            'step': self.step_count,
        }
        
        return rgb, depth, metrics
    
    def render_display(self, rgb, depth, metrics):
        """渲染可视化显示"""
        args = self.args
        scale = args.display_scale
        img_h = args.image_height * scale
        img_w = args.image_width * scale
        
        # 处理 RGB 图像显示
        if rgb is not None:
            rgb_np = rgb[0].cpu().numpy()
            # PyTorch3D 输出是 [0, 1] 范围的浮点数
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            # RGB -> BGR (OpenCV 格式)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            # 放大
            rgb_display = cv2.resize(rgb_bgr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_display = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # 处理深度图显示
        if depth is not None:
            depth_np = depth[0].cpu().numpy()
            # 归一化到 0-255
            depth_vis = np.clip(depth_np, 0.3, 10.0)
            depth_vis = (depth_vis - 0.3) / (10.0 - 0.3) * 255
            depth_vis = depth_vis.astype(np.uint8)
            # 应用颜色映射
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            # 放大
            depth_color = cv2.resize(depth_color, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_color = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # 在图像上添加标签
        cv2.putText(rgb_display, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_color, "Depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 将 RGB 和深度图并排显示
        images_combined = np.hstack([rgb_display, depth_color])
        
        # 创建信息面板
        panel_width = 400
        panel_height = args.image_height * scale
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # 深灰色背景
        
        # 绘制信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 22
        y_offset = 25
        
        def put_text(text, y, color=(255, 255, 255)):
            cv2.putText(panel, text, (10, y), font, font_scale, color, 1, cv2.LINE_AA)
        
        # 状态信息
        put_text(f"=== Episode {self.episode_count} | Step {metrics.get('step', 0)} ===", y_offset, (0, 255, 255))
        y_offset += line_height + 5
        
        # 位置信息
        pos = metrics.get('position', [0, 0, 0])
        put_text(f"Position: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})", y_offset)
        y_offset += line_height
        
        # 速度信息
        vel = metrics.get('velocity', [0, 0, 0])
        speed = metrics.get('speed', 0)
        put_text(f"Velocity: ({vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f})", y_offset)
        y_offset += line_height
        put_text(f"Speed: {speed:.2f} m/s", y_offset, (0, 255, 0) if speed < 3 else (0, 165, 255))
        y_offset += line_height + 5
        
        # 目标信息
        target = metrics.get('target', [0, 0, 0])
        dist_target = metrics.get('distance_to_target', 0)
        put_text(f"Target: ({target[0]:6.2f}, {target[1]:6.2f}, {target[2]:6.2f})", y_offset, (255, 255, 0))
        y_offset += line_height
        color = (0, 255, 0) if dist_target < 1 else (255, 255, 255)
        put_text(f"Distance to target: {dist_target:.2f} m", y_offset, color)
        y_offset += line_height + 5
        
        # 安全信息
        margin = metrics.get('margin', 0)
        dist_obs = metrics.get('distance_to_obstacle', 0)
        clearance = metrics.get('clearance', 0)
        
        put_text(f"Safety margin: {margin:.3f} m", y_offset)
        y_offset += line_height
        put_text(f"Distance to obstacle: {dist_obs:.3f} m", y_offset)
        y_offset += line_height
        
        # 间隙状态 - 带颜色指示
        if clearance > margin:
            color = (0, 255, 0)  # 绿色 - 安全
            status = "SAFE"
        elif clearance > 0:
            color = (0, 165, 255)  # 橙色 - 警告
            status = "WARNING"
        else:
            color = (0, 0, 255)  # 红色 - 危险
            status = "DANGER"
        put_text(f"Clearance: {clearance:.3f} m [{status}]", y_offset, color)
        y_offset += line_height + 10
        
        # 统计信息
        put_text(f"--- Statistics ---", y_offset, (200, 200, 200))
        y_offset += line_height
        put_text(f"Total episodes: {self.episode_count}", y_offset)
        y_offset += line_height
        put_text(f"Collisions: {self.collision_count}", y_offset, 
                 (0, 0, 255) if self.collision_count > 0 else (255, 255, 255))
        y_offset += line_height + 10
        
        # 控制提示
        put_text("--- Controls ---", y_offset, (200, 200, 200))
        y_offset += line_height
        put_text("R: Reset | SPACE: Pause | Q/ESC: Quit", y_offset, (150, 150, 150))
        y_offset += line_height
        put_text(f"+/-: Speed x{self.speed_multiplier:.1f}", y_offset, (150, 150, 150))
        
        if self.paused:
            # 暂停指示
            cv2.putText(panel, "PAUSED", (panel_width // 2 - 50, panel_height // 2),
                        font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 合并图像和信息面板：[RGB | Depth | Panel]
        display = np.hstack([images_combined, panel])
        
        return display
    
    def run(self):
        """主运行循环"""
        print("\n" + "=" * 50)
        print("Drone Visualization Test")
        print("=" * 50)
        print(f"Model: {self.args.model_path}")
        print(f"Batch size: {self.args.batch_size}")
        print("Press 'R' to reset, 'SPACE' to pause, 'Q' to quit")
        print("=" * 50)
        
        self.reset_episode(save_previous=False)  # 第一次不保存（没有上一个episode）
        
        frame_delay = int(1000 / self.args.fps_limit)
        
        try:
            while True:
                # 执行一步
                rgb, depth, metrics = self.step()
                
                # 渲染显示
                display = self.render_display(rgb, depth, metrics)
                cv2.imshow(self.window_name, display)
                
                # 检查是否需要重置
                if self.step_count >= self.args.timesteps:
                    print(f"Episode ended after {self.step_count} steps")
                    if metrics.get('reached', False):
                        self.success_count += 1
                        print("✓ 抵达目标！")
                    self.reset_episode()
                
                # 按键处理
                key = cv2.waitKey(frame_delay) & 0xFF
                
                if key == ord('q') or key == 27:  # Q 或 ESC
                    break
                elif key == ord('r'):  # R - 重置
                    print("Manual reset")
                    self.reset_episode()
                elif key == ord(' '):  # 空格 - 暂停
                    self.paused = not self.paused
                    print("暂停" if self.paused else "继续")
                elif key == ord('+') or key == ord('='):  # 加速
                    self.speed_multiplier = min(self.speed_multiplier * 1.5, 5.0)
                    print(f"速度: x{self.speed_multiplier:.1f}")
                elif key == ord('-'):  # 减速
                    self.speed_multiplier = max(self.speed_multiplier / 1.5, 0.2)
                    print(f"速度: x{self.speed_multiplier:.1f}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # 保存最后一个 episode 的轨迹图
            if self.args.save_trajectory and len(self.trajectory_history) > 1:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"episode_{self.episode_count:04d}_{timestamp}_final.png"
                save_path = os.path.join(self.args.trajectory_dir, filename)
                self.plot_trajectory(save_path)
            
            cv2.destroyAllWindows()
            
        # 打印最终统计
        print("\n" + "=" * 50)
        print("Final Statistics")
        print("=" * 50)
        print(f"Total episodes: {self.episode_count}")
        print(f"Collisions: {self.collision_count}")
        print(f"Success rate: {self.success_count}/{self.episode_count}")
        print("=" * 50)


def main():
    args = parse_args()
    visualizer = DroneVisualizer(args)
    visualizer.run()


if __name__ == '__main__':
    main()
