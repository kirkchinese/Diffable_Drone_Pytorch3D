"""
无人机可微分仿真训练脚本 - 动态场景版本

支持动态障碍物的训练和可视化。
使用 DynamicSceneRenderer 进行场景渲染。

作者: KirkChinese
日期: 2026-01-12
"""

import os
import math
import argparse
from collections import defaultdict
from random import normalvariate
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from drone_env import DroneSimulator
from drone_renderer_dynamic import DynamicSceneRenderer, DynamicDroneSimulator
from model import Model
from loss import DroneLoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机动态场景避障训练脚本')
    
    # 训练参数
    parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_iters', type=int, default=5000, help='训练迭代次数')
    parser.add_argument('--timesteps', type=int, default=100, help='每次迭代的模拟步数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减系数')
    parser.add_argument('--ctl_dt', type=float, default=1/15, help='控制时间步长 (秒)')
    
    # 损失函数权重
    parser.add_argument('--coef_v', type=float, default=1.0, help='速度跟踪损失权重')
    parser.add_argument('--coef_speed', type=float, default=0.0, help='速度损失权重')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度预测损失权重')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞损失权重')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='障碍物回避损失权重')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='加速度正则化权重')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='加加速度正则化权重')
    parser.add_argument('--coef_d_snap', type=float, default=0.0, help='snap正则化权重')
    parser.add_argument('--coef_ground_affinity', type=float, default=0.0, help='地面亲和损失权重')
    parser.add_argument('--coef_bias', type=float, default=0.0, help='方向偏差损失权重')
    parser.add_argument('--window_size', type=int, default=30, help='速度平均窗口大小')
    
    # 环境参数 - 渲染
    parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角')
    parser.add_argument('--image_height', type=int, default=48, help='图像高度')
    parser.add_argument('--image_width', type=int, default=64, help='图像宽度')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj', help='静态场景网格路径')
    parser.add_argument('--num_samples', type=int, default=100000, help='障碍物点云采样数')
    
    # 动态障碍物参数
    parser.add_argument('--num_dynamic_obs', type=int, default=3, help='动态障碍物数量')
    parser.add_argument('--obs_pos_range', type=float, default=3.0, help='动态障碍物位置范围')
    parser.add_argument('--obs_vel_range', type=float, default=0.3, help='动态障碍物速度范围')
    parser.add_argument('--obs_scale_min', type=float, default=0.2, help='动态障碍物最小缩放')
    parser.add_argument('--obs_scale_max', type=float, default=0.6, help='动态障碍物最大缩放')
    parser.add_argument('--randomize_each_episode', action='store_true', default=True, 
                        help='每个 episode 随机化障碍物')
    
    # 环境参数 - 无人机物理
    parser.add_argument('--margin_min', type=float, default=0.1, help='无人机安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.3, help='无人机安全半径最大值')
    parser.add_argument('--init_p_range', type=float, default=2.0, help='初始位置范围')
    parser.add_argument('--noise_std', type=float, default=0.04, help='环境扰动噪声标准差')
    parser.add_argument('--yaw_inertia', type=float, default=5.0, help='偏航惯性')
    parser.add_argument('--yaw_ctl_delay', type=float, default=12.0, help='偏航控制延迟')
    parser.add_argument('--pitch_ctl_delay', type=float, default=12.0, help='俯仰控制延迟')
    parser.add_argument('--airmode_coef', type=float, default=0.5, help='Airmode 系数')
    parser.add_argument('--enable_airmode', action='store_true', default=True, help='启用 Airmode')
    parser.add_argument('--disable_airmode', action='store_true', default=False, help='禁用 Airmode')
    
    # 模型参数
    parser.add_argument('--no_odom', default=False, action='store_true', help='不使用里程计速度作为输入')
    parser.add_argument('--yaw_drift', default=False, action='store_true', help='启用航向漂移')
    parser.add_argument('--debug', default=False, action='store_true', help='启用 anomaly detection 调试模式')
    
    # 保存和可视化参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--visualize_interval', type=int, default=500, help='可视化间隔')
    parser.add_argument('--save_video', action='store_true', default=False, help='保存可视化视频')
    
    return parser.parse_args()


def is_save_iter(i):
    """判断是否为保存迭代"""
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


class DynamicDroneTrainer:
    """支持动态障碍物的无人机训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 控制时间步长
        self.ctl_dt = getattr(args, 'ctl_dt', 1/15)
        
        # 处理 airmode 开关
        enable_airmode = getattr(args, 'enable_airmode', True)
        if getattr(args, 'disable_airmode', False):
            enable_airmode = False
        
        # ============ 初始化基础环境 ============
        self.base_env = DroneSimulator(
            batch_size=args.batch_size,
            dt=self.ctl_dt,
            mesh_path=args.mesh_path,
            image_size=(args.image_height, args.image_width),
            device=self.device,
            enable_airmode=enable_airmode,
            enable_induced_drag=False,
            noise_std=getattr(args, 'noise_std', 0.04),
            grad_decay=args.grad_decay,
            yaw_inertia=getattr(args, 'yaw_inertia', 5.0),
            yaw_ctl_delay=getattr(args, 'yaw_ctl_delay', 12.0),
            pitch_ctl_delay=getattr(args, 'pitch_ctl_delay', 12.0),
            airmode_coef=getattr(args, 'airmode_coef', 0.5),
            init_p_range=getattr(args, 'init_p_range', 2.0),
            init_margin_range=(getattr(args, 'margin_min', 0.1), getattr(args, 'margin_max', 0.3)),
            num_samples=args.num_samples
        )
        
        # ============ 创建动态场景渲染器 ============
        self.dynamic_renderer = DynamicSceneRenderer(
            static_mesh_path=args.mesh_path,
            device=self.device,
            image_size=(args.image_height, args.image_width),
            focal_length=500.0,
            num_samples=args.num_samples
        )
        
        # 替换基础环境的渲染器
        self.base_env.renderer = self.dynamic_renderer
        
        # 创建包装器，方便同时更新障碍物状态
        self.env = DynamicDroneSimulator(self.base_env, self.dynamic_renderer)
        
        # ============ 初始化模型 ============
        if args.no_odom:
            self.model = Model(dim_obs=7, dim_action=6).to(self.device)
        else:
            self.model = Model(dim_obs=10, dim_action=6).to(self.device)
        
        # 加载预训练模型
        if args.resume:
            self._load_model(args.resume)
        
        # ============ 优化器和调度器 ============
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.num_iters, eta_min=args.lr * 0.01)
        
        # ============ 损失函数 ============
        self.losser = DroneLoss(
            coef_v=args.coef_v,
            coef_speed=args.coef_speed,
            coef_v_pred=args.coef_v_pred,
            coef_collide=args.coef_collide,
            coef_obj_avoidance=args.coef_obj_avoidance,
            coef_d_acc=args.coef_d_acc,
            coef_d_jerk=args.coef_d_jerk,
            coef_d_snap=args.coef_d_snap,
            coef_ground_affinity=args.coef_ground_affinity,
            coef_bias=args.coef_bias,
            ctl_dt=self.ctl_dt,
            window_size=getattr(args, 'window_size', 30)
        )
        
        # ============ TensorBoard ============
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = os.path.join(args.log_dir, f'drone_dynamic_{timestamp}')
        self.writer = SummaryWriter(log_path)
        print(f"TensorBoard logs: {log_path}")
        
        # 确保保存目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 指标平滑队列
        self.scaler_q = defaultdict(list)
        
        # 重力标准向量
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)
        
        # 可视化数据存储
        self.vis_data = None
        
    def _load_model(self, path):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loaded model from {path}")
    
    def _smooth_dict(self, ori_dict):
        """平滑指标记录"""
        for k, v in ori_dict.items():
            self.scaler_q[k].append(float(v))
    
    def _compute_local_R(self):
        """计算用于速度/目标向量转换的局部坐标系旋转矩阵"""
        fwd = self.base_env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
        fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8), 
                          torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(fwd))
        R = torch.stack([fwd, torch.linalg.cross(up, fwd), up], -1)
        return R
    
    def _randomize_obstacles(self):
        """随机化动态障碍物"""
        args = self.args
        self.dynamic_renderer.randomize_obstacles(
            num_obstacles=args.num_dynamic_obs,
            position_range=(-args.obs_pos_range, args.obs_pos_range),
            velocity_range=(-args.obs_vel_range, args.obs_vel_range),
            scale_range=(args.obs_scale_min, args.obs_scale_max)
        )
    
    def _get_obstacle_positions(self):
        """获取当前所有动态障碍物位置"""
        positions = []
        for obs in self.dynamic_renderer.dynamic_obstacles:
            positions.append(obs.position.cpu().numpy())
        return np.array(positions) if positions else np.array([])
    
    def run_episode(self, iteration, record_vis=False):
        """
        运行一个 episode 并返回损失
        
        Args:
            iteration: 当前迭代次数
            record_vis: 是否记录可视化数据
        """
        args = self.args
        B = args.batch_size
        
        # 重置环境和模型
        self.base_env.reset()
        self.model.reset()
        
        # 随机化动态障碍物
        if args.randomize_each_episode:
            self._randomize_obstacles()
        
        # 历史记录
        p_history = []
        v_history = []
        target_v_history = []
        vec_to_pt_history = []
        v_preds = []
        vid = []
        rgb_vid = []
        obstacle_positions_history = []
        
        # GRU 隐藏状态
        h = None
        
        # 动作延迟缓冲
        act_lag = 1
        initial_act = self.base_env.act_curr.clone()
        act_buffer = [initial_act.clone() for _ in range(act_lag + 1)]
        
        # 目标位置
        p_target = torch.rand(B, 3, device=self.device) * 18.0 - 9.0
        target_v_raw = p_target - self.base_env.p
        
        # 个体随机化最大速度
        max_speed = 0.75 + 2.5 * torch.rand((B, 1), device=self.device)
        
        # 推力估计误差
        thr_est_error = 1.0 + 0.1 * torch.randn((B, 1), device=self.device)
        
        for t in range(args.timesteps):
            # 随机化控制间隔
            current_dt = normalvariate(self.ctl_dt, self.ctl_dt * 0.1)
            
            # 更新动态障碍物位置
            self.dynamic_renderer.step_obstacles(current_dt)
            
            # 渲染深度图
            with torch.no_grad():
                rgb, depth = self.base_env.render(
                    camera_pitch=args.cam_angle,
                    return_tensor=True,
                    return_rgb=True,
                    return_depth=True,
                    dt=current_dt
                )
            depth = depth.requires_grad_(False)
            
            # 记录状态
            p_history.append(self.base_env.p.clone())
            v_history.append(self.base_env.v.clone())
            vec_to_pt_history.append(self.base_env.vec_to_obj())
            
            # 记录可视化数据
            if record_vis:
                vid.append(depth[0].cpu())
                rgb_vid.append(rgb[0].cpu())
                obstacle_positions_history.append(self._get_obstacle_positions())
            elif is_save_iter(iteration) and B > 4:
                vid.append(depth[4])
            
            # 更新目标向量
            target_v_raw = p_target - self.base_env.p.detach()
            
            # 执行动作
            self.base_env.step(act_cmd=act_buffer[t], 
                              target_pos_vector=target_v_raw, 
                              dt=current_dt)
            
            # 计算局部坐标系
            R_local = self._compute_local_R()
            
            # 计算目标速度向量
            target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
            target_v_unit = target_v_raw / (target_v_norm + 1e-6)
            target_v = target_v_unit * torch.minimum(target_v_norm, max_speed)
            target_v_history.append(target_v)
            
            # 构建观测状态
            target_v_local = torch.squeeze(target_v[:, None] @ R_local, 1)
            local_v = torch.squeeze(self.base_env.v[:, None] @ R_local, 1)
            
            state_parts = [
                target_v_local,
                self.base_env.R[:, 2],
                self.base_env.margin[:, None]
            ]
            if not args.no_odom:
                state_parts.insert(0, local_v)
            state = torch.cat(state_parts, dim=-1)
            
            # 深度图预处理
            x = depth.clamp(0.3, 24.0)
            x = 3.0 / x - 0.6
            x = x + torch.randn_like(x) * 0.02
            x = F.max_pool2d(x[:, None], kernel_size=4, stride=4)
            
            # 模型推理
            act_raw, _, h = self.model(x, state, h)
            
            # 解析输出
            act_reshaped = act_raw.reshape(B, 2, 3).permute(0, 2, 1)
            act_world = R_local @ act_reshaped
            a_pred, v_pred = act_world.unbind(-1)
            
            v_preds.append(v_pred)
            
            # 计算实际动作
            act = (a_pred - v_pred - self.g_std) * thr_est_error + self.g_std
            act_buffer.append(act)
        
        # 堆叠历史
        p_history = torch.stack(p_history)
        v_history = torch.stack(v_history)
        target_v_history = torch.stack(target_v_history)
        vec_to_pt_history = torch.stack(vec_to_pt_history)
        v_preds = torch.stack(v_preds)
        act_buffer_stacked = torch.stack(act_buffer)
        
        # 计算损失
        loss, metrics = self.losser.forward(
            p_history=p_history,
            v_history=v_history,
            target_vel_history=target_v_history,
            act_history=act_buffer_stacked,
            vec_to_obj_history=vec_to_pt_history,
            v_preds=v_preds,
            env_margin=self.base_env.margin,
            env_g_std=self.g_std
        )
        
        # 计算额外指标
        with torch.no_grad():
            distance = torch.norm(vec_to_pt_history, 2, -1) - self.base_env.margin
            speed_history = v_history.norm(2, -1)
            avg_speed = speed_history.mean(0)
            success = torch.all(distance.flatten(0, 1) > 0, 0)
            success_rate = success.float().mean()
            
            metrics['success_rate'] = success_rate
            metrics['avg_speed'] = avg_speed.mean()
            metrics['max_speed'] = speed_history.max(0).values.mean()
            metrics['ar'] = (success.float() * avg_speed).mean()
        
        # 可视化数据
        vis_data = {
            'p_history': p_history,
            'v_history': v_history,
            'act_history': act_buffer_stacked,
            'vid': vid,
            'rgb_vid': rgb_vid,
            'obstacle_positions': obstacle_positions_history,
            'target_pos': p_target
        }
        
        return loss, metrics, vis_data
    
    def train(self):
        """主训练循环"""
        args = self.args
        
        pbar = tqdm(range(args.num_iters), ncols=100)
        
        for i in pbar:
            # 判断是否记录可视化
            record_vis = (i + 1) % args.visualize_interval == 0
            
            # 运行一个 episode
            loss, metrics, vis_data = self.run_episode(i, record_vis=record_vis)
            
            # 检查 NaN
            if torch.isnan(loss):
                print("Loss is NaN, exiting...")
                break
            
            # 更新进度条
            pbar.set_description(f'loss: {loss:.3f} sr: {metrics["success_rate"]:.2f}')
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录指标
            self._smooth_dict({
                'loss': loss,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))}
            })
            
            # 定期日志和保存
            if is_save_iter(i):
                self._log_figures(i, vis_data)
            
            # 可视化
            if record_vis:
                self.vis_data = vis_data
                self._visualize_episode(i, vis_data)
            
            if (i + 1) % 10000 == 0:
                save_path = os.path.join(args.save_dir, f'checkpoint_dynamic_{(i+1)//10000:04d}.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"\nSaved model to {save_path}")
            
            if (i + 1) % 25 == 0:
                for k, v in self.scaler_q.items():
                    self.writer.add_scalar(k, sum(v) / len(v), i + 1)
                self.scaler_q.clear()
        
        # 保存最终模型
        final_path = os.path.join(args.save_dir, 'checkpoint_dynamic_final.pth')
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")
        
        self.writer.close()
    
    def _log_figures(self, iteration, vis_data):
        """记录可视化图表"""
        p_history = vis_data['p_history']
        v_history = vis_data['v_history']
        act_buffer = vis_data['act_history']
        
        if p_history.shape[1] <= 4:
            return
        
        # 位置历史图
        fig_p, ax = plt.subplots()
        p_cpu = p_history[:, 4].cpu().detach()
        ax.plot(p_cpu[:, 0], label='x')
        ax.plot(p_cpu[:, 1], label='y')
        ax.plot(p_cpu[:, 2], label='z')
        ax.legend()
        ax.set_title('Position History')
        
        # 速度历史图
        fig_v, ax = plt.subplots()
        v_cpu = v_history[:, 4].cpu().detach()
        ax.plot(v_cpu[:, 0], label='x')
        ax.plot(v_cpu[:, 1], label='y')
        ax.plot(v_cpu[:, 2], label='z')
        ax.legend()
        ax.set_title('Velocity History')
        
        # 动作历史图
        fig_a, ax = plt.subplots()
        act_cpu = act_buffer[:, 4].cpu().detach()
        ax.plot(act_cpu[:, 0], label='x')
        ax.plot(act_cpu[:, 1], label='y')
        ax.plot(act_cpu[:, 2], label='z')
        ax.legend()
        ax.set_title('Action History')
        
        self.writer.add_figure('p_history', fig_p, iteration + 1)
        self.writer.add_figure('v_history', fig_v, iteration + 1)
        self.writer.add_figure('a_history', fig_a, iteration + 1)
        
        plt.close('all')
    
    def _visualize_episode(self, iteration, vis_data):
        """可视化一个 episode 的数据"""
        print(f"\n📊 可视化 Episode {iteration + 1}")
        
        p_history = vis_data['p_history'][:, 0].cpu().numpy()  # 只取第一个样本
        vid = vis_data['vid']
        rgb_vid = vis_data['rgb_vid']
        obstacle_positions = vis_data['obstacle_positions']
        target_pos = vis_data['target_pos'][0].cpu().numpy()
        
        # 创建可视化目录
        vis_dir = os.path.join(self.args.log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. 3D 轨迹图
        fig = plt.figure(figsize=(15, 5))
        
        # 左子图：3D 轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(p_history[:, 0], p_history[:, 1], p_history[:, 2], 
                'b-', linewidth=2, label='Drone Trajectory')
        ax1.scatter(p_history[0, 0], p_history[0, 1], p_history[0, 2], 
                   color='green', s=100, marker='o', label='Start')
        ax1.scatter(p_history[-1, 0], p_history[-1, 1], p_history[-1, 2], 
                   color='red', s=100, marker='x', label='End')
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   color='gold', s=150, marker='*', label='Target')
        
        # 绘制障碍物最终位置
        if len(obstacle_positions) > 0 and len(obstacle_positions[-1]) > 0:
            final_obs = obstacle_positions[-1]
            for obs_pos in final_obs:
                ax1.scatter(obs_pos[0], obs_pos[1], obs_pos[2], 
                           color='orange', s=200, marker='s', alpha=0.7)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory (Iter {iteration + 1})')
        ax1.legend()
        
        # 中子图：深度图序列
        ax2 = fig.add_subplot(132)
        if len(vid) > 0:
            # 展示关键帧
            n_frames = min(4, len(vid))
            frame_indices = np.linspace(0, len(vid)-1, n_frames, dtype=int)
            
            depth_concat = np.concatenate([vid[i].numpy() for i in frame_indices], axis=1)
            ax2.imshow(depth_concat, cmap='viridis', aspect='auto')
            ax2.set_title('Depth Images (Start → End)')
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No depth data', ha='center', va='center')
            ax2.set_title('Depth Images')
        
        # 右子图：RGB 图序列
        ax3 = fig.add_subplot(133)
        if len(rgb_vid) > 0:
            n_frames = min(4, len(rgb_vid))
            frame_indices = np.linspace(0, len(rgb_vid)-1, n_frames, dtype=int)
            
            rgb_concat = np.concatenate([rgb_vid[i].numpy() for i in frame_indices], axis=1)
            # 归一化 RGB
            rgb_concat = (rgb_concat - rgb_concat.min()) / (rgb_concat.max() - rgb_concat.min() + 1e-8)
            ax3.imshow(rgb_concat, aspect='auto')
            ax3.set_title('RGB Images (Start → End)')
            ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, 'No RGB data', ha='center', va='center')
            ax3.set_title('RGB Images')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'episode_{iteration + 1:06d}.png'), dpi=150)
        plt.close()
        
        # 保存视频（如果启用）
        if self.args.save_video and len(vid) > 0:
            self._save_video(iteration, vis_data, vis_dir)
    
    def _save_video(self, iteration, vis_data, vis_dir):
        """保存 episode 的视频"""
        vid = vis_data['vid']
        rgb_vid = vis_data['rgb_vid']
        
        if len(vid) == 0:
            return
        
        # 创建动画
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        def update(frame):
            axes[0].clear()
            axes[1].clear()
            
            if frame < len(vid):
                axes[0].imshow(vid[frame].numpy(), cmap='viridis')
                axes[0].set_title(f'Depth Frame {frame}')
                axes[0].axis('off')
            
            if frame < len(rgb_vid):
                rgb = rgb_vid[frame].numpy()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                axes[1].imshow(rgb)
                axes[1].set_title(f'RGB Frame {frame}')
                axes[1].axis('off')
            
            return axes
        
        anim = animation.FuncAnimation(fig, update, frames=len(vid), interval=50)
        video_path = os.path.join(vis_dir, f'episode_{iteration + 1:06d}.mp4')
        anim.save(video_path, writer='ffmpeg', fps=20)
        plt.close()
        print(f"  📹 Video saved to {video_path}")
    
    def evaluate(self, num_episodes=10):
        """评估模型性能"""
        self.model.eval()
        
        total_success = 0
        total_speed = 0
        total_ar = 0
        
        with torch.no_grad():
            for ep in range(num_episodes):
                _, metrics, vis_data = self.run_episode(ep, record_vis=(ep == 0))
                
                total_success += metrics['success_rate'].item()
                total_speed += metrics['avg_speed'].item()
                total_ar += metrics['ar'].item()
                
                if ep == 0:
                    self.vis_data = vis_data
        
        self.model.train()
        
        print(f"\n📊 评估结果 ({num_episodes} episodes):")
        print(f"  成功率: {total_success / num_episodes:.2%}")
        print(f"  平均速度: {total_speed / num_episodes:.2f} m/s")
        print(f"  AR (Success × Speed): {total_ar / num_episodes:.3f}")
        
        return {
            'success_rate': total_success / num_episodes,
            'avg_speed': total_speed / num_episodes,
            'ar': total_ar / num_episodes
        }
    
    def visualize_last_episode(self):
        """可视化最后一次 episode 的结果"""
        if self.vis_data is None:
            print("没有可视化数据，请先运行 evaluate() 或训练")
            return
        
        self._visualize_episode(-1, self.vis_data)


def main():
    args = parse_args()
    print("=" * 60)
    print("无人机动态场景训练")
    print("=" * 60)
    print("\n训练参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    
    # 启用调试模式
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  Anomaly detection enabled!")
    
    trainer = DynamicDroneTrainer(args)
    
    # 训练前评估
    print("\n📝 训练前评估...")
    trainer.evaluate(num_episodes=5)
    
    # 开始训练
    print("\n🚀 开始训练...")
    trainer.train()
    
    # 训练后评估
    print("\n📝 训练后评估...")
    trainer.evaluate(num_episodes=10)
    
    # 可视化最后的 episode
    trainer.visualize_last_episode()


if __name__ == '__main__':
    main()
