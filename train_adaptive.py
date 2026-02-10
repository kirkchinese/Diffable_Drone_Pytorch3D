"""
无人机可微分仿真训练脚本

基于参考项目 DiffPhysDrone 的训练逻辑实现
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

from drone_env import DroneSimulator
from model import Model, Model_bigger,Model_adaptive
from loss import DroneLoss
from scene_generator import SceneGenerator
from training_monitor import TrainingMonitor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机避障训练脚本')
    
    # 训练参数
    parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_iters', type=int, default=50000, help='训练迭代次数')
    parser.add_argument('--timesteps', type=int, default=200, help='每次迭代的模拟步数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减系数')
    parser.add_argument('--ctl_dt', type=float, default=1/15, help='控制时间步长 (秒)')
    
    # 损失函数权重
    parser.add_argument('--coef_v', type=float, default=1.0, help='速度跟踪损失权重')
    parser.add_argument('--coef_speed', type=float, default=0.0, help='速度损失权重 (legacy)')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度预测损失权重')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞损失权重')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='障碍物回避损失权重')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='加速度正则化权重')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='加加速度正则化权重')
    parser.add_argument('--coef_d_snap', type=float, default=0.0, help='snap正则化权重 (legacy)')
    parser.add_argument('--coef_ground_affinity', type=float, default=0.1, help='高度惩罚损失权重 (防止飞高规避)')
    parser.add_argument('--coef_bias', type=float, default=0.0, help='方向偏差损失权重')
    parser.add_argument('--window_size', type=int, default=30, help='速度平均窗口大小')
    
    # 环境参数 - 渲染
    parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角')
    parser.add_argument('--image_height', type=int, default=240, help='图像高度')
    parser.add_argument('--image_width', type=int, default=320, help='图像宽度')
    parser.add_argument('--hfov', type=float, default=90.0,
                        help='相机水平视场角 (度)，默认90°。焦距由 FOV 和图像宽度自动计算')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj', help='障碍物网格路径')
    parser.add_argument('--num_samples', type=int, default=100000, help='障碍物点云采样数')
    parser.add_argument('--subdivide_times', type=int, default=0,
                        help='网格细分次数 (默认0, 配合z_clip无质量损失且渲染快10x+)')
    
    # 场景随机化参数
    parser.add_argument('--random_scene', action='store_true', default=False,
                        help='启用随机场景生成 (每 episode 随机组合障碍物)')
    parser.add_argument('--num_obstacles_min', type=int, default=20, help='每场景最少障碍物数')
    parser.add_argument('--num_obstacles_max', type=int, default=40, help='每场景最多障碍物数')
    parser.add_argument('--obstacle_scale_min', type=float, default=0.3, help='障碍物最小缩放')
    parser.add_argument('--obstacle_scale_max', type=float, default=1.5, help='障碍物最大缩放')
    parser.add_argument('--arena_range', type=float, default=6.0, help='场景水平范围 [-R,R]')
    parser.add_argument('--safe_spawn', action='store_true', default=False,
                        help='启用碰撞安全的出生点/目标点采样')
    parser.add_argument('--safe_clearance', type=float, default=1.0,
                        help='安全出生点到障碍物的最小距离')
    parser.add_argument('--force_cross_map', action='store_true', default=False,
                        help='强制出生/目标点在场景对向两侧（防止绕行）')
    parser.add_argument('--spawn_z_max', type=float, default=3.0,
                        help='出生/目标点最大高度（防止飞高规避）')
    
    # 环境参数 - 无人机物理
    parser.add_argument('--margin_min', type=float, default=0.1, help='无人机安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.7, help='无人机安全半径最大值')
    parser.add_argument('--init_p_range', type=float, default=8.0, help='初始位置范围')
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
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--save_freq', type=int, default=100, help='模型保存频率 (迭代次数)')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    
    # 硬件参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    
    return parser.parse_args()


def is_save_iter(i):
    """判断是否为保存迭代"""
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


class DroneTrainer:
    """无人机训练器"""
    
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu}')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # 控制时间步长 (参考项目使用 1/15 秒)
        self.ctl_dt = getattr(args, 'ctl_dt', 1/15)
        
        # 处理 airmode 开关
        enable_airmode = getattr(args, 'enable_airmode', True)
        if getattr(args, 'disable_airmode', False):
            enable_airmode = False
        
        # 创建场景生成器（如果启用随机场景）
        self.scene_generator = None
        if getattr(args, 'random_scene', False):
            self.scene_generator = SceneGenerator(
                device=self.device,
                arena_range=getattr(args, 'arena_range', 6.0),
                num_obstacles_range=(
                    getattr(args, 'num_obstacles_min', 20),
                    getattr(args, 'num_obstacles_max', 40),
                ),
                obstacle_scale_range=(
                    getattr(args, 'obstacle_scale_min', 0.3),
                    getattr(args, 'obstacle_scale_max', 1.5),
                ),
            )
            print(f"[SceneGenerator] 已启用随机场景生成, "
                  f"障碍物数量: {args.num_obstacles_min}-{args.num_obstacles_max}, "
                  f"归一化原语+网格抖动+3D旋转")
        
        self.safe_spawn = getattr(args, 'safe_spawn', False) or getattr(args, 'random_scene', False)
        if self.safe_spawn:
            print(f"[SafeSpawn] 已启用安全出生点/目标点, 最小安全距离: {getattr(args, 'safe_clearance', 1.0)}")
        
        # 根据 FOV 计算焦距
        import math
        focal_length = (args.image_width / 2.0) / math.tan(math.radians(args.hfov / 2.0))
        hfov_actual = 2 * math.degrees(math.atan(args.image_width / 2.0 / focal_length))
        vfov_actual = 2 * math.degrees(math.atan(args.image_height / 2.0 / focal_length))
        print(f"[Camera] HFOV={hfov_actual:.0f}° VFOV={vfov_actual:.0f}° "
              f"focal={focal_length:.1f} image={args.image_width}x{args.image_height}")

        # 初始化环境
        self.env = DroneSimulator(
            batch_size=args.batch_size,
            dt=self.ctl_dt,
            mesh_path=args.mesh_path,
            image_size=(args.image_height, args.image_width),
            focal_length=focal_length,
            device=self.device,
            # 动力学参数
            enable_airmode=enable_airmode,
            enable_induced_drag=False,
            noise_std=getattr(args, 'noise_std', 0.04),
            grad_decay=args.grad_decay,
            yaw_inertia=getattr(args, 'yaw_inertia', 5.0),
            yaw_ctl_delay=getattr(args, 'yaw_ctl_delay', 12.0),
            pitch_ctl_delay=getattr(args, 'pitch_ctl_delay', 12.0),
            airmode_coef=getattr(args, 'airmode_coef', 0.5),
            # 初始化参数
            init_p_range=getattr(args, 'init_p_range', 2.0),
            init_margin_range=(getattr(args, 'margin_min', 0.1), getattr(args, 'margin_max', 0.3)),
            # 点云采样
            num_samples=args.num_samples,
            # 渲染优化: subdivide_times=0 配合 z_clip_value 可获得等价质量, 面片从 106 万降至 1.6 万
            subdivide_times=args.subdivide_times,
            # 场景随机化
            enable_random_scene=getattr(args, 'random_scene', False),
            scene_generator=self.scene_generator,
            safe_spawn_clearance=getattr(args, 'safe_clearance', 1.0),
        )
        
        # 初始化模型
        if args.no_odom:
            self.model = Model_adaptive(dim_obs=7, dim_action=6).to(self.device) # 这里换了个大一点的模型，如果换回去就用 Model
        else:
            self.model = Model_adaptive(dim_obs=10, dim_action=6).to(self.device)  # 7 + 3 (local_v)
        
        # 加载预训练模型
        if args.resume:
            self._load_model(args.resume)
        
        # 优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.num_iters, eta_min=args.lr * 0.01)
        
        # 损失函数
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
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = os.path.join(args.log_dir, f'drone_train_{timestamp}')
        self.writer = SummaryWriter(log_path)
        print(f"TensorBoard logs: {log_path}")
        
        # 训练监控器（CSV 日志 + 损失曲线 PNG + 控制台摘要），输出到检查点目录
        self.monitor = TrainingMonitor(
            log_dir=args.save_dir,
            smoothing_window=50,
            csv_flush_interval=25,
            curve_save_interval=500,
            console_summary_interval=100,
        )
        
        # 确保保存目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 指标平滑队列
        self.scaler_q = defaultdict(list)
        
        # 重力标准向量
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)
        
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
        """
        计算用于速度/目标向量转换的局部坐标系旋转矩阵
        (参考项目的逻辑：使用水平投影后的前向轴)
        
        返回的 R_local 用于将世界坐标系向量转换到局部坐标系：
        v_local = v_world @ R_local  (等价于 R_local.T @ v_world)
        
        注意：这里返回的矩阵列向量是局部坐标系的基向量在世界坐标系中的表示
        """
        fwd = self.env.R[:, :, 0].clone()  # 机体 X 轴 (前向)
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0  # 水平投影
        up[:, 2] = 1   # 世界 Z 轴作为上方向
        # 添加数值稳定性：如果 fwd 太小，使用默认前向方向
        fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
        fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8), 
                          torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(fwd))
        # 构建旋转矩阵：列向量分别为 [fwd, left, up]
        R = torch.stack([fwd, torch.linalg.cross(up, fwd), up], -1)  # (B, 3, 3)
        return R
    
    def run_episode(self, iteration):
        """
        运行一个 episode 并返回损失
        """
        args = self.args
        B = args.batch_size
        
        # 重置环境和模型
        self.env.reset()
        self.model.reset()
        
        # 随机场景生成 (如果启用)
        if self.env.enable_random_scene and self.env.scene_generator is not None:
            self.env.randomize_scene()
        
        # 安全出生点采样 (如果启用)
        if self.safe_spawn:
            if getattr(self.args, 'force_cross_map', False):
                # 跨地图模式：出生/目标在对向两侧，防止绕行
                spawn_z_max = getattr(self.args, 'spawn_z_max', 3.0)
                _, p_target = self.env.safe_reset_cross_map(
                    arena_range=getattr(self.args, 'arena_range', 6.0),
                    z_range=(1.0, spawn_z_max),
                )
                cross_map_mode = True
            else:
                self.env.safe_reset(
                    arena_range=getattr(self.args, 'init_p_range', 8.0),
                    z_range=(1.0, getattr(self.args, 'spawn_z_max', 3.0)),
                )
                cross_map_mode = False
        else:
            cross_map_mode = False
        
        # 历史记录
        p_history = []
        v_history = []
        target_v_history = []
        vec_to_pt_history = []
        v_preds = []
        vid = []
        
        # GRU 隐藏状态
        h = None
        
        # 动作延迟缓冲 (模拟真实控制延迟)
        # 关键修复：参考项目使用 env.act 初始化，长度为 act_lag + 1
        # 然后每步使用 act_buffer[t] 访问，同时 append 新动作
        # 由于 t 从 0 开始，初始长度为 act_lag + 1，所以 act_buffer[t] 始终有效
        act_lag = 1
        initial_act = self.env.act_curr.clone()  # 使用环境当前的执行器状态初始化
        act_buffer = [initial_act.clone() for _ in range(act_lag + 1)]
        
        # 目标位置 (随机生成)
        spawn_z_max = getattr(self.args, 'spawn_z_max', 3.0)
        if cross_map_mode:
            # 跨地图模式下 p_target 已由 safe_reset_cross_map 返回
            pass
        elif self.safe_spawn:
            # 使用碰撞安全的目标点采样（低空）
            p_target = self.env.sample_safe_target(
                arena_range=getattr(self.args, 'arena_range', 6.0),
                z_range=(1.0, spawn_z_max),
                min_distance=3.0,
                max_distance=8.0,
            )
        else:
            # 原始方式：基于角度/距离偏移，不检查碰撞
            angle = torch.rand(B, device=self.device) * 2 * math.pi
            dist = torch.rand(B, device=self.device) * 5.0 + 3.0 # 距离 3m ~ 8m
            
            offset_x = torch.cos(angle) * dist
            offset_y = torch.sin(angle) * dist
            offset_z = torch.randn(B, device=self.device) * 2.0 # Z轴差异
            
            p_target = self.env.p.clone()
            p_target[:, 0] += offset_x
            p_target[:, 1] += offset_y
            p_target[:, 2] += offset_z
            
            p_target[:, 2] = p_target[:, 2].clamp(1.5, spawn_z_max)  # 限制 Z 范围
        
        target_v_raw = p_target - self.env.p
        
        # 个体随机化最大速度 (参考项目逻辑: 0.75 + 2.5 * rand) 我给他弄快了一点
        # 重要：这个值在整个 episode 中应保持不变
        max_speed = 0.75 + 5 * torch.rand((B, 1), device=self.device)
        
        # 推力估计误差 (模拟真实无人机的推力不确定性)
        thr_est_error = 1.0 + 0.01 * torch.randn((B, 1), device=self.device)
        
        # Per-sample 相机俯仰角随机化 (参考项目在 reset 中一次性设定整个 episode)
        # 模拟每架无人机相机安装角的个体差异，episode 内保持不变
        cam_pitch = args.cam_angle + torch.randn(B, device=self.device)
        
        # 航向漂移 (可选)
        if args.yaw_drift:
            drift_av = torch.randn(B, device=self.device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            R_drift = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ], -1).reshape(B, 3, 3)
        
        for t in range(args.timesteps):
            # 随机化控制间隔 (参考项目: mean=ctl_dt, std=0.1*ctl_dt)
            current_dt = normalvariate(self.ctl_dt, self.ctl_dt * 0.1)
            
            # 渲染深度图 - 使用 no_grad() 避免 PyTorch3D 透视投影反向传播的数值问题
            # 这里这个是个大坑，参考项目也是这么做的，我之前没有注意到，不这么做，会导致梯度爆炸，这个问题是在pytorch3d中产生的，问题很隐蔽，查了我很久。
            with torch.no_grad():
                _, depth = self.env.render(
                    camera_pitch=cam_pitch,
                    return_tensor=True,
                    return_rgb=False,
                    return_depth=True,
                    dt=current_dt
                )
            # 重新启用梯度（深度图作为模型输入）
            depth = depth.requires_grad_(False)  # 确保深度图不需要梯度
            
            # 记录 step 之前的状态 (与参考项目一致)
            p_history.append(self.env.p)
            vec_to_pt_history.append(self.env.vec_to_obj_subdivided(dt=current_dt))
            
            # 保存可视化帧 (第5个样本)
            if is_save_iter(iteration) and B > 4:
                vid.append(depth[4])
            
            # 更新目标向量 (可选航向漂移)
            if args.yaw_drift:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            else:
                target_v_raw = p_target - self.env.p.detach()  # detach 防止通过瞬移优化
            
            # 执行动作 (使用延迟缓冲中的动作)
            # 关键：使用 act_buffer[t] 而不是取模，参考项目的实现方式
            self.env.step(act_cmd=act_buffer[t], 
                         target_pos_vector=target_v_raw, 
                         dt=current_dt)
            
            # 计算局部坐标系
            R_local = self._compute_local_R()
            
            # 计算目标速度向量
            target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
            target_v_unit = target_v_raw / (target_v_norm + 1e-6)
            target_v = target_v_unit * torch.minimum(target_v_norm, max_speed)
            # 注意：target_v_history 在循环末尾与 v_history 一起记录
            
            # 构建观测状态
            # 转换到局部坐标系 - 关键修复！
            # 参考项目: torch.squeeze(target_v[:, None] @ R, 1)
            # 这里 v @ R 相当于 v.unsqueeze(1) @ R，结果是 (B, 1, 3) -> squeeze -> (B, 3)
            # 这是一种将世界坐标转换到以 R 的列向量为基的坐标系的方式
            target_v_local = torch.squeeze(target_v[:, None] @ R_local, 1)
            local_v = torch.squeeze(self.env.v[:, None] @ R_local, 1)
            
            state_parts = [
                target_v_local,           # 目标速度 (local) [3]
                self.env.R[:, 2],         # 机体姿态的第 3 行 [3] - 与参考项目保持一致
                                          # 注：参考项目用 R[:, 2] 而不是 R[:, :, 2]
                self.env.margin[:, None]  # 安全边距 [1]
            ]
            if not args.no_odom:
                state_parts.insert(0, local_v)  # 加入局部速度 [3]
            state = torch.cat(state_parts, dim=-1)  # [7] or [10]
            
            # 深度图预处理
            bg_mask = (depth < 0)  # PyTorch3D 背景像素 zbuf = -1
            x = depth.clamp(0.3, 24.0)
            x = 3.0 / x - 0.6  # 转换为近似线性空间
            x[bg_mask] = 0.0    # 背景设为 0（无障碍物），避免 -1 被误映射为最大近距信号
            x = x + torch.randn_like(x) * 0.02  # 添加噪声
            # x = F.max_pool2d(x[:, None], kernel_size=4, stride=4) # 缩小尺寸 # 这里我换了个模型，不用缩小了
            x = x.unsqueeze(1)  # 恢复通道维度 (B, H, W) -> (B, 1, H, W) # 与上一行对立，如果你要用小模型就注释这一行，恢复上一行

            # 模型推理
            act_raw, _, h = self.model(x, state, h)
            
            # 解析输出: [加速度向量(3), 速度预测(3)]
            # 参考项目: a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
            # act_raw: (B, 6) -> reshape to (B, 3, 2) 
            # R @ (B, 3, 2) -> (B, 3, 2) -> unbind(-1) -> [a_pred, v_pred]
            act_reshaped = act_raw.reshape(B, 3, 2)  # 关键修复！直接 reshape 为 (B, 3, 2)
            act_world = R_local @ act_reshaped  # 转换到世界坐标系 (B, 3, 2)
            a_pred, v_pred = act_world.unbind(-1)  # 分离加速度和速度预测
            
            v_preds.append(v_pred)
            
            # 计算实际动作 (参考项目公式)
            # act = (a_pred - v_pred - g_std) * thr_est_error + g_std
            act = (a_pred - v_pred - self.g_std) * thr_est_error + self.g_std
            
            # 加入动作缓冲
            act_buffer.append(act)
            
            # 关键修复：v_history 和 target_v_history 在 step 之后记录
            # 与参考项目一致
            v_history.append(self.env.v)
            target_v_history.append(target_v)
        
        # 堆叠历史
        p_history = torch.stack(p_history)          # (T, B, 3)
        v_history = torch.stack(v_history)          # (T, B, 3)
        target_v_history = torch.stack(target_v_history)  # (T, B, 3)
        vec_to_pt_history = torch.stack(vec_to_pt_history)  # (T, B, 3)
        v_preds = torch.stack(v_preds)              # (T, B, 3)
        act_buffer_stacked = torch.stack(act_buffer)  # (T + lag + 1, B, 3)
        
        # 计算损失
        loss, metrics = self.losser.forward(
            p_history=p_history,
            v_history=v_history,
            target_vel_history=target_v_history,
            act_history=act_buffer_stacked,
            vec_to_obj_history=vec_to_pt_history,
            v_preds=v_preds,
            env_margin=self.env.margin,
            env_g_std=self.g_std
        )
        
        # 计算额外指标
        with torch.no_grad():
            distance = torch.norm(vec_to_pt_history, 2, -1) - self.env.margin
            speed_history = v_history.norm(2, -1)
            avg_speed = speed_history.mean(0)
            success = torch.all(distance.flatten(0, 1) > 0, 0)
            success_rate = success.float().mean()
            
            metrics['success_rate'] = success_rate
            metrics['avg_speed'] = avg_speed.mean()
            metrics['max_speed'] = speed_history.max(0).values.mean()
            metrics['ar'] = (success.float() * avg_speed).mean()  # 成功率 × 平均速度
        
        return loss, metrics, (p_history, v_history, act_buffer_stacked, vid)
    
    def train(self):
        """主训练循环"""
        args = self.args
        
        pbar = tqdm(range(args.num_iters), ncols=160, bar_format='{l_bar}{bar:20}{r_bar}')
        
        for i in pbar:
            # 运行一个 episode
            loss, metrics, debug_data = self.run_episode(i)
            
            # 检查 NaN
            if torch.isnan(loss):
                print("Loss is NaN, exiting...")
                break
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录当前学习率到 metrics
            metrics['lr'] = self.scheduler.get_last_lr()[0]
            
            # 训练监控：CSV + 曲线 + 控制台摘要 + tqdm
            self.monitor.step(i, loss, metrics, pbar=pbar)
            
            # TensorBoard 记录
            self._smooth_dict({
                'loss': loss,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))}
            })
            
            # 定期日志和保存
            if is_save_iter(i):
                self._log_figures(i, debug_data)
            
            if (i + 1) % args.save_freq == 0:
                save_path = os.path.join(args.save_dir, f'checkpoint_{i+1:06d}.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"\nSaved model to {save_path}")
            
            if (i + 1) % 25 == 0:
                for k, v in self.scaler_q.items():
                    self.writer.add_scalar(k, sum(v) / len(v), i + 1)
                self.scaler_q.clear()
        
        # 保存最终模型
        final_path = os.path.join(args.save_dir, 'checkpoint_final.pth')
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")
        
        self.monitor.close()
        self.writer.close()
    
    def _log_figures(self, iteration, debug_data):
        """记录可视化图表"""
        p_history, v_history, act_buffer, vid = debug_data
        
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


def main():
    args = parse_args()
    print("Training arguments:")
    print(args)
    
    # 启用调试模式
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled!")
    
    trainer = DroneTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
