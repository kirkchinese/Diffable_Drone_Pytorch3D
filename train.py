"""
无人机可微分仿真训练脚本

基于参考项目 DiffPhysDrone 的训练逻辑实现
"""

import os
import gc
import random
import argparse
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffsim import EnvBuildContext, make_env
from diffsim.losses import LossBuildContext, make_loss
from model import (
    Model, Model_bigger, Model_adaptive,
    Model_attention, Model_multiscale, Model_residual, Model_lightweight,
    Model_lidar, Model_fusion,
    DecayController, LossGuide, MetaController, LossNetwork,
)
from navigation_utils import (
    compute_navigation_metrics_torch,
    DronePolicy,
)
from scene_generator import SceneGenerator
from training_monitor import TrainingMonitor
from drone_renderer import hfov_to_focal, focal_to_hfov, focal_to_vfov
from lidar_sensor import LiDARSensor


class _ArgParser(argparse.ArgumentParser):
    """支持 @file 语法从文件读取参数，一行一条，支持整行和行内 # 注释"""
    def convert_arg_line_to_args(self, line):
        line = line.split('#')[0].strip()   # 去除行内注释
        if not line:
            return []
        return line.split()


def parse_args():
    """解析命令行参数"""
    parser = _ArgParser(description='无人机避障训练脚本', fromfile_prefix_chars='@')

    parser.add_argument(
        '--env_type', '--env-type', type=str, default='drone',
        choices=['drone', 'go2'],
        help='模拟环境类型。drone 保持原数值路径；go2 使用独立可微四足 trainer。',
    )
    parser.add_argument('--go2_batch_size', type=int, default=None,
                        help='Go2 批大小；未设置时默认 64，不改变无人机默认值')
    parser.add_argument('--go2_control_dt', type=float, default=0.02,
                        help='Go2 控制周期，默认 20 ms')
    parser.add_argument('--physics_dt', type=float, default=0.001,
                        help='Go2 物理子步，默认 1 ms')
    parser.add_argument('--go2_sensor_mode', choices=['blind', 'heightmap', 'depth'],
                        default='heightmap', help='Go2 感知对照模式')
    parser.add_argument('--go2_randomize', action=argparse.BooleanOptionalAction, default=True,
                        help='Go2 质量/惯量/执行器/传感器/场景域随机化')
    parser.add_argument('--go2_max_obstacles', type=int, default=24)
    parser.add_argument('--go2_rollout_steps', type=int, default=100)
    parser.add_argument('--go2_tbptt_steps', type=int, default=25)
    parser.add_argument('--go2_hidden_size', type=int, default=128)
    parser.add_argument('--go2_symmetry_weight', type=float, default=0.05)
    parser.add_argument('--go2_gdecay', type=float, default=1.0,
                        help='Go2 跨控制步状态梯度衰减；1.0 保持原始梯度')
    parser.add_argument('--go2_grad_clip_norm', type=float, default=1.0)
    
    # 训练参数
    parser.add_argument('--resume', default=None, help='恢复训练的模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_iters', type=int, default=50000, help='训练迭代次数')
    parser.add_argument('--timesteps', type=int, default=150, help='每次迭代的模拟步数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减系数')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0,
                        help='梯度裁剪最大范数。>0 时启用 clip_grad_norm_ 防止梯度爆炸导致策略退化。推荐值 1.0')
    parser.add_argument('--ctl_dt', type=float, default=1/15, help='控制时间步长 (秒)')
    parser.add_argument('--render_interval', type=int, default=1, help='渲染间隔帧数 (1=每帧渲染, 2=隔帧渲染, 节省渲染开销)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子：设置后会 seed random/numpy/torch/cuda 以复现实验场景和初始化；'
                             'PyTorch3D rasterizer/GPU 路径不保证完整 bitwise deterministic')
    
    # 损失函数权重
    parser.add_argument('--coef_v', type=float, default=1.0, help='速度跟踪损失权重')
    parser.add_argument('--coef_speed', type=float, default=0.0, help='速度损失权重 (legacy)')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度预测损失权重')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞损失权重')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='障碍物回避损失权重')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='加速度正则化权重')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='加加速度正则化权重')
    parser.add_argument('--coef_d_snap', type=float, default=0.0, help='snap正则化权重 (legacy)')
    parser.add_argument('--coef_ground_affinity', type=float, default=0.0, help='高度天花板损失权重 (仅惩罚超过 ga_z_ceiling 的高度)')
    parser.add_argument('--ga_z_ceiling', type=float, default=5.0,
                        help='高度天花板 (m), 仅在 z > 此值时产生惩罚梯度 (默认 5.0)')
    parser.add_argument('--coef_bias', type=float, default=0.0, help='方向偏差损失权重')
    parser.add_argument('--coef_lateral', type=float, default=0.0, help='横向运动惩罚权重 (速度分解loss中的横向分量，0=横向免罚)')
    parser.add_argument('--coef_drone_collide', type=float, default=0.0,
                        help='[已废弃] 无人机间碰撞损失权重。默认 0.0：无人机碰撞已由 combined_vec_to_nearest '
                             '统一纳入 collide + obj_avoidance，与参考项目一致。'
                             '设置 >0 会导致双重惩罚')
    parser.add_argument('--window_size', type=int, default=30, help='速度平均窗口大小')
    parser.add_argument('--loss_v_mode', type=str, default='mse',
                        choices=['mse', 'decomposed', 'adaptive'],
                        help='速度损失模式: '
                             'mse=参考项目式(惩罚全方向,默认), '
                             'decomposed=分解式(横向免罚), '
                             'adaptive=自适应(终点指数衰减制动)')
    parser.add_argument('--loss_curriculum', action='store_true', default=False,
                        help='启用损失课程学习: 前期用 mse 损失(稳定学习基础飞行), '
                             '后期切换到 adaptive 损失(优化制动和达标精度)')
    parser.add_argument('--curriculum_transition', type=float, default=0.4,
                        help='损失课程转换点 (占总训练比例, 默认 0.4 = 40%% 时切换)')
    parser.add_argument('--coef_goal_reaching', type=float, default=0.0,
                        help='目标达成辅助损失系数。>0 时添加终点距离目标的直接梯度信号，'
                        '补充密集速度跟踪损失。推荐值 0.1-0.5')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='策略权重 EMA 衰减率。>0 时维护影子参数 θ_ema=α*θ_ema+(1-α)*θ, '
                        '缓解训练后期性能退化。推荐值 0.999 或 0.9995')
    parser.add_argument('--adaptive_decay_rate', type=float, default=2.0,
                        help='adaptive模式的指数衰减率λ, alpha=exp(-λ*V_target), 默认2.0')
    
    # 环境参数 - 渲染
    parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角')
    parser.add_argument('--cam_mount_roll', type=float, default=0.0,
                        help='相机安装横滚角 (度，正值=左倾)')
    parser.add_argument('--cam_mount_yaw', type=float, default=0.0,
                        help='相机安装偏航角 (度，正值=左转)')
    parser.add_argument('--cam_mode', type=str, default='auto', choices=['auto', 'manual'],
                        help='相机安装模式: auto=网格比例自动计算+随机化, manual=用户3×4外参矩阵')
    parser.add_argument('--cam_extrinsic', type=float, nargs=12, default=None,
                        help='手动模式相机外参 [R|t] 3×4行优先 (12个浮点数)'
                             ' 例: 1 0 0 0.5  0 1 0 0  0 0 1 -0.03')
    parser.add_argument('--cam_rand_xy', type=float, default=0.02,
                        help='相机 XY 偏移随机化半径 (m)，模拟安装误差 (仅auto模式)')
    parser.add_argument('--cam_rand_z_range', type=float, nargs=2, default=[-0.04, 0.04],
                        help='相机 Z 偏移随机范围 [min, max] (m) (仅auto模式)')
    parser.add_argument('--cam_rand_rpy', type=float, default=2.0,
                        help='相机 roll/yaw 随机化半径 (度) (仅auto模式)')
    parser.add_argument('--image_height', type=int, default=48, help='图像高度')
    parser.add_argument('--image_width', type=int, default=64, help='图像宽度')
    parser.add_argument('--hfov', type=float, default=79.0,
                        help='相机水平视场角 (度)。默认 79° 对齐参考项目多机配置 '
                             '(fov_x_half_tan=0.82 → 78.7°)。焦距由 FOV 和图像宽度自动计算')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj', help='障碍物网格路径')
    parser.add_argument('--num_samples', type=int, default=100000, help='障碍物点云采样数')
    parser.add_argument('--subdivide_times', type=int, default=0,
                        help='网格细分次数 (默认0, 配合z_clip无质量损失且渲染快)')
    parser.add_argument('--depth_min', type=float, default=0.3,
                        help='深度图近截断距离 (米)，近于此的深度被截断。同时也是渲染器近平面裁剪值')
    parser.add_argument('--depth_max', type=float, default=24.0,
                        help='深度图远截断距离 (米)，远于此的深度被截断。'
                             '对齐参考项目 24.0')
    # parser.add_argument('--depth_dilate', type=int, default=0,
    #                     help='深度图障碍物膨胀核大小 (奇数, 0=禁用)。'
    #                          'stride=1 max_pool2d 使障碍物在深度图中向外扩展，'
    #                          '类似机器人导航中的 C-space 膨胀。'
    #                          '默认禁用，仅在擦碰问题严重时尝试开启 (5 或 7)')
    
    # 场景随机化参数
    parser.add_argument('--random_scene', action='store_true', default=False,
                        help='启用随机场景生成 (每 episode 随机组合障碍物)')
    parser.add_argument('--num_obstacles_min', type=int, default=40, help='每场景最少障碍物数')
    parser.add_argument('--num_obstacles_max', type=int, default=80, help='每场景最多障碍物数')
    parser.add_argument('--obstacle_scale_min', type=float, default=0.3, help='障碍物最小缩放')
    parser.add_argument('--obstacle_scale_max', type=float, default=1.5, help='障碍物最大缩放')
    parser.add_argument('--arena_range', type=float, default=6.0, help='场景水平范围 [-R,R]')
    parser.add_argument('--ground_ratio', type=float, default=0.6,
                        help='接地物体比例 (0~1)，接地物体底部紧贴地面形成柱子/墙壁')
    parser.add_argument('--cluster_ratio', type=float, default=0.3,
                        help='簇生物体比例 (0~1)，放置在已有物体附近形成复合形状')
    parser.add_argument('--cluster_spread', type=float, default=1.5,
                        help='簇生物体相对父物体的最大水平偏移 (m)')
    parser.add_argument('--safe_spawn', action='store_true', default=False,
                        help='启用碰撞安全的出生点/目标点采样')
    parser.add_argument('--safe_clearance', type=float, default=1.0,
                        help='安全出生点到障碍物的最小距离')
    parser.add_argument('--min_spawn_inter_distance', type=float, default=0.0,
                        help='无人机之间的最小出生间距 (米)。0=自动计算,基于 2*margin_max+0.5')
    parser.add_argument('--force_cross_map', action='store_true', default=False,
                        help='强制出生/目标点在场景对向两侧（防止绕行）')
    parser.add_argument('--spawn_z_max', type=float, default=3.0,
                        help='出生/目标点最大高度')
    
    # 环境参数 - 无人机物理
    parser.add_argument('--margin_min', type=float, default=0.3, help='无人机安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.8, help='无人机安全半径最大值')
    parser.add_argument('--init_p_range', type=float, default=8.0, help='初始位置范围')
    parser.add_argument('--noise_std', type=float, default=0.04, help='环境扰动噪声标准差')
    parser.add_argument('--yaw_inertia', type=float, default=5.0, help='偏航惯性')
    parser.add_argument('--yaw_ctl_delay', type=float, default=12.0, help='偏航控制延迟')
    parser.add_argument('--pitch_ctl_delay', type=float, default=12.0, help='俯仰控制延迟')
    parser.add_argument('--airmode_coef', type=float, default=0.5, help='Airmode 系数')
    parser.add_argument('--enable_airmode', action='store_true', default=True, help='启用 Airmode')
    parser.add_argument('--disable_airmode', action='store_true', default=False, help='禁用 Airmode')
    
    # 无人机网格与多机交互
    parser.add_argument('--drone_mesh_path', type=str, default="./data/base_model/drone.obj",
                        help='无人机网格路径 (如 ./data/base_model/drone.obj)，'
                             '用于计算安全半径和未来的无人机间渲染')
    parser.add_argument('--n_drones_per_group', type=int, default=None,
                        help='多机分组大小 (默认=batch_size, 即组内全部交互; 1=禁用碰撞检测)')
    parser.add_argument('--aero_margin', type=float, default=0.05,
                        help='无人机包围球之外的气动安全余量 (m)')
    parser.add_argument('--max_drone_faces', type=int, default=500,
                        help='无人机网格最大面片数 (超过自动简化, 0=不简化)')

    # 动态障碍物
    parser.add_argument('--enable_dynamic_obstacles', action='store_true', default=False,
                        help='启用动态障碍物（每个 episode 随机生成移动的球体/立方体）')
    parser.add_argument('--num_dynamic_obstacles_min', type=int, default=2,
                        help='动态障碍物最小数量')
    parser.add_argument('--num_dynamic_obstacles_max', type=int, default=10,
                        help='动态障碍物最大数量')
    parser.add_argument('--dynamic_obs_speed_min', type=float, default=-0.5,
                        help='动态障碍物最小速度')
    parser.add_argument('--dynamic_obs_speed_max', type=float, default=0.5,
                        help='动态障碍物最大速度')
    parser.add_argument('--dynamic_obs_scale_min', type=float, default=0.2,
                        help='动态障碍物最小缩放')
    parser.add_argument('--dynamic_obs_scale_max', type=float, default=0.8,
                        help='动态障碍物最大缩放')
    
    # 模型参数
    parser.add_argument('--no_odom', default=False, action='store_true', help='不使用里程计速度作为输入')
    parser.add_argument('--model_type', type=str, default='bigger',
                        choices=['base', 'bigger', 'adaptive', 'attention', 'multiscale',
                                 'residual', 'lightweight', 'lidar', 'fusion'],
                        help='模型类型: base=Model, bigger=Model_bigger, adaptive=Model_adaptive, '
                             'attention=注意力, multiscale=多尺度, residual=残差+LSTM, '
                             'lightweight=轻量级, lidar=LiDAR距离图, fusion=深度+LiDAR融合')
    parser.add_argument('--sensor_mode', type=str, default='depth',
                        choices=['depth', 'lidar', 'fusion'],
                        help='传感器模式: depth=仅深度图, lidar=仅LiDAR距离图, fusion=双分支融合')
    parser.add_argument('--lidar_beams', type=int, default=16,
                        help='LiDAR 垂直光束数 (仿 VLP-16)')
    parser.add_argument('--lidar_points', type=int, default=64,
                        help='LiDAR 每条光束水平采样点数')
    parser.add_argument('--yaw_drift', default=False, action='store_true', help='启用航向漂移')
    parser.add_argument('--debug', default=False, action='store_true', help='启用 anomaly detection 调试模式')
    
    # CMA-ES 参数
    parser.add_argument('--use_cmaes', action='store_true', default=False,
                        help='启用 CMA-ES 进化优化（与梯度训练双重循环）')
    parser.add_argument('--cma_mode', type=str, default='guide',
                        choices=['decay', 'guide', 'meta', 'loss_net'],
                        help='CMA-ES 模式: decay=进化梯度衰减控制器, guide=进化损失系数, '
                             'meta=A-网络元控制器, loss_net=A-网络损失函数')
    parser.add_argument('--cma_pop_size', type=int, default=20, help='CMA-ES 种群大小')
    parser.add_argument('--cma_sigma0', type=float, default=0.5, help='CMA-ES 初始步长')
    parser.add_argument('--cma_eval_interval', type=int, default=50,
                        help='CMA-ES 评估间隔 (每 N 个训练步进行一次进化)')
    parser.add_argument('--cma_warmup', type=int, default=0,
                        help='CMA-ES 预热期: 前 N 步仅用标准损失训练，之后再激活进化。'
                             '解决冷启动问题: 策略先学会基本飞行，CMA-ES 才有有效信号。')
    parser.add_argument('--decay_min', type=float, default=0.2, help='DecayController 最小衰减因子')
    parser.add_argument('--decay_max', type=float, default=1.0, help='DecayController 最大衰减因子')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--save_freq', type=int, default=100, help='模型保存频率 (迭代次数)')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    
    # 硬件参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--reach_radius', type=float, default=0.5,
                        help='判定到达目标点的半径阈值 (米)')
    parser.add_argument('--random_init_yaw', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在 reset 时随机化无人机初始偏航角')
    parser.add_argument('--reset_lr', action='store_true', default=False,
                        help='Resume 时重置学习率调度器 (从 lr 开始新的 cosine 周期)。'
                             '当 resume 后改变训练参数 (场景/损失权重等) 时应启用此选项，'
                             '否则 lr 会从上次衰减终点 (~lr*0.01) 缓慢回升，前1000步几乎无学习能力')
    
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
        
        # GPU 性能优化
        torch.backends.cudnn.benchmark = True   # 自动选择最快的 cuDNN 卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True   # 允许 TF32 加速矩阵乘法
        torch.backends.cudnn.allow_tf32 = True          # 允许 cuDNN TF32
        
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
                ground_ratio=getattr(args, 'ground_ratio', 0.6),
                cluster_ratio=getattr(args, 'cluster_ratio', 0.3),
                cluster_spread=getattr(args, 'cluster_spread', 1.5),
                seed=getattr(args, 'seed', None),
            )
            print(f"[SceneGenerator] 已启用随机场景生成, "
                  f"障碍物数量: {args.num_obstacles_min}-{args.num_obstacles_max}, "
                  f"接地率: {args.ground_ratio:.0%}, "
                  f"簇生率: {args.cluster_ratio:.0%}, "
                  f"归一化+网格抖动+3D旋转")
        
        self.safe_spawn = getattr(args, 'safe_spawn', False) or getattr(args, 'random_scene', False)
        if self.safe_spawn:
            print(f"[SafeSpawn] 已启用安全出生点/目标点, 最小安全距离: {getattr(args, 'safe_clearance', 1.0)}")
        
        # 根据 FOV 计算焦距: 使用公共工具函数
        focal_length = hfov_to_focal(args.hfov, args.image_width)
        hfov_actual = focal_to_hfov(focal_length, args.image_width)
        vfov_actual = focal_to_vfov(focal_length, args.image_height)
        print(f"[Camera] HFOV={hfov_actual:.0f}° VFOV={vfov_actual:.0f}° "
              f"focal={focal_length:.1f} image={args.image_width}x{args.image_height}")

        # 环境由注册表构造。默认 drone 仍调用原 DroneSimulator，参数与数值路径不变。
        env_type = getattr(args, 'env_type', 'drone')
        self.env = make_env(
            env_type,
            EnvBuildContext(
                args=args,
                device=self.device,
                control_dt=self.ctl_dt,
                focal_length=focal_length,
                scene_generator=self.scene_generator,
                extras={'enable_airmode': enable_airmode},
            ),
        )
        print(f"[Environment] type={env_type} class={type(self.env).__name__}")
        
        # 初始化模型
        dim_obs = 7 if args.no_odom else 10
        model_map = {
            'base': Model,
            'bigger': Model_bigger,
            'adaptive': Model_adaptive,
            'attention': Model_attention,
            'multiscale': Model_multiscale,
            'residual': Model_residual,
            'lightweight': Model_lightweight,
            'lidar': Model_lidar,
            'fusion': Model_fusion,
        }
        model_type = getattr(args, 'model_type', 'bigger')
        ModelClass = model_map[model_type]
        self.model = ModelClass(dim_obs=dim_obs, dim_action=6).to(self.device)
        print(f"[Model] 使用 {ModelClass.__name__} (model_type={model_type}, dim_obs={dim_obs})")
        
        # 加载预训练模型
        if args.resume:
            self._load_model(args.resume)
        
        # 重力标准向量
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)

        # LiDAR 传感器（sensor_mode 含 LiDAR 时创建）
        sensor_mode = getattr(args, 'sensor_mode', 'depth')
        self.lidar_sensor = None
        if sensor_mode in ('lidar', 'fusion'):
            self.lidar_sensor = LiDARSensor(
                num_beams=getattr(args, 'lidar_beams', 16),
                points_per_beam=getattr(args, 'lidar_points', 64),
                max_range=args.depth_max,
                min_range=args.depth_min,
                device=self.device,
            ).setup(focal_length, args.image_height, args.image_width)
            print(f"[LiDAR] {self.lidar_sensor.num_beams}×{self.lidar_sensor.points_per_beam} "
                  f"range [{self.lidar_sensor.min_range}, {self.lidar_sensor.max_range}]m")

        # 策略适配器（封装 obs 构造 / 模型前向 / 动作后处理）
        self.policy = DronePolicy(
            model=self.model,
            g_std=self.g_std,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            no_odom=args.no_odom,
            sensor_mode=sensor_mode,
            lidar_sensor=self.lidar_sensor,
        )
        
        # 优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.num_iters, eta_min=args.lr * 0.01)

        # EMA 影子参数: 指数移动平均缓解训练后期性能退化
        self.ema_decay = getattr(args, 'ema_decay', 0.0)
        if self.ema_decay > 0:
            self.ema_shadow = {k: v.clone() for k, v in self.model.state_dict().items()}
            print(f"[EMA] decay={self.ema_decay}, shadow params initialized")
        else:
            self.ema_shadow = None
        
        # 恢复优化器 / 调度器 / best-metric 状态（必须在 optimizer/scheduler 创建之后）
        # start_iter: 无 --reset_lr 的 resume 会从 checkpoint 的迭代数继续，
        # 保证总训练步数恰为 num_iters（而非重跑一整轮）
        self.start_iter = 0
        if args.resume:
            self._restore_training_state()
        
        # 损失函数
        self.losser = make_loss(
            'drone_navigation',
            LossBuildContext(args=args, control_dt=self.ctl_dt),
        )
        # 保存初始系数（LossGuide 模式下用于合并未被进化覆盖的系数）
        self._base_coefs = dict(self.losser.coefs)
        
        # CMA-ES 初始化
        self.use_cmaes = getattr(args, 'use_cmaes', False)
        self.cma_mode = getattr(args, 'cma_mode', 'guide')
        self.decay_controller = None
        self.loss_guide = None
        self.meta_controller = None
        self.loss_network = None
        self._cma_target = None  # 当前 CMA-ES 目标对象引用 (消除多态 if/elif 分支)
        self.cma_es = None
        
        if self.use_cmaes:
            import cma as _cma
            self._cma = _cma
            
            if self.cma_mode == 'decay':
                # 从模型推断 CNN 特征维度
                if hasattr(self.model, 'gru'):
                    feat_dim = self.model.gru.input_size
                elif hasattr(self.model, 'lstm'):
                    feat_dim = self.model.lstm.input_size
                else:
                    feat_dim = 256
                self.decay_controller = DecayController(
                    feat_dim=feat_dim,
                    decay_min=getattr(args, 'decay_min', 0.2),
                    decay_range=getattr(args, 'decay_max', 1.0) - getattr(args, 'decay_min', 0.2),
                ).to(self.device)
                x0 = self.decay_controller.get_params_vector().cpu().numpy()
                n_params = self.decay_controller.num_params
                print(f"[CMA-ES] DecayController: {n_params} params, "
                      f"decay ∈ [{args.decay_min}, {args.decay_max}]")
                if n_params > 50:
                    print(f"[CMA-ES] ⚠ 警告: DecayController 有 {n_params} 参数，"
                          f"CMA-ES 在 >50 维空间效率极低，建议使用 guide 模式或简化参数化")
            elif self.cma_mode == 'guide':
                self.loss_guide = LossGuide().to(self.device)
                x0 = self.loss_guide.get_params_vector().cpu().numpy()
                print(f"[CMA-ES] LossGuide: {self.loss_guide.num_params} params (损失系数进化)")
            elif self.cma_mode == 'meta':
                self.meta_controller = MetaController(hidden=4, ema_alpha=0.1).to(self.device)
                x0 = self.meta_controller.get_params_vector().cpu().numpy()
                print(f"[CMA-ES] MetaController (A-网络): {self.meta_controller.num_params} params, "
                      f"输入={MetaController.N_INPUT}D 训练状态, 输出={MetaController.N_OUTPUT}D 调制信号")
            else:  # loss_net
                self.loss_network = LossNetwork(
                    hidden=16, default_coefs=dict(self.losser.coefs),
                ).to(self.device)
                x0 = self.loss_network.get_params_vector().cpu().numpy()
                print(f"[CMA-ES] LossNetwork (A-网络/损失函数): {self.loss_network.num_params} params, "
                      f"输入={LossNetwork.N_LOSS}+1D, 激活=LeakyReLU, 温启动=线性组合")
            
            # 统一引用: 所有 CMA-ES 模式共享 get/set_params_vector 接口
            self._cma_target = (self.decay_controller or self.loss_guide
                                or self.meta_controller or self.loss_network)
            
            cma_options = {
                'popsize': getattr(args, 'cma_pop_size', 20),
                'seed': 42,
                'maxiter': int(1e6),
                'verbose': -1,
            }
            self.cma_es = _cma.CMAEvolutionStrategy(x0.tolist(), args.cma_sigma0, cma_options)
            print(f"[CMA-ES] pop_size={args.cma_pop_size}, sigma0={args.cma_sigma0}, "
                  f"eval_interval={args.cma_eval_interval}")

            # 从 checkpoint 恢复 CMA-ES 控制器参数 (控制器必须已创建)
            resume_ckpt = getattr(self, '_resume_ckpt', None)
            if resume_ckpt is not None:
                if self.decay_controller is not None and 'decay_controller' in resume_ckpt:
                    self.decay_controller.load_state_dict(resume_ckpt['decay_controller'])
                    print(f"[Resume] DecayController 参数已恢复")
                if self.loss_guide is not None and 'loss_guide' in resume_ckpt:
                    self.loss_guide.load_state_dict(resume_ckpt['loss_guide'])
                    print(f"[Resume] LossGuide 参数已恢复")
                if self.meta_controller is not None and 'meta_controller' in resume_ckpt:
                    self.meta_controller.load_state_dict(resume_ckpt['meta_controller'])
                    if 'meta_controller_ema' in resume_ckpt:
                        self.meta_controller._ema = resume_ckpt['meta_controller_ema']
                    print(f"[Resume] MetaController 参数已恢复 (含 EMA)")
                if self.loss_network is not None and 'loss_network' in resume_ckpt:
                    self.loss_network.load_state_dict(resume_ckpt['loss_network'])
                    print(f"[Resume] LossNetwork 参数已恢复")
                # 恢复 CMA-ES 完整内部状态 (sigma, 协方差, 进化路径)
                if 'cma_es_state' in resume_ckpt:
                    import pickle
                    self.cma_es = pickle.loads(resume_ckpt['cma_es_state'])
                    print(f"[Resume] CMA-ES 内部状态已恢复 (sigma={self.cma_es.sigma:.4f})")
                else:
                    # 旧 checkpoint 无 CMA-ES 状态, 用恢复的参数重建
                    x0_restored = self._cma_target.get_params_vector().cpu().numpy()
                    self.cma_es = _cma.CMAEvolutionStrategy(x0_restored.tolist(), args.cma_sigma0, cma_options)
                    print(f"[Resume] CMA-ES 起点已更新为恢复的参数 (旧格式, 无内部状态)")
        
        # 释放 checkpoint 缓存 (CMA-ES 控制器恢复已在上方完成)
        if hasattr(self, '_resume_ckpt'):
            del self._resume_ckpt

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
            resume_step=self.start_iter,  # 续跑时保留 metrics.csv 历史并追加
        )
        
        # 确保保存目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 指标平滑队列
        self.scaler_q = defaultdict(list)

        # Best AR 追踪（用于保存最佳策略 checkpoint）
        # AR = success_rate × avg_speed，综合衡量"安全且高效"的行为质量
        self.best_ar = -1.0
        self.best_ar_iter = -1
        self.ar_ema = 0.0           # 指数移动平均，减少单次波动
        self.ar_ema_alpha = 0.05    # EMA 平滑系数
        self.best_task_score = -1.0
        self.best_task_iter = -1
        self.task_score_ema = 0.0
        
    @staticmethod
    def _get_git_hash():
        """获取当前 git commit hash，失败时返回 'unknown'。"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except Exception:
            return 'unknown'

    def _make_checkpoint(self, iteration, extra=None):
        """构建完整 checkpoint dict（含模型权重、优化器、调度器、超参数、迭代计数、版本号、git hash）。"""
        ckpt = {
            'version': 2,
            'git_hash': self._get_git_hash(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': iteration,
            'args': vars(self.args),
            # RNG 状态: 使中断-续跑与不间断训练走同一随机轨迹
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            },
        }
        # CMA-ES 状态
        if self.decay_controller is not None:
            ckpt['decay_controller'] = self.decay_controller.state_dict()
        if self.loss_guide is not None:
            ckpt['loss_guide'] = self.loss_guide.state_dict()
        if self.meta_controller is not None:
            ckpt['meta_controller'] = self.meta_controller.state_dict()
            ckpt['meta_controller_ema'] = dict(self.meta_controller._ema)
        if self.loss_network is not None:
            ckpt['loss_network'] = self.loss_network.state_dict()
        # CMA-ES 优化器内部状态 (sigma, 协方差矩阵, 进化路径等)
        if self.cma_es is not None:
            import pickle
            ckpt['cma_es_state'] = pickle.dumps(self.cma_es)
        # EMA 影子参数
        if self.ema_shadow is not None:
            ckpt['ema_state_dict'] = {k: v.cpu() for k, v in self.ema_shadow.items()}
        if extra:
            ckpt.update(extra)
        return ckpt

    def _load_model(self, path):
        """加载模型权重（兼容旧纯 state_dict 与新 checkpoint dict），并缓存 ckpt 供后续恢复优化器状态。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            saved_iter = ckpt.get('iteration', '?')
            ver = ckpt.get('version', 1)
            git_hash = ckpt.get('git_hash', '无')
            print(f"[Checkpoint] v{ver}, 迭代={saved_iter}, git={git_hash}")
            # 自动 diff 超参数
            saved_args = ckpt.get('args', {})
            if saved_args:
                current_args = vars(self.args)
                diffs = []
                for k in sorted(set(saved_args) | set(current_args)):
                    old_v, new_v = saved_args.get(k), current_args.get(k)
                    if old_v != new_v:
                        diffs.append(f"  {k}: {old_v} → {new_v}")
                if diffs:
                    print(f"[Checkpoint] 超参差异 ({len(diffs)} 项):")
                    for d in diffs:
                        print(d)
                else:
                    print(f"[Checkpoint] 超参与当前一致")
            self._resume_ckpt = ckpt  # 缓存，供 _restore_training_state 使用
        else:
            state_dict = ckpt
            print(f"[Checkpoint] 旧格式（仅权重），无超参数记录")
            self._resume_ckpt = None
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loaded model from {path}")

    def _restore_training_state(self):
        """从缓存的 checkpoint 恢复优化器、调度器及 best-metric 状态（须在 optimizer/scheduler 创建后调用）。"""
        ckpt = getattr(self, '_resume_ckpt', None)
        if ckpt is None:
            return
        reset_lr = getattr(self.args, 'reset_lr', False)
        if reset_lr:
            # 参考项目方式: 只恢复模型权重, optimizer/scheduler 从头开始
            # 适用于改变训练参数后的 fine-tune 场景（start_iter 保持 0，重跑完整轮次）
            print(f"[Resume] --reset_lr: 跳过优化器/调度器恢复, lr 从 {self.args.lr} 开始新 cosine 周期")
        else:
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f"[Resume] 优化器状态已恢复")
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print(f"[Resume] 调度器状态已恢复 (last_lr={self.scheduler.get_last_lr()[0]:.6f})")
            # 中断-续跑语义: 从存档迭代继续，总步数 = num_iters
            self.start_iter = int(ckpt.get('iteration', 0))
            if self.start_iter > 0:
                print(f"[Resume] 从 iter {self.start_iter} 继续 (剩余 "
                      f"{self.args.num_iters - self.start_iter} 步)")
            # RNG 轨迹恢复（跨机器 GPU 数不同等情况下降级为警告）
            rng = ckpt.get('rng_state')
            if rng is not None:
                try:
                    random.setstate(rng['python'])
                    np.random.set_state(rng['numpy'])
                    torch.set_rng_state(rng['torch_cpu'].cpu())
                    if rng.get('torch_cuda') is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(
                            [s.cpu() for s in rng['torch_cuda']])
                    print(f"[Resume] RNG 状态已恢复")
                except Exception as e:  # 不因 RNG 恢复失败而中断训练
                    print(f"[Resume] 警告: RNG 状态恢复失败 ({e}), 随机轨迹将不连续")
        # 恢复 best metric 追踪
        if 'best_ar' in ckpt:
            self.best_ar = ckpt['best_ar']
            self.best_ar_iter = ckpt.get('best_ar_iter', -1)
            print(f"[Resume] best_ar={self.best_ar:.4f} @ iter {self.best_ar_iter}")
        if 'best_task_score' in ckpt:
            self.best_task_score = ckpt['best_task_score']
            self.best_task_iter = ckpt.get('best_task_iter', -1)
            print(f"[Resume] best_task_score={self.best_task_score:.4f} @ iter {self.best_task_iter}")
        # EMA 影子参数恢复
        if 'ema_state_dict' in ckpt and self.ema_shadow is not None:
            for k in self.ema_shadow:
                if k in ckpt['ema_state_dict']:
                    self.ema_shadow[k].copy_(ckpt['ema_state_dict'][k].to(self.device))
            print(f"[Resume] EMA 影子参数已恢复")
        # 注意: 不在此处删除 _resume_ckpt，CMA-ES 控制器状态在控制器创建后恢复
    
    def _smooth_dict(self, ori_dict):
        """平滑指标记录"""
        for k, v in ori_dict.items():
            self.scaler_q[k].append(float(v))
    
    def run_episode(self, iteration, skip_scene_gen=False):
        """
        运行一个 episode 并返回损失
        
        Args:
            iteration: 当前迭代号，-1 表示 CMA-ES 评估
            skip_scene_gen: 跳过场景/动态障碍物生成（CMA-ES 评估时复用当前场景）
        """
        args = self.args
        B = args.batch_size
        
        # 重置环境和模型
        self.env.reset()
        self.model.reset()
        
        # 随机场景生成 (如果启用)
        # skip_scene_gen=True 时跳过，CMA-ES 评估复用训练 episode 的场景
        if not skip_scene_gen and self.env.enable_random_scene and self.env.scene_generator is not None:
            self.env.randomize_scene()

        # 动态障碍物随机化 (如果启用)
        if not skip_scene_gen and self.env.enable_dynamic_obstacles:
            self.env.randomize_dynamic_obstacles(
                arena_range=getattr(self.args, 'arena_range', 6.0),
            )
        
        # 安全出生点采样 (如果启用)
        # 随机场景时自动启用跨地图模式，确保飞行路径穿越障碍区域
        use_cross_map = getattr(self.args, 'force_cross_map', False) or self.env.enable_random_scene
        # 采样范围必须与障碍物分布范围一致
        spawn_arena = getattr(self.args, 'arena_range', 6.0) if self.env.enable_random_scene \
                      else getattr(self.args, 'init_p_range', 8.0)
        if self.safe_spawn:
            if use_cross_map and self.env.enable_random_scene:
                # 跨地图模式：出生/目标在对向两侧，防止绕行
                spawn_z_max = getattr(self.args, 'spawn_z_max', 3.0)
                _, p_target = self.env.safe_reset_cross_map(
                    arena_range=spawn_arena,
                    z_range=(1.0, spawn_z_max),
                )
                cross_map_mode = True
            else:
                self.env.safe_reset(
                    arena_range=spawn_arena,
                    z_range=(1.0, getattr(self.args, 'spawn_z_max', 3.0)),
                )
                cross_map_mode = False
        else:
            cross_map_mode = False
        
        # 历史记录
        p_history = []
        v_history = []
        target_v_history = []
        target_dist_history = []
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
                arena_range=spawn_arena,
                z_range=(1.0, spawn_z_max),
                min_distance=3.0,
                max_distance=spawn_arena * 1.2,
            )
        else:
            # 原始方式：基于角度/距离偏移，不检查碰撞
            angle = torch.rand(B, device=self.device) * 2 * torch.pi
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
        p_start = self.env.p.detach().clone()  # CMA-ES fitness 计算用
        
        # 个体随机化最大速度 (参考项目逻辑: 0.75 + 2.5 * rand) 我给他弄快了一点
        # 重要：这个值在整个 episode 中应保持不变
        max_speed = 0.75 + 3 * torch.rand((B, 1), device=self.device)  # 随机范围 [0.75, 3.75) m/s
        # max_speed = torch.full((B, 1), 6.0, device=self.device)  # 固定最大速度 6.0 m/s
        
        # 推力估计误差 (模拟真实无人机的推力不确定性)
        thr_est_error = 1.0 + 0.01 * torch.randn((B, 1), device=self.device)
        
        # Per-sample 相机安装旋转矩阵和位置偏移
        from drone_renderer import build_cam_mount_R
        if getattr(args, 'cam_mode', 'auto') == 'manual' and self.env._cam_manual_R is not None:
            # 手动模式：固定使用用户提供的外参矩阵，无随机化
            cam_mount_R = self.env._cam_manual_R.unsqueeze(0).expand(B, -1, -1)
            cam_offset_body = self.env._cam_manual_t.unsqueeze(0).expand(B, -1)
        else:
            # 自动模式：含三轴随机化，模拟每架无人机相机安装角的个体差异
            rpy_rand = getattr(args, 'cam_rand_rpy', 2.0)
            cam_pitch_per_sample = args.cam_angle + torch.randn(B, device=self.device)
            cam_roll_per_sample = getattr(args, 'cam_mount_roll', 0.0) + rpy_rand * torch.randn(B, device=self.device)
            cam_yaw_per_sample = getattr(args, 'cam_mount_yaw', 0.0) + rpy_rand * torch.randn(B, device=self.device)
            cam_mount_R = build_cam_mount_R(
                roll_deg=cam_roll_per_sample,
                pitch_deg=cam_pitch_per_sample,
                yaw_deg=cam_yaw_per_sample,
                device=self.device, batch_size=B,
            )

            # 基准偏移由网格几何自动计算，已按当前 margin 缩放
            cam_offset_base = self.env.get_scaled_cam_offset()  # (B, 3)
            xy_rand = getattr(args, 'cam_rand_xy', 0.02)
            z_lo, z_hi = getattr(args, 'cam_rand_z_range', [-0.04, 0.04])
            cam_offset_body = cam_offset_base + torch.stack([
                xy_rand * torch.randn(B, device=self.device),
                xy_rand * torch.randn(B, device=self.device),
                torch.rand(B, device=self.device) * (z_hi - z_lo) + z_lo,
            ], dim=-1)
        
        # 航向漂移 (可选)
        if args.yaw_drift:
            drift_av = torch.randn(B, device=self.device) * (5 * torch.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            R_drift = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ], -1).reshape(B, 3, 3)
        
        # CMA-ES DecayController 跟踪
        prev_decay = None  # 首步使用默认 grad_decay
        decay_history = []

        # CMA-ES MetaController (A-网络): 在 episode 开始前查询,
        # 同时设置 grad_decay (物理仿真用) 和 loss 系数 (损失计算用)
        self._meta_grad_decay = None
        if self.meta_controller is not None:
            progress = iteration / max(self.args.num_iters, 1) if iteration >= 0 else 0.5
            with torch.no_grad():
                meta_out = self.meta_controller(progress)
            merged = dict(self._base_coefs)
            for k, v in meta_out.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if k == 'grad_decay':
                    self._meta_grad_decay = val
                    prev_decay = val  # 全局标量, broadcast 到所有 sample
                elif k in merged:
                    merged[k] = val
            self.losser.coefs = merged
        
        # 预生成全部时间步的随机 dt (GPU tensor, 避免每步 CPU normalvariate)
        dt_all = torch.normal(
            mean=self.ctl_dt,
            std=self.ctl_dt * 0.1,
            size=(args.timesteps,),
            device=self.device,
        ).clamp(min=self.ctl_dt * 0.5)  # 下限防止负值
        # 一次性转 CPU，避免循环内每步 .item() 触发 GPU→CPU 同步
        dt_all_cpu = dt_all.tolist()
        
        # 视频帧缓存 (在循环外收集, 避免 save iter 时热循环中的 GPU→CPU 同步)
        vid_depth_indices = []
        
        for t in range(args.timesteps):
            current_dt = dt_all_cpu[t]
            
            # 渲染深度图 - no_grad()
            # 参考项目验证：渲染管线不参与梯度反向传播。参考项目 CUDA raycasting
            # 写入 torch.empty (切断计算图)，梯度仅通过物理模拟路径回传。
            # PyTorch3D 同理：深度图仅作为 sensor（状态→观测），不是可微路径。
            # 跳帧优化: 深度图不参与梯度计算, 相邻帧变化很小, 可隔帧渲染节省开销
            if t % args.render_interval == 0:
                with torch.no_grad():
                    _, depth = self.env.render(
                        cam_mount_R=cam_mount_R,
                        cam_offset_body=cam_offset_body,
                        return_tensor=True,
                        return_rgb=False,
                        return_depth=True,
                        dt=current_dt
                    )
                # depth = depth.requires_grad_(False)
            
            # 记录 step 之前的状态 (与参考项目一致)
            p_history.append(self.env.p)
            vec_to_pt_history.append(self.env.combined_vec_to_nearest(dt=current_dt))
            
            # 标记需要保存的帧 (延迟到循环结束后批量 CPU 转移)
            if is_save_iter(iteration) and B > 4:
                vid_depth_indices.append((t, depth[4].detach().clone()))
            
            # 更新目标向量 (可选航向漂移)
            if args.yaw_drift:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            else:
                target_v_raw = p_target - self.env.p.detach()  # detach 防止通过瞬移优化
            target_dist_history.append(torch.norm(target_v_raw, p=2, dim=-1))
            
            # 执行动作 (使用延迟缓冲中的动作)
            # 关键：使用 act_buffer[t] 而不是取模，参考项目的实现方式
            self.env.step(act_cmd=act_buffer[t], 
                         target_pos_vector=target_v_raw, 
                         dt=current_dt,
                         override_grad_decay=prev_decay)
            
            # 策略推理（obs 构造 → 模型前向 → 动作后处理）
            act_cmd, v_pred, target_v, h = self.policy.infer(
                depth, self.env.R, self.env.v, target_v_raw,
                self.env.margin, max_speed, thr_est_error, h,
                depth_noise_std=0.02,
            )
            
            # CMA-ES DecayController: 从当前帧的 CNN 特征计算下一步的梯度衰减
            if self.decay_controller is not None:
                img_feat = getattr(self.policy, '_last_img_feat', None)
                if img_feat is not None:
                    with torch.no_grad():
                        prev_decay = self.decay_controller(img_feat.detach())
                    decay_history.append(prev_decay)

            v_preds.append(v_pred)
            act_buffer.append(act_cmd)
            
            v_history.append(self.env.v)
            target_v_history.append(target_v)

        p_history = torch.stack(p_history)          # (T, B, 3)
        v_history = torch.stack(v_history)          # (T, B, 3)
        target_v_history = torch.stack(target_v_history)  # (T, B, 3)
        target_dist_history = torch.stack(target_dist_history)  # (T, B)
        vec_to_pt_history = torch.stack(vec_to_pt_history)  # (T, S, B, 3) 子步细分
        v_preds = torch.stack(v_preds)              # (T, B, 3)
        act_buffer_stacked = torch.stack(act_buffer)  # (T + lag + 1, B, 3)
        
        # 批量 GPU→CPU 转移视频帧 (不在热循环中做)
        for _, frame_gpu in vid_depth_indices:
            vid.append(frame_gpu.cpu())

        
        # CMA-ES LossGuide: 注入进化的损失系数
        if self.loss_guide is not None:
            with torch.no_grad():
                evolved_coefs = self.loss_guide()
            # 合并：进化系数覆盖对应项，未覆盖的保持初始值
            merged = dict(self._base_coefs)
            merged.update({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in evolved_coefs.items()})
            self.losser.coefs = merged
        
        # 计算损失
        loss, metrics = self.losser.forward(
            p_history=p_history,
            v_history=v_history,
            target_vel_history=target_v_history,
            act_history=act_buffer_stacked,
            vec_to_obj_history=vec_to_pt_history,
            v_preds=v_preds,
            env_margin=self.env.margin,
            env_g_std=self.g_std,
            inter_drone_dist_history=None,
        )
        
        # A-网络 LossNetwork: 用进化的神经网络替代线性损失组合
        # 注意: 此处 **不能** 用 torch.no_grad(), 梯度必须流过 LossNetwork 到策略网络
        if self.loss_network is not None:
            progress = iteration / max(self.args.num_iters, 1) if iteration >= 0 else 0.5
            loss = self.loss_network(metrics, progress)
        
        # 目标达成辅助损失: 直接优化终点与目标的距离
        # 补充密集速度损失(每步指导)与稀疏终端信号(全局目标)的双重监督
        coef_goal = getattr(self.args, 'coef_goal_reaching', 0.0)
        if coef_goal > 0:
            final_dist = (p_target - p_history[-1]).norm(dim=-1)  # (B,)
            loss_goal_reaching = final_dist.mean()
            loss = loss + coef_goal * loss_goal_reaching
            metrics['loss_goal_reaching'] = loss_goal_reaching
        
        # 计算额外指标
        with torch.no_grad():
            distance = torch.norm(vec_to_pt_history, 2, -1) - self.env.margin
            speed_history = v_history.norm(2, -1)
            collision_history = distance <= 0
            metrics.update(
                compute_navigation_metrics_torch(
                    target_dist_history=target_dist_history,
                    collision_history=collision_history,
                    speed_history=speed_history,
                    reach_radius=getattr(args, 'reach_radius', 0.5),
                )
            )
            # CMA-ES decay 统计
            if decay_history:
                decay_vals = torch.stack(decay_history)
                metrics['decay_mean'] = decay_vals.mean().item()
                metrics['decay_std'] = decay_vals.std().item()

            # MetaController (A-网络): 更新训练状态 EMA + 记录当前调制值
            if self.meta_controller is not None:
                self.meta_controller.update_ema(metrics)
                # 记录所有 meta 输出到 metrics
                progress = iteration / max(self.args.num_iters, 1) if iteration >= 0 else 0.5
                with torch.no_grad():
                    meta_out = self.meta_controller(progress)
                for k, v in meta_out.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    metrics[f'meta_{k}'] = val

            # LossNetwork: 记录网络输出的损失值 (区别于线性组合的损失)
            if self.loss_network is not None:
                metrics['loss_net_output'] = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # 非保存迭代时 detach debug_data，避免计算图被引用残留到下一轮
        debug_out = (
            p_history.detach(), v_history.detach(),
            act_buffer_stacked.detach(), vid
        )
        
        # CMA-ES 评估所需的额外数据
        extra = {
            'p_history': p_history.detach(),
            'p_start': p_start,
            'p_target': p_target,
            'vec_to_obj_history': vec_to_pt_history.detach(),
            'margin': self.env.margin,
        }
        
        return loss, metrics, debug_out, extra
    
    def compute_fitness(self, extra):
        """
        CMA-ES 适应度函数。
        
        Fitness = progress × (1 - collision_rate) × (1 + 0.1 × avg_speed)
        
        防作弊设计:
        - 原地不动: progress=0 → fitness=0
        - 碰撞: collision=1 → fitness=0
        """
        p_history = extra['p_history']
        p_start = extra['p_start']
        p_target = extra['p_target']
        vec_to_obj_history = extra['vec_to_obj_history']
        margin = extra['margin']
        
        total_dist = (p_target - p_start).norm(dim=-1)  # (B,)
        final_dist = (p_target - p_history[-1]).norm(dim=-1)  # (B,)
        progress = (1.0 - final_dist / (total_dist + 1e-6)).clamp(0, 1)
        
        distance = vec_to_obj_history.norm(dim=-1)
        if distance.dim() == 3:  # (T, S, B) 子步细分
            distance = distance.amin(dim=1)
        distance = distance - margin
        collision = (distance < 0).any(dim=0).float()
        
        if p_history.shape[0] > 1:
            avg_speed = (p_history[1:] - p_history[:-1]).norm(dim=-1).mean(0)
        else:
            avg_speed = torch.zeros(p_history.shape[1], device=p_history.device)
        
        per_sample = progress * (1.0 - collision) * (1.0 + 0.1 * avg_speed)
        return per_sample.mean().item()
    
    def evaluate_cma_individual(self, params_vector):
        """评估单个 CMA-ES 个体的 fitness（复用当前场景，不重新生成）"""
        vec = torch.tensor(params_vector, dtype=torch.float32, device=self.device)
        self._cma_target.set_params_vector(vec)
        
        with torch.no_grad():
            _, _, _, extra = self.run_episode(iteration=-1, skip_scene_gen=True)
        return self.compute_fitness(extra)

    def _evaluate_cma_solutions(self, solutions):
        """
        批量评估 CMA-ES 种群。

        loss_net / guide 模式: A-网络不影响轨迹仿真，只影响损失函数。
        每个个体仍需独立 episode 以获取无偏 fitness (不同初始条件 → 不同轨迹质量)。
        场景复用 (skip_scene_gen=True) 避免重复生成障碍物。

        decay / meta 模式: A-网络影响物理仿真 (grad_decay)，
        不同个体产生不同轨迹，必须逐个运行。

        Returns:
            list[float]: CMA-ES 适应度值 (已取负, 待 CMA 最小化)
        """
        # 保存 MetaController EMA, 防止评估 episode 污染训练状态
        ema_backup = None
        if self.meta_controller is not None:
            ema_backup = dict(self.meta_controller._ema)

        save_vec = self._cma_target.get_params_vector().clone()
        fitnesses = []
        for sol in solutions:
            try:
                fit = self.evaluate_cma_individual(sol)
                fitnesses.append(-fit)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                fitnesses.append(0.0)

        # 恢复评估前参数和 EMA
        self._cma_target.set_params_vector(save_vec)
        if ema_backup is not None:
            self.meta_controller._ema = ema_backup
        return fitnesses
    
    def train(self):
        """主训练循环"""
        args = self.args
        
        pbar = tqdm(range(self.start_iter, args.num_iters), ncols=160,
                    initial=self.start_iter, total=args.num_iters,
                    bar_format='{l_bar}{bar:20}{r_bar}')
        cma_gen = 0
        
        for i in pbar:
            try:
                # 运行一个 episode
                # 损失课程学习: 在训练前期使用稳定的 MSE 损失，后期切换到 adaptive
                if getattr(args, 'loss_curriculum', False):
                    progress = i / max(args.num_iters, 1)
                    transition = getattr(args, 'curriculum_transition', 0.4)
                    self.losser.loss_v_mode = 'mse' if progress < transition else 'adaptive'

                loss, metrics, debug_data, extra = self.run_episode(i)
                
                # 检查 NaN
                if torch.isnan(loss):
                    print("Loss is NaN, exiting...")
                    break
                
                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 记录裁剪前的梯度范数（用于分析梯度爆炸现象）
                _total_norm_sq = 0.0
                for _p in self.model.parameters():
                    if _p.grad is not None:
                        _total_norm_sq += _p.grad.data.norm(2).item() ** 2
                metrics['grad_norm'] = _total_norm_sq ** 0.5
                if getattr(args, 'grad_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                # EMA 影子参数更新
                if self.ema_shadow is not None:
                    with torch.no_grad():
                        for k, v in self.model.state_dict().items():
                            self.ema_shadow[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
            except torch.cuda.OutOfMemoryError:
                # OOM 时 Python traceback 会持有 run_episode 内所有局部变量的引用
                # （包括计算图、历史 tensor 等），必须先 gc.collect() 打断引用链
                self.optimizer.zero_grad(set_to_none=True)  # 释放残留梯度显存
                gc.collect()
                torch.cuda.empty_cache()
                print(f"\n[OOM] iter {i}: episode OOM, skipping")
                continue
            
            # CMA-ES 外层循环: 预热期后每 N 步进化一代
            cma_warmup = getattr(args, 'cma_warmup', 0)
            if self.use_cmaes and (i + 1) >= cma_warmup and (i + 1) % args.cma_eval_interval == 0 and not self.cma_es.stop():
                best_before = self._cma_target.get_params_vector().clone()
                
                solutions = self.cma_es.ask()
                fitnesses = self._evaluate_cma_solutions(solutions)
                
                self.cma_es.tell(solutions, fitnesses)
                cma_gen += 1
                
                # 恢复当前最优个体
                best = self.cma_es.result.xbest
                if best is not None:
                    vec = torch.tensor(best, dtype=torch.float32, device=self.device)
                    self._cma_target.set_params_vector(vec)
                else:
                    self._cma_target.set_params_vector(best_before)
                
                # 记录 CMA-ES 指标
                best_fitness = -min(fitnesses)
                mean_fitness = -sum(fitnesses) / len(fitnesses)
                metrics['cma_best_fitness'] = best_fitness
                metrics['cma_mean_fitness'] = mean_fitness
                metrics['cma_gen'] = cma_gen
                
                # LossGuide 模式：记录当前进化的系数
                if self.loss_guide is not None:
                    with torch.no_grad():
                        current_coefs = self.loss_guide()
                    for name, val in current_coefs.items():
                        v = val.item() if isinstance(val, torch.Tensor) else val
                        metrics[f'guide_{name}'] = v

                # MetaController (A-网络) 模式: 记录当前调制输出
                if self.meta_controller is not None:
                    progress = (i + 1) / max(self.args.num_iters, 1)
                    with torch.no_grad():
                        current_meta = self.meta_controller(progress)
                    for name, val in current_meta.items():
                        v = val.item() if isinstance(val, torch.Tensor) else val
                        metrics[f'meta_{name}'] = v
            
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
                # 先把指标缓冲落盘，保证磁盘 CSV 覆盖到本 checkpoint 的迭代，
                # 中断后从该 checkpoint 续跑不会留指标空洞
                self.monitor.flush()
                save_path = os.path.join(args.save_dir, f'checkpoint_{i+1:06d}.pth')
                torch.save(self._make_checkpoint(i + 1), save_path)
                print(f"\nSaved checkpoint to {save_path}")
            
            # ---- Best AR checkpoint（独立保存，不覆盖常规 checkpoint）----
            current_ar = float(metrics.get('ar', 0.0))
            self.ar_ema = self.ar_ema_alpha * current_ar + (1 - self.ar_ema_alpha) * self.ar_ema
            # 前 200 步预热：EMA 还不稳定，跳过
            if i >= 200 and self.ar_ema > self.best_ar:
                self.best_ar = self.ar_ema
                self.best_ar_iter = i + 1
                best_path = os.path.join(args.save_dir, 'best_ar.pth')
                torch.save(self._make_checkpoint(i + 1, {'best_ar': self.best_ar}), best_path)
                # EMA 版 best checkpoint: 用影子参数替换 model_state_dict
                if self.ema_shadow is not None:
                    ema_ckpt = self._make_checkpoint(i + 1, {'best_ar': self.best_ar})
                    ema_ckpt['model_state_dict'] = {k: v.cpu() for k, v in self.ema_shadow.items()}
                    torch.save(ema_ckpt, os.path.join(args.save_dir, 'best_ar_ema.pth'))
                # 同时记录到 TensorBoard
                self.writer.add_scalar('best_ar', self.best_ar, i + 1)

            current_task_score = float(metrics.get('task_score', 0.0))
            self.task_score_ema = self.ar_ema_alpha * current_task_score + (1 - self.ar_ema_alpha) * self.task_score_ema
            if i >= 200 and self.task_score_ema > self.best_task_score:
                self.best_task_score = self.task_score_ema
                self.best_task_iter = i + 1
                best_task_path = os.path.join(args.save_dir, 'best_task_score.pth')
                torch.save(self._make_checkpoint(i + 1, {'best_task_score': self.best_task_score}), best_task_path)
                self.writer.add_scalar('best_task_score', self.best_task_score, i + 1)
            
            if (i + 1) % 25 == 0:
                for k, v in self.scaler_q.items():
                    self.writer.add_scalar(k, sum(v) / len(v), i + 1)
                self.scaler_q.clear()
        
        # 保存最终模型
        final_extra = {
            'best_ar': self.best_ar,
            'best_ar_iter': self.best_ar_iter,
            'best_task_score': self.best_task_score,
            'best_task_iter': self.best_task_iter,
        }
        final_path = os.path.join(args.save_dir, 'checkpoint_final.pth')
        torch.save(self._make_checkpoint(args.num_iters, final_extra), final_path)
        # EMA 版最终 checkpoint
        if self.ema_shadow is not None:
            ema_final = self._make_checkpoint(args.num_iters, final_extra)
            ema_final['model_state_dict'] = {k: v.cpu() for k, v in self.ema_shadow.items()}
            ema_final_path = os.path.join(args.save_dir, 'checkpoint_final_ema.pth')
            torch.save(ema_final, ema_final_path)
            print(f"EMA final checkpoint saved to {ema_final_path}")
        print(f"Training complete. Final checkpoint saved to {final_path}")
        if self.best_ar_iter > 0:
            print(f"Best AR model: best_ar.pth (AR={self.best_ar:.4f} @ iter {self.best_ar_iter})")
        if self.best_task_iter > 0:
            print(f"Best task-score model: best_task_score.pth (score={self.best_task_score:.4f} @ iter {self.best_task_iter})")
        
        self.monitor.close()
        self.writer.close()
    
    def _log_figures(self, iteration, debug_data):
        """记录可视化图表"""
        p_history, v_history, act_buffer, vid = debug_data
        
        if p_history.shape[1] <= 4:
            return
        
        # 位置历史图
        fig_p, ax = plt.subplots()
        p_cpu = p_history[:, 4].detach().cpu()
        ax.plot(p_cpu[:, 0], label='x')
        ax.plot(p_cpu[:, 1], label='y')
        ax.plot(p_cpu[:, 2], label='z')
        ax.legend()
        ax.set_title('Position History')
        
        # 速度历史图
        fig_v, ax = plt.subplots()
        v_cpu = v_history[:, 4].detach().cpu()
        ax.plot(v_cpu[:, 0], label='x')
        ax.plot(v_cpu[:, 1], label='y')
        ax.plot(v_cpu[:, 2], label='z')
        ax.legend()
        ax.set_title('Velocity History')
        
        # 动作历史图
        fig_a, ax = plt.subplots()
        act_cpu = act_buffer[:, 4].detach().cpu()
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

    # 固定随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"  随机种子:   {args.seed}")

    print("Training arguments:")
    print(args)
    
    # 启用调试模式
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled!")
    
    if args.env_type == 'go2':
        from diffsim.go2.trainer import Go2Trainer

        trainer = Go2Trainer(args)
    else:
        trainer = DroneTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
