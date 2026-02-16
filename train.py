"""
无人机可微分仿真训练脚本

基于参考项目 DiffPhysDrone 的训练逻辑实现
"""

import os
import gc
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
from model import Model, Model_bigger, Model_bigger_yaw
from loss import DroneLoss
from scene_generator import SceneGenerator
from training_monitor import TrainingMonitor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机避障训练脚本')
    
    # 训练参数
    parser.add_argument('--resume', default=None,
                        help='恢复训练的模型 checkpoint 路径 (.pth)。\n'
                             '支持跨架构热启动（如 Model_bigger → Model_bigger_yaw），\n'
                             '维度不匹配的层会自动零填充新增维度')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每个 episode 并行仿真的无人机数量。\n'
                             '影响显存占用和梯度估计方差。参考项目默认 16。\n'
                             '48x64 分辨率下 B=16 约需 4GB 显存，B=32 约需 7GB'
                             '存在边际递减效应,B=128只需要9.7G')
    parser.add_argument('--num_iters', type=int, default=50000,
                        help='总训练迭代次数。每次迭代运行一个完整 episode。\n'
                             '通常 5000 步可见初步避障行为，20000+ 步趋于收敛')
    parser.add_argument('--timesteps', type=int, default=150,
                        help='每个 episode 的仿真步数。在 ctl_dt=1/15s 下，\n'
                             '150 步 = 10 秒飞行时间。更长的 episode 能学到更远距离\n'
                             '的导航策略，但显存和计算量线性增加')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='AdamW 优化器初始学习率。使用 CosineAnnealing 调度，\n'
                             '从 lr 退火到 lr*0.01。参考项目使用 1e-3，太大会导致\n'
                             '策略抖动，太小收敛慢')
    parser.add_argument('--grad_decay', type=float, default=0.4,
                        help='梯度衰减系数 (0~1)。每个仿真步对位置/速度梯度乘以\n'
                             'grad_decay^dt，防止长序列反向传播时梯度爆炸。\n'
                             '0.4 表示每步衰减约 95%%，使近期步的梯度信号远强于早期步。\n'
                             '设为 1.0 则不衰减（通常会导致训练不稳定）')
    parser.add_argument('--ctl_dt', type=float, default=1/15,
                        help='控制循环时间步长 (秒)。即模型每 ctl_dt 秒输出一次动作。\n'
                             '参考项目使用 1/15≈0.067s (15Hz)。训练时会在此基础上\n'
                             '添加 ±10%% 随机抖动模拟真实控制器的时钟不确定性')
    parser.add_argument('--render_interval', type=int, default=1, 
                        help='渲染间隔帧数 (1=每帧渲染, 2=隔帧渲染, 节省渲染开销)。\n'
                             '强烈不建议设为 >1，隔帧渲染会导致观测严重滞后，\n'
                             '模型性能大幅下降且训练反而更慢（梯度信号更差需要更多步收敛）')
    parser.add_argument('--arrival_threshold', type=float, default=1.0,
                        help='到达判定阈值 (米)。当任一时刻到目标距离 <= 该值时判定为到达。\n'
                            '训练阶段 success_rate 定义为“无碰撞且到达”')
    
    # 损失函数权重
    parser.add_argument('--coef_v', type=float, default=1.0,
                        help='速度跟踪损失权重。计算滑动窗口平均速度与目标速度的 SmoothL1 损失，\n'
                             '是驱动无人机飞向目标的核心损失项。\n'
                             '参考项目默认 1.0。设为 0 则无人机不会主动飞向目标')
    parser.add_argument('--coef_speed', type=float, default=0.0,
                        help='速度标量损失权重 (legacy，已弃用)。\n'
                             '只约束速度大小不约束方向，效果不如 coef_v。保留仅为兼容性')
    parser.add_argument('--coef_v_pred', type=float, default=2.0,
                        help='速度预测辅助损失权重。模型同时输出加速度指令和速度预测，\n'
                             '速度预测与实际速度的 MSE 损失作为自监督信号，\n'
                             '帮助模型建立"动作→状态变化"的内部模型。\n'
                             '默认 2.0。通常不需要调整')
    parser.add_argument('--coef_collide', type=float, default=5.0,
                        help='碰撞损失权重 (参考项目单机配置=7.5, 多机=5.0)')
    parser.add_argument('--coef_obj_avoidance', type=float, default=3.0,
                        help='障碍物回避损失权重 (参考项目单机配置=3.0, 多机=2.0)')
    parser.add_argument('--coef_d_acc', type=float, default=0.01,
                        help='加速度 L2 正则化权重。惩罚过大的推力指令，\n'
                             '使飞行更加平滑省电。参考项目默认 0.01')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001,
                        help='加加速度 (jerk) 正则化权重。惩罚加速度的快速变化，\n'
                             '抑制动作抖动。jerk = d(acc)/dt，乘以 1/ctl_dt 归一化。\n'
                             '参考项目默认 0.001。过大会使避障反应变迟钝')
    parser.add_argument('--coef_d_snap', type=float, default=0.0,
                        help='snap (加加加速度) 正则化权重 (legacy，默认禁用)。\n'
                             '惩罚推力方向的二阶变化率。参考项目中有定义但通常不启用，\n'
                             '因 jerk 正则化已足够平滑')
    parser.add_argument('--coef_ground_affinity', type=float, default=0.0,
                        help='高度惩罚损失权重。惩罚 Z>0 (ROS坐标系中 Z 向上) 的飞行高度，\n'
                             '防止模型学会"飞到障碍物上方"来规避碰撞。\n'
                             '用于低空穿越场景，默认禁用')
    parser.add_argument('--coef_bias', type=float, default=0.0,
                        help='方向偏差损失权重。惩罚速度中垂直于目标方向的分量，\n'
                             '鼓励模型沿目标方向直线飞行。默认禁用，\n'
                             '因为过强会阻碍绕行行为')
    parser.add_argument('--coef_stall', type=float, default=0.0,
                        help='停滞惩罚权重，惩罚速度低于 0.3m/s 的状态。'
                             '打破正对障碍物时"原地不动"的局部极小值，实测是劣化项')
    parser.add_argument('--coef_progress', type=float, default=0.0,
                        help='路径进度损失权重。奖励任何缩短与目标距离的运动，\n'
                             '允许模型绕行而不被强制沿直线飞行。推荐值 0.3~1.0')
    parser.add_argument('--window_size', type=int, default=30,
                        help='速度跟踪损失的滑动平均窗口大小 (帧数)。\n'
                             '对 v_history 做 window_size 帧的移动平均后再与目标速度比较，\n'
                             '平滑瞬时速度波动。在 15Hz 下 30 帧 = 2 秒平均。\n'
                             '过小导致高频抖动被惩罚，过大导致响应迟缓')
    
    # 环境参数 - 渲染
    parser.add_argument('--cam_angle', type=int, default=10,
                        help='相机安装俱仰角 (度)。正值表示向下倾斜（俸视）。\n'
                             '训练时会为每个 batch 样本添加 ±1° 的随机偏移，\n'
                             '模拟真实无人机相机安装角的个体差异。\n'
                             '10° 适合前方+地面均可见；增大可看到更多地面但丢失远处信息')
    parser.add_argument('--image_height', type=int, default=48,
                        help='渲染深度图高度 (像素)。与 image_width 共同决定渲染分辨率。\n'
                             '48x64 为低分辨率快速训练配置 (Model_bigger)；\n'
                             '240x320 为中分辨率配置 (Model_adaptive)；\n'
                             '分辨率越高渲染越慢但感知细节更丰富')
    parser.add_argument('--image_width', type=int, default=64,
                        help='渲染深度图宽度 (像素)。宽高比建议保持 4:3 或 16:9。\n'
                             '模型输入分辨率必须与此一致 (Model_bigger 要求 48x64)')
    parser.add_argument('--hfov', type=float, default=90.0,
                        help='相机水平视场角 (度)，默认90°。焦距由 FOV 和图像宽度自动计算')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample4.obj',
                        help='障碍物场景 .obj 网格文件路径。仅在未启用 --random_scene 时使用。\n'
                             '网格会被加载到 PyTorch3D 渲染器，同时从表面采样点云用于碰撞检测。\n'
                             'data/sample/ 下有多个预制场景可选')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='从障碍物网格表面采样的点云点数。用于 KNN 最近邻碰撞检测。\n'
                             '点数越多碰撞检测精度越高，但 KNN 查询更慢。\n'
                             '简单场景 50000 足够，复杂随机场景建议 100000+')
    parser.add_argument('--subdivide_times', type=int, default=0,
                        help='网格细分次数 (默认0, 配合z_clip无质量损失且渲染快)')
    parser.add_argument('--depth_min', type=float, default=0.3,
                        help='深度图近截断距离 (米)，近于此的深度被截断。同时也是渲染器近平面裁剪值')
    parser.add_argument('--depth_max', type=float, default=10.0,
                        help='深度图远截断距离 (米)，远于此的深度被截断。'
                             '参考项目使用 24.0；过小会导致模型"近视"来不及避障')
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
    parser.add_argument('--force_cross_map', action='store_true', default=False,
                        help='强制出生/目标点在场景对向两侧（防止绕行）')
    parser.add_argument('--spawn_z_max', type=float, default=3.0,
                        help='出生/目标点最大高度（防止飞高规避）')
    
    # 环境参数 - 无人机物理
    parser.add_argument('--margin_min', type=float, default=0.3,
                        help='无人机安全碰撞半径区间下界 (米)。每个 batch 样本的 margin\n'
                             '在 [margin_min, margin_max] 间随机采样。\n'
                             'margin 用于碰撞检测：distance_to_obstacle - margin < 0 则碰撞。\n'
                             '随机化 margin 使模型不依赖固定机体大小，增强鲁棒性')
    parser.add_argument('--margin_max', type=float, default=0.8,
                        help='无人机安全碰撞半径区间上界 (米)。\n'
                             '建议设为实际无人机外接球半径的 1.5~2 倍。\n'
                             '过大会导致狭窄通道无法通过，过小会增加擦碰概率')
    parser.add_argument('--init_p_range', type=float, default=8.0,
                        help='无人机初始位置随机范围 (米)。\n'
                             '出生点 XY 在 [-R, R] 内均匀采样，Z 在 [0.5, R+0.5] 内采样。\n'
                             '应与场景大小匹配（随机场景时使用 arena_range 代替）')
    parser.add_argument('--noise_std', type=float, default=0.04,
                        help='环境扰动加速度噪声标准差 (m/s²)。\n'
                             '模拟阵风/气流扰动，通过 Ornstein-Uhlenbeck 过程产生\n'
                             '时间相关的随机扰动。0.04 对应轻微扰动，0.2 对应强扰动')
    parser.add_argument('--yaw_inertia', type=float, default=5.0,
                        help='偏航惯性系数。控制机头朝向跟踪速度方向的灵敏度。\n'
                             '公式: heading_mix = old_heading * yaw_inertia + velocity。\n'
                             '值越大机头转向越慢（风向标效应更弱），\n'
                             '低速悬停时因速度小而自然保持当前朝向。参考项目默认 5.0')
    parser.add_argument('--yaw_ctl_delay', type=float, default=12.0,
                        help='偏航响应速率系数。控制机头朝向跟踪的指数平滑速度。\n'
                             '公式: alpha = exp(-delay * dt)，值越大响应越快。\n'
                             '注意命名容易误导：值越大延迟反而越小。参考项目默认 12.0')
    parser.add_argument('--pitch_ctl_delay', type=float, default=12.0,
                        help='姿态/推力响应速率系数。控制执行器（电机）从当前状态\n'
                             '趋向目标的低通滤波速度。公式: act_next = cmd*(1-α) + curr*α，\n'
                             '其中 α = exp(-delay * dt)。值越大电机响应越快。\n'
                             '参考项目默认 12.0。过大会放大高频噪声')
    parser.add_argument('--airmode_coef', type=float, default=0.5,
                        help='Airmode 效应系数。模拟真实无人机在快速改变推力方向时\n'
                             '产生的额外加速度（角速度诱导加速度）。\n'
                             '公式: a_airmode = thrust_dir * angular_vel * coef。\n'
                             '参考项目默认 0.5。设为 0 则禁用此效应')
    parser.add_argument('--enable_airmode', action='store_true', default=True,
                        help='启用 Airmode 效应模拟。Airmode 使无人机在急转弯时\n'
                             '产生额外推力方向加速度，更接近真实飞行动力学')
    parser.add_argument('--disable_airmode', action='store_true', default=False,
                        help='禁用 Airmode 效应模拟 (覆盖 --enable_airmode)。\n'
                             '在简化动力学实验或调试时使用')
    
    # 模型参数
    parser.add_argument('--no_odom', default=False, action='store_true',
                        help='不使用里程计速度作为观测输入。启用后 dim_obs 从 10 降为 7，\n'
                             '模型仅依赖深度图和目标方向，不知道自身速度。\n'
                             '用于测试纯视觉导航能力或模拟里程计故障场景')
    parser.add_argument('--yaw_drift', default=False, action='store_true',
                        help='启用航向漂移数据增强。模拟磁罗盘标定误差导致的\n'
                             '目标方向持续旋转 (每步约 ±1.2°)。\n'
                             '使模型学会容忍航向不确定性，增强鲁棒性')
    parser.add_argument('--enable_yaw_control', default=False, action='store_true',
                        help='启用模型自主偏航控制。模型额外输出 yaw_rate，累积为偏航偏移量\n'
                             '旋转目标方向向量，使无人机可以主动转头寻找绕行路径。\n'
                             '需要从头训练或从 Model_bigger checkpoint 热启动 (strict=False)')
    parser.add_argument('--coef_yaw_explore', type=float, default=1.0,
                        help='偏航探索损失权重 (仅 enable_yaw_control 时生效)。\n'
                             '低速时小角度偏航提供负损失（奖励探索），超过 X° 变为惩罚；\n'
                             '高速时任何偏航都是惩罚（抑制危险转向）')
    parser.add_argument('--yaw_penalty_start_deg', type=float, default=60.0,
                        help='偏航探索损失中“开始惩罚”的角度阈值（度）。\n'
                            '低速时 |yaw_offset| 小于该阈值为奖励区间，大于该阈值为惩罚区间'
                            '不建议设置的过大，模型会希望让偏航一直保持在奖励区间来获得持续负损失，导致过度转头和不稳定行为，会导致严重的训练退化'
                            '不建议设置的超过FOVX')
    parser.add_argument('--enable_panoramic', default=False, action='store_true',
                        help='启用全向障碍物感知。在水平面上将 360° 分为 N 个扇区，\n'
                             '每个扇区返回最近障碍物距离。解决纯前视深度图在面对大障碍物时\n'
                             '"不知道侧方是否有可通行路径"的信息缺失问题')
    parser.add_argument('--n_panoramic_sectors', type=int, default=8,
                        help='全向感知扇区数量（默认 8，每 45° 一个）')
    parser.add_argument('--panoramic_max_range', type=float, default=8.0,
                        help='全向感知最大探测距离（米）')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='启用 PyTorch autograd anomaly detection。\n'
                             '检测 NaN/Inf 梯度并打印产生异常梯度的前向传播位置。\n'
                             '会严重降低训练速度 (约 3~5x)，仅在排查梯度问题时使用')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型 checkpoint 保存目录。还会保存 TrainingMonitor 的\n'
                             'CSV 日志和损失曲线 PNG。常规 checkpoint 按 save_freq 保存，\n'
                             'Best AR checkpoint 独立保存为 best_ar.pth')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='常规 checkpoint 保存间隔 (迭代次数)。\n'
                             '文件名格式: checkpoint_NNNNNN.pth。\n'
                             '另外在前 2000 步每 250 步、之后每 1000 步会保存可视化图到 TensorBoard')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='TensorBoard 日志目录。每次训练会创建带时间戳的子目录，\n'
                             '如 logs/drone_train_20260215-143000/。\n'
                             '使用 tensorboard --logdir=./logs 查看')
    
    # 硬件参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的 GPU 编号。多 GPU 机器上指定训练使用的设备。\n'
                             '仅支持单 GPU 训练，不支持 DataParallel/DDP')
    
    return parser.parse_args()


def is_save_iter(i):
    """判断是否为保存迭代"""
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


class DroneTrainer:
    """
    无人机可微分仿真训练器。

    封装了完整的训练流程：环境初始化、模型创建、episode 仿真循环、
    损失计算、反向传播、TensorBoard 日志和 checkpoint 保存。

    训练流程:
        1. 每次迭代运行一个 episode (run_episode)
        2. episode 内：重置环境 → 采样目标 → 循环 {render → model → step → 记录}
        3. episode 结束后计算总损失，反向传播更新模型
        4. 定期保存 checkpoint 和可视化图表

    支持功能:
        - Model_bigger / Model_bigger_yaw 模型选择
        - 偏航控制 (--enable_yaw_control)
        - 全向障碍物感知 (--enable_panoramic)
        - 随机场景生成 (--random_scene)
        - 安全出生点/目标点采样 (--safe_spawn)
        - Best AR checkpoint 自动保存

    Args:
        args: 命令行参数 (argparse.Namespace)
    """
    
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
                ground_ratio=getattr(args, 'ground_ratio', 0.6),
                cluster_ratio=getattr(args, 'cluster_ratio', 0.3),
                cluster_spread=getattr(args, 'cluster_spread', 1.5),
            )
            print(f"[SceneGenerator] 已启用随机场景生成, "
                  f"障碍物数量: {args.num_obstacles_min}-{args.num_obstacles_max}, "
                  f"接地率: {args.ground_ratio:.0%}, "
                  f"簇生率: {args.cluster_ratio:.0%}, "
                  f"归一化+网格抖动+3D旋转")
        
        self.safe_spawn = getattr(args, 'safe_spawn', False) or getattr(args, 'random_scene', False)
        if self.safe_spawn:
            print(f"[SafeSpawn] 已启用安全出生点/目标点, 最小安全距离: {getattr(args, 'safe_clearance', 1.0)}")
        
        # 根据 FOV 计算焦距: f = (W/2) / tan(hfov/2)
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
            # 渲染优化: subdivide_times=0 在 48x64 下无质量损失, 面片从 106 万降至 1.6 万
            subdivide_times=args.subdivide_times,
            # 渲染近平面裁剪: 与 depth_min 对齐，避免 clip_faces OOM
            z_clip_value=getattr(args, 'depth_min', 0.3),
            # 场景随机化
            enable_random_scene=getattr(args, 'random_scene', False),
            scene_generator=self.scene_generator,
            safe_spawn_clearance=getattr(args, 'safe_clearance', 1.0),
        )
        
        # 初始化模型
        yaw_control = getattr(args, 'enable_yaw_control', False)
        panoramic = getattr(args, 'enable_panoramic', False)
        n_panoramic_sectors = getattr(args, 'n_panoramic_sectors', 8)
        # 偏航控制增加 1 维 obs (yaw_offset)
        obs_extra = 0
        if yaw_control:
            obs_extra += 1
        if panoramic:
            obs_extra += n_panoramic_sectors
        if args.no_odom:
            dim_obs = 7 + obs_extra
        else:
            dim_obs = 10 + obs_extra  # 7 + 3 (local_v) + obs_extra
        
        if yaw_control:
            self.model = Model_bigger_yaw(dim_obs=dim_obs, dim_action=6).to(self.device)
            print(f"[YawControl] 已启用模型自主偏航控制, dim_obs={dim_obs}, "
                  f"coef_yaw_explore={getattr(args, 'coef_yaw_explore', 0.5)}, "
                  f"yaw_penalty_start_deg={getattr(args, 'yaw_penalty_start_deg', 29.0)}")
        else:
            self.model = Model_bigger(dim_obs=dim_obs, dim_action=6).to(self.device)
        
        if panoramic:
            print(f"[Panoramic] 已启用全向障碍物感知, {n_panoramic_sectors} 扇区, "
                  f"最大探测 {getattr(args, 'panoramic_max_range', 8.0)}m, dim_obs={dim_obs}")
        
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
            coef_stall=getattr(args, 'coef_stall', 0.0),
            coef_yaw_explore=getattr(args, 'coef_yaw_explore', 0.5) if getattr(args, 'enable_yaw_control', False) else 0.0,
            coef_progress=getattr(args, 'coef_progress', 0.0),
            yaw_penalty_start_rad=math.radians(getattr(args, 'yaw_penalty_start_deg', 29.0)),
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

        # Best AR 追踪（用于保存最佳策略 checkpoint）
        # AR = success_rate × avg_speed，综合衡量"安全且高效"的行为质量
        self.best_ar = -1.0
        self.best_ar_iter = -1
        self.ar_ema = 0.0           # 指数移动平均，减少单次波动
        self.ar_ema_alpha = 0.05    # EMA 平滑系数
        
    def _load_model(self, path):
        """加载模型，支持跨架构热启动（如 Model_bigger → Model_bigger_yaw）"""
        state_dict = torch.load(path, map_location=self.device)
        
        # 处理维度不匹配的层（如 v_proj 从 dim_obs=10 → 11）
        model_state = self.model.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                old_shape = state_dict[key].shape
                new_shape = model_state[key].shape
                print(f"[HotStart] {key}: {old_shape} → {new_shape}, 零填充新增维度")
                new_param = model_state[key].clone()  # 保留当前模型的初始化值
                # 计算可以复制的公共切片
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                new_param[slices] = state_dict[key][slices]
                state_dict[key] = new_param
        
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
        运行一个完整的 episode 并返回损失。

        单次 episode 流程:
            1. 重置环境和模型状态
            2. 可选：随机场景生成 + 安全出生点采样
            3. 采样目标点和随机化参数 (max_speed, thr_est_error, cam_pitch)
            4. timesteps 次循环: render → 构建观测 → 模型推理 → 物理步进
            5. 堆叠历史记录，调用 DroneLoss 计算总损失

        Args:
            iteration: 当前训练迭代编号，用于判断是否保存可视化帧

        Returns:
            loss: 标量总损失 (torch.Tensor)，带梯度
            metrics: 各项指标字典 (success_rate, avg_speed, ar 等)
            debug_out: 调试用数据元组 (p_history, v_history, act_buffer, vid)
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
        vec_to_pt_history = []
        v_preds = []
        yaw_history = []  # 偏航偏移累积历史 (仅 yaw_control 模式)
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
        max_speed = 0.75 + 3 * torch.rand((B, 1), device=self.device)  # 随机范围 [0.75, 3.75) m/s
        # max_speed = torch.full((B, 1), 6.0, device=self.device)  # 固定最大速度 6.0 m/s
        
        # 推力估计误差 (模拟真实无人机的推力不确定性)
        thr_est_error = 1.0 + 0.01 * torch.randn((B, 1), device=self.device)
        
        # Per-sample 相机俯仰角随机化 (参考项目在 reset 中一次性设定整个 episode)
        # 模拟每架无人机相机安装角的个体差异，episode 内保持不变
        cam_pitch = args.cam_angle + torch.randn(B, device=self.device)
        
        # 偏航控制状态
        yaw_control = getattr(args, 'enable_yaw_control', False)
        yaw_offset = torch.zeros(B, device=self.device)  # 累积偏航偏移 (弧度)
        
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
            # 跳帧优化: 深度图不参与梯度计算, 相邻帧变化很小, 可隔帧渲染节省开销
            if t % args.render_interval == 0:
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
            
            # 偏航控制：将目标向量按 yaw_offset 旋转
            # 旋转后无人机机头朝向旋转后的方向，而非直接朝向目标
            if yaw_control:
                cos_y = torch.cos(yaw_offset)  # (B,)
                sin_y = torch.sin(yaw_offset)
                # 绕 Z 轴旋转 target_v_raw 的 XY 分量
                tx, ty = target_v_raw[:, 0], target_v_raw[:, 1]
                rotated_x = cos_y * tx - sin_y * ty
                rotated_y = sin_y * tx + cos_y * ty
                heading_vector = torch.stack([rotated_x, rotated_y, target_v_raw[:, 2]], dim=-1)
            else:
                heading_vector = target_v_raw
            
            # 执行动作 (使用延迟缓冲中的动作)
            # 关键：使用 act_buffer[t] 而不是取模，参考项目的实现方式
            self.env.step(act_cmd=act_buffer[t], 
                         target_pos_vector=heading_vector, 
                         dt=current_dt)
            
            # 计算局部坐标系
            R_local = self._compute_local_R()
            
            # 计算目标速度向量
            target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
            target_v_unit = target_v_raw / (target_v_norm + 1e-6)
            target_v = target_v_unit * torch.minimum(target_v_norm, max_speed)
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
            if yaw_control:
                state_parts.append(yaw_offset[:, None])  # 当前偏航偏移 [1]
            if getattr(args, 'enable_panoramic', False):
                # 全向障碍物距离扫描 (body-frame, 不需要梯度)
                panoramic_raw = self.env.get_panoramic_clearance(
                    n_sectors=getattr(args, 'n_panoramic_sectors', 8),
                    max_range=getattr(args, 'panoramic_max_range', 8.0),
                )  # (B, n_sectors) 单位: 米
                # 归一化为逆距离特征: 近→高值, 远→低值, 与深度图编码一致
                panoramic_max = getattr(args, 'panoramic_max_range', 8.0)
                panoramic_feat = 1.0 / panoramic_raw.clamp(min=0.3) - 1.0 / panoramic_max
                panoramic_feat = panoramic_feat + torch.randn_like(panoramic_feat) * 0.02
                state_parts.append(panoramic_feat)  # [n_sectors]
            state = torch.cat(state_parts, dim=-1)
            
            # 深度图预处理
            # 空像素判定: PyTorch3D 背景 zbuf=-1 和超出探测距离的远处物体都视为「空」
            bg_mask = (depth < 0) | (depth > args.depth_max)
            x = depth.clamp(args.depth_min, args.depth_max)
            x = 3.0 / x - 0.6  # 逆距离变换: 近→高值(9.4@0.3m), 远→低值(0.0@5m)
            x[bg_mask] = 0.0    # 空像素设为 0（无障碍物）
            x = x + torch.randn_like(x) * 0.02  # 添加噪声
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            # 可选: stride=1 max_pool 膨胀障碍物，类似 C-space expansion。
            # 默认禁用；仅在擦碰问题严重时通过 --depth_dilate 5 开启。
            # if args.depth_dilate > 1:
            #     pad = args.depth_dilate // 2
            #     x = F.max_pool2d(x, kernel_size=args.depth_dilate, stride=1, padding=pad)

            # 模型推理
            act_raw, yaw_rate_raw, h = self.model(x, state, h)
            
            # 偏航控制：累积 yaw_rate 到 yaw_offset
            if yaw_control and yaw_rate_raw is not None:
                # tanh 限制 yaw_rate 到 [-1, 1] rad/s，防止极端旋转
                yaw_rate = torch.tanh(yaw_rate_raw) * 1.0  # 最大 ±1 rad/s ≈ ±57°/s
                yaw_offset = yaw_offset + yaw_rate * current_dt
                # 限制累积偏航到 [-π, π]，防止无限旋转
                yaw_offset = torch.remainder(yaw_offset + math.pi, 2 * math.pi) - math.pi
                yaw_history.append(yaw_offset)
            
            # Truncated BPTT: 每 30 步截断 GRU 梯度，避免 150 步全序列反向传播占满显存
            if t > 0 and t % 30 == 0:
                h = h.detach()
            
            act_reshaped = act_raw.reshape(B, 3, 2)  # 关键修复！直接 reshape 为 (B, 3, 2)
            act_world = R_local @ act_reshaped  # 转换到世界坐标系 (B, 3, 2)
            a_pred, v_pred = act_world.unbind(-1)  # 分离加速度和速度预测
            
            v_preds.append(v_pred)
            
            act = (a_pred - v_pred - self.g_std) * thr_est_error + self.g_std
        
            act_buffer.append(act)
            
            v_history.append(self.env.v)
            target_v_history.append(target_v)

        p_history = torch.stack(p_history)          # (T, B, 3)
        v_history = torch.stack(v_history)          # (T, B, 3)
        target_v_history = torch.stack(target_v_history)  # (T, B, 3)
        vec_to_pt_history = torch.stack(vec_to_pt_history)  # (T, B, 3)
        v_preds = torch.stack(v_preds)              # (T, B, 3)
        act_buffer_stacked = torch.stack(act_buffer)  # (T + lag + 1, B, 3)
        yaw_history_stacked = torch.stack(yaw_history) if yaw_history else None  # (T, B) or None
        
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
            yaw_history=yaw_history_stacked,
            p_target=p_target
        )
        
        # 计算额外指标
        with torch.no_grad():
            distance = torch.norm(vec_to_pt_history, 2, -1) - self.env.margin
            speed_history = v_history.norm(2, -1)
            avg_speed = speed_history.mean(0)
            if distance.dim() == 3:
                no_collision = torch.all(distance > 0, dim=(0, 1))
            else:
                no_collision = torch.all(distance > 0, dim=0)

            dist_to_target = torch.norm(p_target.unsqueeze(0) - p_history, 2, -1)
            reached = torch.any(dist_to_target <= args.arrival_threshold, dim=0)
            success = no_collision & reached
            success_rate = success.float().mean()
            
            metrics['success_rate'] = success_rate
            metrics['no_collision_rate'] = no_collision.float().mean()
            metrics['reach_rate'] = reached.float().mean()
            metrics['avg_speed'] = avg_speed.mean()
            metrics['max_speed'] = speed_history.max(0).values.mean()
            metrics['ar'] = (success.float() * avg_speed).mean()  # 成功率 × 平均速度
        
        # 非保存迭代时 detach debug_data，避免计算图被引用残留到下一轮
        debug_out = (
            p_history.detach(), v_history.detach(),
            act_buffer_stacked.detach(), vid
        )
        return loss, metrics, debug_out
    
    def train(self):
        """
        主训练循环。

        每次迭代调用 run_episode() 获取损失，执行反向传播和参数更新。
        包含 OOM 容错、NaN 检测、定期保存、Best AR 追踪等机制。
        训练结束后保存 checkpoint_final.pth 并打印 Best AR 信息。
        """
        args = self.args
        
        pbar = tqdm(range(args.num_iters), ncols=160, bar_format='{l_bar}{bar:20}{r_bar}')
        
        for i in pbar:
            try:
                # 运行一个 episode
                loss, metrics, debug_data = self.run_episode(i)
                
                # 检查 NaN
                if torch.isnan(loss):
                    print("Loss is NaN, exiting...")
                    break
                
                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.scheduler.step()
            except torch.cuda.OutOfMemoryError:
                # OOM 时 Python traceback 会持有 run_episode 内所有局部变量的引用
                # （包括计算图、历史 tensor 等），必须先 gc.collect() 打断引用链
                self.optimizer.zero_grad(set_to_none=True)  # 释放残留梯度显存
                gc.collect()
                torch.cuda.empty_cache()
                print(f"\n[OOM] iter {i}: episode OOM, skipping")
                continue
            
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
            
            # ---- Best AR checkpoint（独立保存，不覆盖常规 checkpoint）----
            current_ar = float(metrics.get('ar', 0.0))
            self.ar_ema = self.ar_ema_alpha * current_ar + (1 - self.ar_ema_alpha) * self.ar_ema
            # 前 200 步预热：EMA 还不稳定，跳过
            if i >= 200 and self.ar_ema > self.best_ar:
                self.best_ar = self.ar_ema
                self.best_ar_iter = i + 1
                best_path = os.path.join(args.save_dir, 'best_ar.pth')
                torch.save(self.model.state_dict(), best_path)
                # 同时记录到 TensorBoard
                self.writer.add_scalar('best_ar', self.best_ar, i + 1)
            
            if (i + 1) % 25 == 0:
                for k, v in self.scaler_q.items():
                    self.writer.add_scalar(k, sum(v) / len(v), i + 1)
                self.scaler_q.clear()
        
        # 保存最终模型
        final_path = os.path.join(args.save_dir, 'checkpoint_final.pth')
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")
        if self.best_ar_iter > 0:
            print(f"Best AR model: best_ar.pth (AR={self.best_ar:.4f} @ iter {self.best_ar_iter})")
        
        self.monitor.close()
        self.writer.close()
    
    def _log_figures(self, iteration, debug_data):
        """
        记录可视化图表到 TensorBoard。

        为第 5 个 batch 样本绘制位置、速度、动作的时序曲线，
        用于监控训练过程中的飞行行为变化。

        Args:
            iteration: 当前迭代编号
            debug_data: run_episode() 返回的调试数据元组
        """
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
