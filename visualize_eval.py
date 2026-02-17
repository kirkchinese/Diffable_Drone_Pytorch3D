"""
无人机可微分仿真 —— 可视化验证程序

功能：
  1. 加载训练好的模型 checkpoint
  2. 在与训练相同的仿真环境中运行推理（无梯度）
  3. 生成丰富的可视化输出：
     - 3D 飞行轨迹图（俯视 + 侧视）
     - 无人机第一人称 RGB / Depth 视频（MP4 或帧序列）
     - 速度、距离、加速度等时序曲线
     - 碰撞检测统计
  4. 支持随机场景生成，验证泛化能力

用法示例：
  python visualize_eval.py \
      --checkpoint ./checkpoints/odom_run4_Bigger/checkpoint_final.pth \
      --num_episodes 4 --timesteps 200 \
      --random_scene --output_dir ./viz_results

作者: Copilot
"""

import os
import math
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 尝试配置中文字体，失败则使用英文
_USE_CN = False
try:
    from matplotlib.font_manager import FontProperties, fontManager
    _cn_fonts = [f.name for f in fontManager.ttflist
                 if any(k in f.name.lower() for k in ('noto sans cjk', 'wqy', 'simhei', 'simsun', 'droid sans'))]
    if _cn_fonts:
        plt.rcParams['font.sans-serif'] = [_cn_fonts[0]] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False
        _USE_CN = True
except Exception:
    pass

def _L(cn: str, en: str) -> str:
    """根据字体可用性返回中文或英文标签"""
    return cn if _USE_CN else en

# 可选：视频输出
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from drone_env import DroneSimulator
from model import Model, Model_bigger, Model_bigger_yaw
from scene_generator import SceneGenerator, obj_to_ros, sample_cross_map_spawn_target


# ================================================================
# 参数解析
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='无人机飞行可视化验证')

    # 核心
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径 (.pth)')
    parser.add_argument('--output_dir', type=str, default='./viz_results',
                        help='可视化输出目录')

    # 模拟参数
    parser.add_argument('--num_episodes', type=int, default=4,
                        help='要运行的 episode 数量')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='每个 episode 的仿真步数')
    parser.add_argument('--arrival_threshold', type=float, default=1.0,
                        help='判定“抵达目标”的距离阈值 (米)。成功定义为：全程无碰撞且任意时刻进入该阈值')
    parser.add_argument('--ctl_dt', type=float, default=1/15,
                        help='控制时间步长 (秒)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='每个 episode 的批量大小 (每个 episode 并行仿真的无人机数)')

    # 环境参数 - 渲染（与训练保持一致）
    parser.add_argument('--cam_angle', type=float, default=10,
                        help='相机俯仰角 (度)')
    parser.add_argument('--image_height', type=int, default=48,
                        help='渲染图像高度')
    parser.add_argument('--image_width', type=int, default=64,
                        help='渲染图像宽度')
    parser.add_argument('--hfov', type=float, default=90.0,
                        help='模型相机水平视场角 (度)，需与训练时一致，默认90°')
    parser.add_argument('--mesh_path', type=str, default='./data/sample/sample.obj',
                        help='障碍物网格路径 (仅在非随机场景时使用)')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='障碍物点云采样数')
    parser.add_argument('--subdivide_times', type=int, default=0,
                        help='网格细分次数')
    parser.add_argument('--depth_min', type=float, default=0.3,
                        help='深度图近截断距离 (米)，同时也是渲染器近平面裁剪值')
    parser.add_argument('--depth_max', type=float, default=3.0,
                        help='深度图远截断距离 (米)')

    # 高分辨率渲染选项（用于可视化输出，不影响模型输入）
    parser.add_argument('--viz_height', type=int, default=480,
                        help='可视化输出 RGB/Depth 的高度')
    parser.add_argument('--viz_width', type=int, default=640,
                        help='可视化输出 RGB/Depth 的宽度')
    parser.add_argument('--viz_fov', type=float, default=90.0,
                        help='可视化渲染水平视场角 (度)，默认90度广角')

    # 场景随机化
    parser.add_argument('--random_scene', action='store_true', default=False,
                        help='启用随机场景生成')
    parser.add_argument('--num_obstacles_min', type=int, default=50,
                        help='最少障碍物数')
    parser.add_argument('--num_obstacles_max', type=int, default=80,
                        help='最多障碍物数')
    parser.add_argument('--obstacle_scale_min', type=float, default=0.3,
                        help='障碍物最小缩放')
    parser.add_argument('--obstacle_scale_max', type=float, default=1.5,
                        help='障碍物最大缩放')
    parser.add_argument('--arena_range', type=float, default=6.0,
                        help='场景 X/Y 范围')
    parser.add_argument('--ground_ratio', type=float, default=0.6,
                        help='接地物体比例 (0~1)')
    parser.add_argument('--cluster_ratio', type=float, default=0.3,
                        help='簇生物体比例 (0~1)')
    parser.add_argument('--cluster_spread', type=float, default=1.5,
                        help='簇生物体最大水平偏移 (m)')
    parser.add_argument('--safe_clearance', type=float, default=1.0,
                        help='安全出生点到障碍物的最小距离')
    parser.add_argument('--force_cross_map', action='store_true', default=False,
                        help='强制出生/目标点在场景对向两侧')
    parser.add_argument('--spawn_z_max', type=float, default=1.5,
                        help='出生/目标点最大高度')

    # 无人机物理（与训练保持一致）
    parser.add_argument('--init_p_range', type=float, default=6.0,
                        help='初始位置范围')
    parser.add_argument('--margin_min', type=float, default=0.3,
                        help='无人机安全半径最小值')
    parser.add_argument('--margin_max', type=float, default=0.7,
                        help='无人机安全半径最大值')
    parser.add_argument('--noise_std', type=float, default=0.04,
                        help='环境扰动噪声标准差')
    parser.add_argument('--grad_decay', type=float, default=0.4,
                        help='梯度衰减系数')
    parser.add_argument('--yaw_inertia', type=float, default=5.0)
    parser.add_argument('--yaw_ctl_delay', type=float, default=12.0)
    parser.add_argument('--pitch_ctl_delay', type=float, default=12.0)
    parser.add_argument('--airmode_coef', type=float, default=0.5)

    # 模型参数
    parser.add_argument('--no_odom', action='store_true', default=False,
                        help='不使用里程计速度作为输入')
    parser.add_argument('--enable_yaw_control', action='store_true', default=False,
                        help='启用模型自主偏航控制 (需与训练时的 --enable_yaw_control 一致)')
    parser.add_argument('--enable_guidance', action='store_true', default=False,
                        help='启用基于深度图的动态引导方向 (需与训练时一致)')
    parser.add_argument('--guidance_steer_gain', type=float, default=2.0,
                        help='引导转向增益系数')
    parser.add_argument('--guidance_gate_threshold', type=float, default=0.35,
                        help='引导门控阈值')
    parser.add_argument('--predicted_guidance', action='store_true', default=False,
                        help='使用模型自己预测的偏转角代替深度图计算的引导方向。\n'
                             '模型需要在训练时使用 --enable_guidance + --coef_steer_pred > 0 学习过偏转预测。\n'
                             '启用后推理时不再需要深度图引导计算，完全用模型预测值')

    # 目标点设置
    parser.add_argument('--max_speed', type=float, default=2.5,
                        help='目标最大速度 (m/s)，固定值用于评估一致性')

    # 输出控制
    parser.add_argument('--save_video', action='store_true', default=True,
                        help='保存 RGB/Depth 视频 (需要 imageio)')
    parser.add_argument('--save_frames', action='store_true', default=False,
                        help='保存每帧 RGB/Depth 图像')
    parser.add_argument('--fps', type=int, default=15, help='视频帧率')
    parser.add_argument('--no_video', action='store_true', default=False,
                        help='禁用视频输出')

    # 硬件
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    return parser.parse_args()


# ================================================================
# 可视化渲染器（高分辨率，仅用于输出可视化图像）
# ================================================================

class HighResRenderer:
    """
    广角高分辨率渲染器：复用环境的网格和光照设置，
    但使用更高分辨率 + 广角 FOV 输出可视化用 RGB / Depth。

    这里犯了个错误，训练时模型使用极窄 FOV (焦距500 / 48x64 → ~7°)，这是不正常的，
    对可视化来说太窄了——用户需要看到周围环境来验证飞行行为。
    因此可视化渲染器使用独立的广角 FOV (默认 ~90°)。
    """
    def __init__(self, env_renderer, viz_size=(480, 640), hfov_deg=90.0):
        from pytorch3d.renderer import (
            RasterizationSettings, MeshRasterizer,
            SoftPhongShader, PerspectiveCameras, PointLights
        )
        self.device = env_renderer.device
        self.viz_H, self.viz_W = viz_size
        self.env_renderer = env_renderer

        # 根据目标水平 FOV 计算焦距: f = (W/2) / tan(hfov/2)
        hfov_rad = math.radians(hfov_deg)
        focal = (self.viz_W / 2.0) / math.tan(hfov_rad / 2.0)
        actual_hfov = 2 * math.degrees(math.atan(self.viz_W / 2.0 / focal))
        actual_vfov = 2 * math.degrees(math.atan(self.viz_H / 2.0 / focal))
        print(f"[HighResRenderer] {self.viz_W}x{self.viz_H}, "
              f"focal={focal:.1f}, FOV={actual_hfov:.0f}°x{actual_vfov:.0f}°")

        self.focal_length = torch.tensor([[focal, focal]], dtype=torch.float32, device=self.device)
        self.principal_point = torch.tensor(
            [[self.viz_W / 2.0, self.viz_H / 2.0]], dtype=torch.float32, device=self.device
        )
        self.image_size_tensor = torch.tensor(
            [[self.viz_H, self.viz_W]], dtype=torch.float32, device=self.device
        )

        # 高分辨率下使用 coarse-to-fine 但增大 bin 容量，
        # 避免随机场景面片数多时 "Bin size was too small" 溢出
        self.raster_settings = RasterizationSettings(
            image_size=(self.viz_H, self.viz_W),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            z_clip_value=0.3,  # 与 depth_min 默认值对齐，避免 clip_faces OOM
            max_faces_per_bin=50000,
        )
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, lights=env_renderer.lights)

    @torch.no_grad()
    def render(self, R_view, T_view):
        """渲染广角高分辨率 RGB + Depth"""
        from pytorch3d.renderer import PerspectiveCameras
        cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            image_size=self.image_size_tensor,
            in_ndc=False,
            R=R_view, T=T_view,
            device=self.device,
        )
        mesh = self.env_renderer.mesh.extend(len(cameras))
        fragments = self.rasterizer(mesh, cameras=cameras)
        images = self.shader(fragments, mesh, cameras=cameras)
        rgb = images[..., :3]
        depth = fragments.zbuf[..., 0]
        return rgb, depth


# ================================================================
# 推理运行器
# ================================================================

class EvalRunner:
    """可视化评估运行器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"[EvalRunner] 设备: {self.device}")

        self.ctl_dt = args.ctl_dt
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=self.device)

        # ---------- 场景生成器 ----------
        self.scene_generator = None
        if args.random_scene:
            self.scene_generator = SceneGenerator(
                device=self.device,
                arena_range=args.arena_range,
                num_obstacles_range=(args.num_obstacles_min, args.num_obstacles_max),
                obstacle_scale_range=(args.obstacle_scale_min, args.obstacle_scale_max),
                ground_ratio=args.ground_ratio,
                cluster_ratio=args.cluster_ratio,
                cluster_spread=args.cluster_spread,
            )
            print(f"[SceneGenerator] 随机场景: 障碍物 {args.num_obstacles_min}–{args.num_obstacles_max}, "
                  f"接地率: {args.ground_ratio:.0%}, 簇生率: {args.cluster_ratio:.0%}")

        # ---------- 根据 FOV 计算焦距 ----------
        focal_length = (args.image_width / 2.0) / math.tan(math.radians(args.hfov / 2.0))
        hfov_actual = 2 * math.degrees(math.atan(args.image_width / 2.0 / focal_length))
        vfov_actual = 2 * math.degrees(math.atan(args.image_height / 2.0 / focal_length))
        print(f"[Camera] 模型相机 HFOV={hfov_actual:.0f}° VFOV={vfov_actual:.0f}° "
              f"focal={focal_length:.1f} image={args.image_width}x{args.image_height}")

        # ---------- 仿真环境 ----------
        self.env = DroneSimulator(
            batch_size=args.batch_size,
            dt=self.ctl_dt,
            mesh_path=args.mesh_path,
            image_size=(args.image_height, args.image_width),
            focal_length=focal_length,
            device=self.device,
            enable_airmode=True,
            noise_std=args.noise_std,
            grad_decay=args.grad_decay,
            yaw_inertia=args.yaw_inertia,
            yaw_ctl_delay=args.yaw_ctl_delay,
            pitch_ctl_delay=args.pitch_ctl_delay,
            airmode_coef=args.airmode_coef,
            init_p_range=args.init_p_range,
            init_margin_range=(args.margin_min, args.margin_max),
            num_samples=args.num_samples,
            subdivide_times=args.subdivide_times,
            z_clip_value=args.depth_min,
            enable_random_scene=args.random_scene,
            scene_generator=self.scene_generator,
            safe_spawn_clearance=args.safe_clearance,
        )

        # ---------- 高分辨率广角渲染器 ----------
        self.hires_renderer = HighResRenderer(
            self.env.renderer,
            viz_size=(args.viz_height, args.viz_width),
            hfov_deg=args.viz_fov,
        )

        # ---------- 模型 (自动检测是否支持偏航控制) ----------
        yaw_control_flag = getattr(args, 'enable_yaw_control', False)
        # 从 checkpoint 中自动检测是否为 yaw 模型
        ckpt_state = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
        ckpt_has_yaw = any('fc_yaw' in k for k in ckpt_state.keys())
        if yaw_control_flag and not ckpt_has_yaw:
            print(f"[WARNING] --enable_yaw_control 已指定, 但 checkpoint 中未检测到 fc_yaw 层")
            print(f"[WARNING] 这是一个旧版模型, 将自动禁用偏航控制可视化")
            yaw_control_flag = False
        elif not yaw_control_flag and ckpt_has_yaw:
            print(f"[AutoDetect] checkpoint 中检测到 fc_yaw 层, 自动启用偏航控制")
            yaw_control_flag = True
        self.yaw_control = yaw_control_flag

        obs_extra = 1 if self.yaw_control else 0
        dim_obs = (7 if args.no_odom else 10) + obs_extra
        if self.yaw_control:
            self.model = Model_bigger_yaw(dim_obs=dim_obs, dim_action=6).to(self.device)
            print(f"[YawControl] 已启用模型自主偏航控制, dim_obs={dim_obs}")
        else:
            self.model = Model_bigger(dim_obs=dim_obs, dim_action=6).to(self.device)
            if not ckpt_has_yaw:
                print(f"[Info] 旧版模型 (无偏航控制头), 偏航偏移量将不会显示")
        self._load_checkpoint(args.checkpoint)
        self.model.eval()

        # ---------- 输出目录 ----------
        os.makedirs(args.output_dir, exist_ok=True)

    def _load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        # 处理维度不匹配的层（跨架构热启动）
        model_state = self.model.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                old_shape = state_dict[key].shape
                new_shape = model_state[key].shape
                print(f"[HotStart] {key}: {old_shape} → {new_shape}, 零填充新增维度")
                new_param = model_state[key].clone()
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                new_param[slices] = state_dict[key][slices]
                state_dict[key] = new_param
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")
        print(f"[Model] 已加载: {path}")

    # ---- 局部坐标系 (复用训练逻辑) ----
    def _compute_local_R(self):
        fwd = self.env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd_norm = torch.norm(fwd, p=2, dim=-1, keepdim=True)
        fwd = torch.where(fwd_norm > 1e-6, fwd / (fwd_norm + 1e-8),
                          torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(fwd))
        R = torch.stack([fwd, torch.linalg.cross(up, fwd), up], -1)
        return R

    def _compute_guidance_direction(self, depth, target_v_raw,
                                    steer_gain=2.0, gate_threshold=0.35):
        """可通行性梯度引导：沿空旷度增加的方向偏转目标航向。与 train.py 中的实现完全一致。"""
        B, H, W = depth.shape
        args = self.args

        bg_mask = (depth < 0) | (depth > args.depth_max)
        d = depth.clone()
        d[bg_mask] = args.depth_max
        d = d.clamp(args.depth_min, args.depth_max)
        col_clear = d.mean(dim=1) / args.depth_max  # (B, W)

        w = torch.linspace(1.0, -1.0, W, device=depth.device)
        grad_c = (col_clear * w.unsqueeze(0)).sum(dim=1) / (W / 2)

        cs, ce = W // 3, W * 2 // 3
        blockage = 1.0 - col_clear[:, cs:ce].mean(dim=1)
        gate = torch.sigmoid(10.0 * (blockage - gate_threshold))

        hfov_rad = math.radians(args.hfov)
        steer = (steer_gain * gate * grad_c).clamp(-hfov_rad * 0.8, hfov_rad * 0.8)

        cos_s, sin_s = torch.cos(steer), torch.sin(steer)
        tx, ty = target_v_raw[:, 0], target_v_raw[:, 1]
        guided = torch.stack([cos_s * tx - sin_s * ty,
                            sin_s * tx + cos_s * ty,
                            target_v_raw[:, 2]], dim=-1)
        return guided, steer

    # ================================================================
    # 运行单个 episode
    # ================================================================
    @torch.no_grad()
    def run_episode(self, episode_idx):
        """
        运行一个无梯度推理 episode，记录完整轨迹与图像。

        Returns:
            data: dict 包含所有记录数据
        """
        args = self.args
        B = args.batch_size

        # ---- 重置 ----
        self.env.reset()
        self.model.reset()

        if self.env.enable_random_scene and self.env.scene_generator is not None:
            self.env.randomize_scene()
            # 随机场景后需要更新高分辨率渲染器的mesh缓存
            self.hires_renderer.env_renderer = self.env.renderer

        # 安全出生点 + 目标点
        spawn_z_max = getattr(args, 'spawn_z_max', 3.0)
        # 随机场景时自动启用跨地图模式（除非用户显式关闭），确保飞行路径穿越障碍区域
        use_cross_map = (getattr(args, 'force_cross_map', False) or self.env.enable_random_scene)
        # 采样范围必须与障碍物分布范围一致，否则安全采样会把点推到障碍物场外面
        spawn_arena = args.arena_range if self.env.enable_random_scene else args.init_p_range
        if use_cross_map and self.env.enable_random_scene:
            _, p_target = self.env.safe_reset_cross_map(
                arena_range=spawn_arena,
                z_range=(1.0, spawn_z_max),
            )
        else:
            self.env.safe_reset(
                arena_range=spawn_arena,
                z_range=(1.0, spawn_z_max),
            )
            # 安全目标点
            p_target = self.env.sample_safe_target(
                arena_range=spawn_arena,
                z_range=(1.0, spawn_z_max),
                min_distance=3.0,
                max_distance=spawn_arena * 1.2,
            )
        target_v_raw = p_target - self.env.p

        # 固定速度用于评估一致性
        max_speed = torch.full((B, 1), args.max_speed, device=self.device)
        thr_est_error = torch.ones((B, 1), device=self.device)  # 评估时不加推力噪声
        cam_pitch = torch.full((B,), args.cam_angle, device=self.device, dtype=torch.float32)

        # 偏航控制状态
        yaw_control = self.yaw_control
        yaw_offset = torch.zeros(B, device=self.device)

        # 动作延迟缓冲
        act_lag = 1
        act_buffer = [self.env.act_curr.clone() for _ in range(act_lag + 1)]
        h = None
        
        # 引导偏转预测: 推理时用上一步的模型预测值
        use_predicted = getattr(args, 'predicted_guidance', False)
        prev_steer_pred = torch.zeros(B, device=self.device)  # 第 0 步无偏转

        # ---- 记录容器 ----
        rec = {
            'p': [],              # (T, B, 3)  位置 (ROS)
            'v': [],              # (T, B, 3)  速度 (ROS)
            'a': [],              # (T, B, 3)  加速度 (ROS)
            'speed': [],          # (T, B)     速率
            'dist_to_obs': [],    # (T, B)     到最近障碍物距离
            'dist_to_target': [], # (T, B)     到目标点距离
            'target_v': [],       # (T, B, 3)  限速后目标速度
            'heading': [],        # (T, B, 3)  航向 (R[:,:,0])
            'attitude_z': [],     # (T, B, 3)  姿态 z 轴 (R[:,2])
            'depth_valid_pct': [],# (T, B)     模型深度图有效像素比例
            'rgb_frames': [],     # list of (B, H, W, 3) uint8 numpy
            'depth_frames': [],   # list of (B, H, W) float numpy  (广角可视化)
            'model_depth_frames': [], # list of (B, H, W) float numpy (模型输入)
            'collision': [],      # (T, B) bool
            'yaw_offset': [],     # (T, B)     偏航累积偏移 (弧度, 仅 yaw_control)
            'yaw_rate': [],       # (T, B)     偏航角速度 (rad/s, 仅 yaw_control)
            'margin': self.env.margin.cpu().numpy().copy(),  # (B,)
            'p_start': self.env.p.clone().cpu().numpy(),
            'p_target': p_target.clone().cpu().numpy(),
        }

        for t in tqdm(range(args.timesteps), desc=f'Episode {episode_idx}', leave=False):
            current_dt = self.ctl_dt  # 评估时使用固定步长

            # ---------- 模型输入深度图 (低分辨率，与训练一致) ----------
            _, depth_lo = self.env.render(
                camera_pitch=cam_pitch,
                return_tensor=True,
                return_rgb=False,
                return_depth=True,
                dt=current_dt,
            )

            # ---------- 高分辨率可视化渲染 ----------
            R_cam, T_cam = self.env.renderer.compute_view_matrix(
                p_ros=self.env.p,
                R_ros=self.env.R,
                camera_pitch_deg=cam_pitch,
                cam_offset_body=self.env.cam_offset_body,
            )
            rgb_hi, depth_hi = self.hires_renderer.render(R_cam, T_cam)

            # 转 numpy
            rgb_np = (rgb_hi.clamp(0, 1) * 255).byte().cpu().numpy()
            depth_np = depth_hi.cpu().numpy()
            rec['rgb_frames'].append(rgb_np)
            rec['depth_frames'].append(depth_np)

            # ---------- 记录状态 ----------
            rec['p'].append(self.env.p.cpu().numpy().copy())
            rec['v'].append(self.env.v.cpu().numpy().copy())
            rec['a'].append(self.env.a.cpu().numpy().copy())
            rec['speed'].append(self.env.v.norm(2, -1).cpu().numpy().copy())
            rec['heading'].append(self.env.R[:, :, 0].cpu().numpy().copy())
            rec['attitude_z'].append(self.env.R[:, 2].cpu().numpy().copy())
            dist = self.env.calc_min_distance()
            rec['dist_to_obs'].append(dist.cpu().numpy().copy())
            rec['collision'].append(
                (dist < self.env.margin).cpu().numpy().copy()
            )
            # 模型深度图有效像素比例
            depth_valid = (depth_lo > 0).float().mean(dim=(-1, -2)).cpu().numpy() * 100
            rec['depth_valid_pct'].append(depth_valid.copy())
            # 保存模型输入深度图 (低分辨率)
            rec['model_depth_frames'].append(depth_lo.cpu().numpy().copy())

            # ---------- 更新目标向量 ----------
            target_v_raw = p_target - self.env.p
            rec['dist_to_target'].append(
                target_v_raw.norm(2, -1).cpu().numpy().copy()
            )

            # ---------- 动态引导方向 ----------
            if use_predicted:
                # 自蒸馏模式: 使用上一步模型预测的偏转角旋转 target_v_raw
                cos_s = torch.cos(prev_steer_pred)
                sin_s = torch.sin(prev_steer_pred)
                tx, ty = target_v_raw[:, 0], target_v_raw[:, 1]
                guided_target_v = torch.stack([
                    cos_s * tx - sin_s * ty,
                    sin_s * tx + cos_s * ty,
                    target_v_raw[:, 2]], dim=-1)
            elif getattr(args, 'enable_guidance', False):
                guided_target_v, _ = self._compute_guidance_direction(
                    depth_lo, target_v_raw,
                    steer_gain=getattr(args, 'guidance_steer_gain', 2.0),
                    gate_threshold=getattr(args, 'guidance_gate_threshold', 0.35),
                )
            else:
                guided_target_v = target_v_raw

            # ---------- 偏航控制: 旋转目标方向 ----------
            if yaw_control and yaw_offset is not None:
                cos_y = torch.cos(yaw_offset)
                sin_y = torch.sin(yaw_offset)
                tx, ty = guided_target_v[:, 0], guided_target_v[:, 1]
                rotated_x = cos_y * tx - sin_y * ty
                rotated_y = sin_y * tx + cos_y * ty
                heading_vector = torch.stack([rotated_x, rotated_y, guided_target_v[:, 2]], dim=-1)
            else:
                heading_vector = guided_target_v

            # ---------- 执行动作 ----------
            self.env.step(act_cmd=act_buffer[t], target_pos_vector=heading_vector, dt=current_dt)

            # ---------- 模型推理 ----------
            R_local = self._compute_local_R()

            # target_v 方向与训练保持一致: 引导开启时用引导方向, 否则用原始方向
            if use_predicted or getattr(args, 'enable_guidance', False):
                effective_target_dir = guided_target_v
            else:
                effective_target_dir = target_v_raw
            target_v_norm = torch.norm(effective_target_dir, p=2, dim=-1, keepdim=True)
            target_v_unit = effective_target_dir / (target_v_norm + 1e-6)
            dist_to_goal = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True)
            target_v = target_v_unit * torch.minimum(dist_to_goal, max_speed)
            target_v_local = torch.squeeze(target_v[:, None] @ R_local, 1)
            local_v = torch.squeeze(self.env.v[:, None] @ R_local, 1)

            state_parts = [
                target_v_local,
                self.env.R[:, 2],
                self.env.margin[:, None],
            ]
            if not args.no_odom:
                state_parts.insert(0, local_v)
            if yaw_control and yaw_offset is not None:
                state_parts.append(yaw_offset[:, None])
            state = torch.cat(state_parts, dim=-1)

            # 深度图预处理（与 train.py 完全一致）
            bg_mask = (depth_lo < 0) | (depth_lo > args.depth_max)
            x = depth_lo.clamp(args.depth_min, args.depth_max)
            x = 3.0 / x - 0.6
            x[bg_mask] = 0.0
            x = x.unsqueeze(1)  # (B, 1, H, W)

            act_raw, yaw_rate_raw, h, steer_pred = self.model(x, state, h)
            
            # 保存当前预测的偏转角供下一步使用 (自回归)
            if use_predicted:
                prev_steer_pred = steer_pred

            # 偏航控制: 累积 yaw_rate
            if yaw_control and yaw_rate_raw is not None:
                yaw_rate = torch.tanh(yaw_rate_raw) * 1.0
                yaw_offset = yaw_offset + yaw_rate * current_dt
                yaw_offset = torch.remainder(yaw_offset + math.pi, 2 * math.pi) - math.pi
                rec['yaw_offset'].append(yaw_offset.cpu().numpy().copy())
                rec['yaw_rate'].append(yaw_rate.cpu().numpy().copy())
            else:
                rec['yaw_offset'].append(np.zeros(B))
                rec['yaw_rate'].append(np.zeros(B))

            act_reshaped = act_raw.reshape(B, 3, 2)
            act_world = R_local @ act_reshaped
            a_pred, v_pred = act_world.unbind(-1)

            act = (a_pred - v_pred - self.g_std) * thr_est_error + self.g_std
            act_buffer.append(act)

            rec['target_v'].append(target_v.cpu().numpy().copy())

        # 堆叠
        for key in ['p', 'v', 'a', 'speed', 'dist_to_obs', 'dist_to_target',
                    'target_v', 'heading', 'attitude_z', 'depth_valid_pct', 'collision',
                    'yaw_offset', 'yaw_rate']:
            rec[key] = np.stack(rec[key], axis=0)

        # 打印 episode 摘要
        for b in range(B):
            n_coll = int(rec['collision'][:, b].sum())
            avg_spd = float(rec['speed'][:, b].mean())
            min_d = float(rec['dist_to_obs'][:, b].min())
            final_d = float(rec['dist_to_target'][-1, b])
            reached = bool((rec['dist_to_target'][:, b] <= args.arrival_threshold).any())
            success = (n_coll == 0) and reached
            avg_depth = float(rec['depth_valid_pct'][:, b].mean())
            max_yaw = float(np.abs(rec['yaw_offset'][:, b]).max())
            print(f"  [Ep{episode_idx} B{b}] Collisions={n_coll}/{args.timesteps} | "
                  f"Avg Speed={avg_spd:.2f} m/s | Min Dist={min_d:.3f} m | "
                f"Final->Target={final_d:.2f} m | Reach={reached} | Success={success} | Avg Depth Valid={avg_depth:.1f}%"
                  f" | Max |yaw|={np.degrees(max_yaw):.1f}°")

        return rec

    # ================================================================
    # 绘制轨迹图
    # ================================================================
    def plot_trajectory(self, rec, episode_idx, sample_idx=0, save_dir=None):
        """生成 3D 轨迹 + 状态曲线的综合图"""
        if save_dir is None:
            save_dir = self.args.output_dir

        p = rec['p'][:, sample_idx]        # (T, 3)
        v = rec['v'][:, sample_idx]        # (T, 3)
        a = rec['a'][:, sample_idx]        # (T, 3)
        speed = rec['speed'][:, sample_idx]
        dist = rec['dist_to_obs'][:, sample_idx]
        d2t = rec['dist_to_target'][:, sample_idx]
        depth_pct = rec['depth_valid_pct'][:, sample_idx]
        collision = rec['collision'][:, sample_idx]
        p_start = rec['p_start'][sample_idx]
        p_target = rec['p_target'][sample_idx]
        T = p.shape[0]
        t_axis = np.arange(T) * self.ctl_dt

        fig = plt.figure(figsize=(24, 24))
        gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.30)

        # ---- 3D 轨迹 (大图) ----
        ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        ax3d.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, s=3, alpha=0.8)
        ax3d.scatter(*p_start, color='lime', s=120, marker='^', label='Start', zorder=5, edgecolors='k')
        ax3d.scatter(*p_target, color='red', s=120, marker='*', label='Target', zorder=5, edgecolors='k')
        # 连接起止点的虚线
        ax3d.plot([p_start[0], p_target[0]],
                  [p_start[1], p_target[1]],
                  [p_start[2], p_target[2]], 'r--', alpha=0.4, linewidth=1)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title(f'Episode {episode_idx} - {_L("3D 飞行轨迹", "3D Flight Trajectory")}', fontsize=14)
        ax3d.legend(fontsize=10)

        # ---- Top-down XY ----
        ax_xy = fig.add_subplot(gs[0, 2])
        ax_xy.plot(p[:, 0], p[:, 1], 'b-', linewidth=0.8, alpha=0.7)
        ax_xy.scatter(p[0, 0], p[0, 1], c='lime', s=80, marker='^', zorder=5, edgecolors='k')
        ax_xy.scatter(p_target[0], p_target[1], c='red', s=80, marker='*', zorder=5, edgecolors='k')
        coll_mask = collision.astype(bool)
        if coll_mask.any():
            ax_xy.scatter(p[coll_mask, 0], p[coll_mask, 1], c='red', s=20, marker='x',
                          label=_L('碰撞', 'Collision'), zorder=6)
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title(_L('俯视 XY', 'Top-down XY'))
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, alpha=0.3)

        # ---- Side XZ ----
        ax_xz = fig.add_subplot(gs[1, 2])
        ax_xz.plot(p[:, 0], p[:, 2], 'b-', linewidth=0.8, alpha=0.7)
        ax_xz.scatter(p[0, 0], p[0, 2], c='lime', s=80, marker='^', zorder=5, edgecolors='k')
        ax_xz.scatter(p_target[0], p_target[2], c='red', s=80, marker='*', zorder=5, edgecolors='k')
        if coll_mask.any():
            ax_xz.scatter(p[coll_mask, 0], p[coll_mask, 2], c='red', s=20, marker='x', zorder=6)
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title(_L('侧视 XZ', 'Side XZ'))
        ax_xz.grid(True, alpha=0.3)

        # ---- Speed ----
        ax_spd = fig.add_subplot(gs[2, 0])
        ax_spd.plot(t_axis, speed, 'b-', linewidth=1, label='|v|')
        ax_spd.plot(t_axis, v[:, 0], '--', linewidth=0.6, alpha=0.6, label='vx')
        ax_spd.plot(t_axis, v[:, 1], '--', linewidth=0.6, alpha=0.6, label='vy')
        ax_spd.plot(t_axis, v[:, 2], '--', linewidth=0.6, alpha=0.6, label='vz')
        ax_spd.axhline(y=self.args.max_speed, color='r', linestyle=':', alpha=0.5, label=f'max={self.args.max_speed}')
        ax_spd.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_spd.set_ylabel(_L('速度 (m/s)', 'Speed (m/s)'))
        ax_spd.set_title(_L('速度曲线', 'Speed Profile'))
        ax_spd.legend(fontsize=8, ncol=2)
        ax_spd.grid(True, alpha=0.3)

        # ---- Obstacle distance ----
        ax_dist = fig.add_subplot(gs[2, 1])
        margin = self.env.margin[sample_idx].item() if sample_idx < self.env.margin.shape[0] else 0.3
        ax_dist.plot(t_axis, dist, 'g-', linewidth=1, label=_L('最近距离', 'Min Dist'))
        ax_dist.axhline(y=margin, color='r', linestyle='--', alpha=0.7,
                         label=f'{_L("碰撞阈值", "Threshold")} ({margin:.2f}m)')
        ax_dist.fill_between(t_axis, 0, margin, alpha=0.1, color='red')
        if coll_mask.any():
            ax_dist.scatter(t_axis[coll_mask], dist[coll_mask], c='red', s=15, marker='x', zorder=5)
        ax_dist.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_dist.set_ylabel(_L('距离 (m)', 'Distance (m)'))
        ax_dist.set_title(_L('到最近障碍物距离', 'Min Obstacle Distance'))
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.3)

        # ---- Position components ----
        ax_pos = fig.add_subplot(gs[2, 2])
        ax_pos.plot(t_axis, p[:, 0], linewidth=1, label='x')
        ax_pos.plot(t_axis, p[:, 1], linewidth=1, label='y')
        ax_pos.plot(t_axis, p[:, 2], linewidth=1, label='z')
        for i, c in enumerate(['r', 'g', 'b']):
            ax_pos.axhline(y=p_target[i], color=c, linestyle=':', alpha=0.4)
        ax_pos.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_pos.set_ylabel(_L('位置 (m)', 'Position (m)'))
        ax_pos.set_title(_L('位置分量', 'Position Components'))
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True, alpha=0.3)

        # ---- Row 4: Distance to target ----
        ax_d2t = fig.add_subplot(gs[3, 0])
        ax_d2t.plot(t_axis, d2t, 'm-', linewidth=1.2, label=_L('到目标距离', 'To Target'))
        ax_d2t.axhline(y=0.5, color='g', linestyle=':', alpha=0.5, label='0.5 m')
        ax_d2t.fill_between(t_axis, 0, 0.5, alpha=0.05, color='green')
        ax_d2t.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_d2t.set_ylabel(_L('距离 (m)', 'Distance (m)'))
        ax_d2t.set_title(_L('到目标距离', 'Distance to Target'))
        ax_d2t.legend(fontsize=8)
        ax_d2t.grid(True, alpha=0.3)

        # ---- Acceleration ----
        ax_acc = fig.add_subplot(gs[3, 1])
        acc_mag = np.linalg.norm(a, axis=-1)
        ax_acc.plot(t_axis, acc_mag, 'r-', linewidth=1, label='|a|')
        ax_acc.plot(t_axis, a[:, 0], '--', linewidth=0.6, alpha=0.6, label='ax')
        ax_acc.plot(t_axis, a[:, 1], '--', linewidth=0.6, alpha=0.6, label='ay')
        ax_acc.plot(t_axis, a[:, 2], '--', linewidth=0.6, alpha=0.6, label='az')
        ax_acc.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_acc.set_ylabel(_L('加速度 (m/s²)', 'Accel (m/s²)'))
        ax_acc.set_title(_L('加速度曲线', 'Acceleration Profile'))
        ax_acc.legend(fontsize=8, ncol=2)
        ax_acc.grid(True, alpha=0.3)

        # ---- Depth coverage ----
        ax_dpct = fig.add_subplot(gs[3, 2])
        ax_dpct.plot(t_axis, depth_pct, 'c-', linewidth=1.2)
        ax_dpct.fill_between(t_axis, 0, depth_pct, alpha=0.2, color='cyan')
        ax_dpct.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_dpct.set_ylabel('%')
        ax_dpct.set_title(_L('深度图有效像素占比', 'Depth Valid Pixel %'))
        ax_dpct.set_ylim(bottom=0)
        ax_dpct.grid(True, alpha=0.3)

        # ---- Row 5: Yaw offset & rate ----
        yaw_off = rec['yaw_offset'][:, sample_idx]   # (T,) rad
        yaw_rt  = rec['yaw_rate'][:, sample_idx]     # (T,) rad/s
        has_yaw = np.any(np.abs(yaw_off) > 1e-6) or np.any(np.abs(yaw_rt) > 1e-6)

        ax_yaw = fig.add_subplot(gs[4, 0])
        ax_yaw.plot(t_axis, np.degrees(yaw_off), 'b-', linewidth=1.2, label=_L('偏航偏移', 'Yaw Offset'))
        ax_yaw.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        # 标注探索奖励区 (±29°)
        ax_yaw.axhline(y=28.6, color='g', linestyle='--', alpha=0.3, label='±α (29°)')
        ax_yaw.axhline(y=-28.6, color='g', linestyle='--', alpha=0.3)
        ax_yaw.fill_between(t_axis, -28.6, 28.6, alpha=0.05, color='green')
        ax_yaw.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_yaw.set_ylabel(_L('角度 (°)', 'Angle (°)'))
        ax_yaw.set_title(_L('偏航累积偏移', 'Yaw Offset') + (' [活跃]' if has_yaw else ' [未启用]'))
        ax_yaw.legend(fontsize=8)
        ax_yaw.grid(True, alpha=0.3)

        ax_yr = fig.add_subplot(gs[4, 1])
        ax_yr.plot(t_axis, np.degrees(yaw_rt), 'r-', linewidth=0.8, label=_L('偏航角速度', 'Yaw Rate'))
        ax_yr.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax_yr.set_xlabel(_L('时间 (s)', 'Time (s)'))
        ax_yr.set_ylabel(_L('角速度 (°/s)', 'Rate (°/s)'))
        ax_yr.set_title(_L('偏航角速度', 'Yaw Rate'))
        ax_yr.legend(fontsize=8)
        ax_yr.grid(True, alpha=0.3)

        # 偏航 vs 速度关系图（与损失函数设计对应）
        ax_yv = fig.add_subplot(gs[4, 2])
        ax_yv.scatter(speed, np.abs(np.degrees(yaw_off)), c=t_axis, cmap='viridis',
                      s=8, alpha=0.6)
        ax_yv.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5,
                       label=_L('速度门控阈值', 'Speed gate (1 m/s)'))
        ax_yv.set_xlabel(_L('速度 (m/s)', 'Speed (m/s)'))
        ax_yv.set_ylabel(_L('|偏航| (°)', '|Yaw| (°)'))
        ax_yv.set_title(_L('偏航 vs 速度', 'Yaw vs Speed'))
        ax_yv.legend(fontsize=8)
        ax_yv.grid(True, alpha=0.3)

        # ---- Summary ----
        n_collisions = int(coll_mask.sum())
        avg_speed_val = float(speed.mean())
        min_dist_val = float(dist.min())
        final_dist = float(d2t[-1])
        init_dist = float(np.linalg.norm(p_start - p_target))
        progress = max(0, (1 - final_dist / max(init_dist, 0.01))) * 100
        fig.suptitle(
            f'Episode {episode_idx} | '
            f'Coll: {n_collisions}/{T} | '
            f'Speed: {avg_speed_val:.2f} m/s | '
            f'Min Obs: {min_dist_val:.3f} m | '
            f'Target: {final_dist:.1f} m ({progress:.0f}%)',
            fontsize=13, y=0.99
        )

        path = os.path.join(save_dir, f'episode_{episode_idx:03d}_trajectory.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Saved] 轨迹图: {path}")
        return path

    # ================================================================
    # 保存视频
    # ================================================================
    def save_video(self, rec, episode_idx, sample_idx=0, save_dir=None):
        """将 RGB 和 Depth 帧保存为 MP4 视频"""
        if save_dir is None:
            save_dir = self.args.output_dir

        T = len(rec['rgb_frames'])
        fps = self.args.fps

        if not HAS_IMAGEIO:
            print("  [SKIP] 视频保存需要 imageio, 请 pip install imageio imageio-ffmpeg")
            return None, None

        # ---- RGB 视频 ----
        rgb_path = os.path.join(save_dir, f'episode_{episode_idx:03d}_rgb.mp4')
        writer = imageio.get_writer(rgb_path, fps=fps, codec='libx264', quality=8)
        for t in range(T):
            frame = rec['rgb_frames'][t][sample_idx]  # (H, W, 3) uint8
            writer.append_data(frame)
        writer.close()
        print(f"  [Saved] RGB 视频: {rgb_path}")

        # ---- Depth 可视化视频 ----
        depth_path = os.path.join(save_dir, f'episode_{episode_idx:03d}_depth.mp4')
        writer = imageio.get_writer(depth_path, fps=fps, codec='libx264', quality=8)
        for t in range(T):
            d = rec['depth_frames'][t][sample_idx]  # (H, W)
            # 归一化可视化
            valid = d > 0
            if valid.any():
                d_vis = d.copy()
                d_vis[~valid] = d_vis[valid].max()
                d_min, d_max = d_vis[valid].min(), d_vis[valid].max()
                if d_max > d_min:
                    d_norm = (d_vis - d_min) / (d_max - d_min)
                else:
                    d_norm = np.zeros_like(d_vis)
            else:
                d_norm = np.zeros_like(d)
            # 使用 viridis colormap
            d_color = plt.cm.viridis(d_norm)[..., :3]
            d_color = (d_color * 255).astype(np.uint8)
            writer.append_data(d_color)
        writer.close()
        print(f"  [Saved] Depth 视频: {depth_path}")

        return rgb_path, depth_path

    # ================================================================
    # 保存帧序列
    # ================================================================
    def save_frames(self, rec, episode_idx, sample_idx=0, save_dir=None):
        """保存每帧 RGB/Depth 为 PNG 图像"""
        if save_dir is None:
            save_dir = self.args.output_dir

        frame_dir = os.path.join(save_dir, f'episode_{episode_idx:03d}_frames')
        os.makedirs(frame_dir, exist_ok=True)

        T = len(rec['rgb_frames'])
        for t in range(T):
            # RGB
            rgb = rec['rgb_frames'][t][sample_idx]
            plt.imsave(os.path.join(frame_dir, f'rgb_{t:04d}.png'), rgb)
            # Depth
            d = rec['depth_frames'][t][sample_idx]
            fig_d, ax_d = plt.subplots(figsize=(6.4, 4.8))
            valid = d > 0
            d_vis = d.copy()
            if valid.any():
                d_vis[~valid] = np.nan
            im = ax_d.imshow(d_vis, cmap='viridis')
            plt.colorbar(im, ax=ax_d, label='Depth (m)')
            ax_d.set_title(f't={t}')
            ax_d.axis('off')
            fig_d.savefig(os.path.join(frame_dir, f'depth_{t:04d}.png'), dpi=100, bbox_inches='tight')
            plt.close(fig_d)

        print(f"  [Saved] 帧序列: {frame_dir}/ ({T} 帧)")

    # ================================================================
    # 综合面板帧（RGB + Depth + Info 合成）
    # ================================================================
    def save_composite_video(self, rec, episode_idx, sample_idx=0, save_dir=None):
        """生成 2x2 综合面板视频：RGB + Depth + 实时轨迹 + 详细状态"""
        if save_dir is None:
            save_dir = self.args.output_dir
        if not HAS_IMAGEIO:
            print("  [SKIP] 综合视频需要 imageio")
            return None

        T = len(rec['rgb_frames'])
        fps = self.args.fps
        comp_path = os.path.join(save_dir, f'episode_{episode_idx:03d}_composite.mp4')
        writer = imageio.get_writer(comp_path, fps=fps, codec='libx264', quality=8,
                                    macro_block_size=1)

        margin = rec['margin'][sample_idx]
        p_target = rec['p_target'][sample_idx]
        p_start = rec['p_start'][sample_idx]
        all_p = rec['p'][:, sample_idx]  # (T, 3)
        all_yaw_offset = rec['yaw_offset'][:, sample_idx]  # (T,) rad
        all_yaw_rate = rec['yaw_rate'][:, sample_idx]      # (T,) rad/s
        yaw_control_active = np.any(np.abs(all_yaw_offset) > 1e-6) or np.any(np.abs(all_yaw_rate) > 1e-6)

        for t in range(T):
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.20)

            # ---- Top-left: RGB + Yaw HUD overlay ----
            ax_rgb = fig.add_subplot(gs[0, 0])
            rgb = rec['rgb_frames'][t][sample_idx].copy()
            ax_rgb.imshow(rgb)
            # Yaw offset HUD: 在 RGB 帧上绘制半透明偏航指示器
            yaw_off_t = all_yaw_offset[t]
            yaw_deg_t = np.degrees(yaw_off_t)
            if yaw_control_active:
                # 在图像右上角绘制偏航罗盘（compass）
                img_h, img_w = rgb.shape[:2]
                cx_frac, cy_frac = 0.88, 0.18  # 右上角 (axes 坐标)
                r_frac = 0.12  # 半径（相对 axes）
                # 画圆环背景
                compass_circle = plt.Circle((cx_frac, 1.0 - cy_frac), r_frac,
                    transform=ax_rgb.transAxes, fill=True,
                    facecolor='black', edgecolor='white', alpha=0.55, linewidth=1.5, zorder=10)
                ax_rgb.add_patch(compass_circle)
                # 零位参考线 (上方竖线 = 0°)
                ax_rgb.plot([cx_frac, cx_frac],
                    [1.0 - cy_frac, 1.0 - cy_frac + r_frac * 0.85],
                    color='gray', linewidth=1, alpha=0.6,
                    transform=ax_rgb.transAxes, zorder=11)
                # 偏航指针 (从圆心指向偏航方向)
                needle_x = cx_frac + r_frac * 0.8 * np.sin(yaw_off_t)
                needle_y = (1.0 - cy_frac) + r_frac * 0.8 * np.cos(yaw_off_t)
                arrow_color = '#FF4444' if abs(yaw_deg_t) > 30 else ('#FFAA00' if abs(yaw_deg_t) > 15 else '#44FF44')
                ax_rgb.annotate('', xy=(needle_x, needle_y),
                    xytext=(cx_frac, 1.0 - cy_frac),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5),
                    transform=ax_rgb.transAxes, zorder=12)
                # 偏航角度文字
                ax_rgb.text(cx_frac, 1.0 - cy_frac - r_frac - 0.03,
                    f'YAW {yaw_deg_t:+.1f}°',
                    transform=ax_rgb.transAxes, fontsize=9, fontweight='bold',
                    ha='center', va='top', color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
                    zorder=12)
            else:
                # 旧模型无偏航控制，在右上角显示提示
                ax_rgb.text(0.88, 0.92,
                    'YAW N/A',
                    transform=ax_rgb.transAxes, fontsize=8, fontweight='bold',
                    ha='center', va='top', color='#888888',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4),
                    zorder=12)
            ax_rgb.set_title('RGB (Wide-Angle FPV)', fontsize=11)
            ax_rgb.axis('off')

            # ---- Top-right: Depth ----
            ax_dep = fig.add_subplot(gs[0, 1])
            d = rec['depth_frames'][t][sample_idx]
            d_vis = d.copy()
            valid = d > 0
            if valid.any():
                d_vis[~valid] = np.nan
                ax_dep.imshow(d_vis, cmap='viridis', vmin=0, vmax=20)
            else:
                ax_dep.imshow(np.zeros_like(d_vis), cmap='viridis', vmin=0, vmax=20)
            ax_dep.set_title('Depth (Wide-Angle)', fontsize=11)
            ax_dep.axis('off')

            # ---- Bottom-left: Real-time trajectory (top-down XY) ----
            ax_traj = fig.add_subplot(gs[1, 0])
            # Plot trajectory so far
            traj = all_p[:t+1]
            ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.2, alpha=0.8)
            ax_traj.scatter(p_start[0], p_start[1], c='lime', s=100, marker='^',
                           zorder=5, edgecolors='k', label='Start')
            ax_traj.scatter(p_target[0], p_target[1], c='red', s=100, marker='*',
                           zorder=5, edgecolors='k', label='Target')
            # Current position
            p_t = all_p[t]
            ax_traj.scatter(p_t[0], p_t[1], c='blue', s=60, marker='o',
                           zorder=6, edgecolors='k')
            # Heading arrow (机体朝向)
            hdg = rec['heading'][t, sample_idx]
            ax_traj.annotate('', xy=(p_t[0] + hdg[0]*0.8, p_t[1] + hdg[1]*0.8),
                            xytext=(p_t[0], p_t[1]),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            # Yaw-adjusted target direction arrow (偏航偏移后的感知目标方向)
            if yaw_control_active:
                yaw_t = all_yaw_offset[t]
                raw_dir = p_target[:2] - p_t[:2]  # XY 方向
                raw_len = np.linalg.norm(raw_dir)
                if raw_len > 0.1:
                    raw_unit = raw_dir / raw_len
                    # 旋转 raw_unit 以显示 yaw offset 后的感知方向
                    cos_y = np.cos(yaw_t)
                    sin_y = np.sin(yaw_t)
                    rotated_unit = np.array([
                        cos_y * raw_unit[0] - sin_y * raw_unit[1],
                        sin_y * raw_unit[0] + cos_y * raw_unit[1],
                    ])
                    arrow_len = min(1.5, raw_len * 0.3)
                    # 实际目标方向 (绿色虚线)
                    ax_traj.annotate('', xy=(p_t[0] + raw_unit[0]*arrow_len,
                                            p_t[1] + raw_unit[1]*arrow_len),
                                    xytext=(p_t[0], p_t[1]),
                                    arrowprops=dict(arrowstyle='->', color='limegreen',
                                                    lw=1.5, linestyle='--'))
                    # 偏航旋转后的感知方向 (洋红色实线)
                    ax_traj.annotate('', xy=(p_t[0] + rotated_unit[0]*arrow_len,
                                            p_t[1] + rotated_unit[1]*arrow_len),
                                    xytext=(p_t[0], p_t[1]),
                                    arrowprops=dict(arrowstyle='->', color='magenta', lw=2))
            # Collision markers
            coll_so_far = rec['collision'][:t+1, sample_idx].astype(bool)
            if coll_so_far.any():
                cp = all_p[:t+1][coll_so_far]
                ax_traj.scatter(cp[:, 0], cp[:, 1], c='red', s=25, marker='x', zorder=7)
            # Set equal aspect with some padding
            all_pts = np.vstack([all_p, p_start[None], p_target[None]])
            xmin, xmax = all_pts[:, 0].min() - 2, all_pts[:, 0].max() + 2
            ymin, ymax = all_pts[:, 1].min() - 2, all_pts[:, 1].max() + 2
            ax_traj.set_xlim(xmin, xmax)
            ax_traj.set_ylim(ymin, ymax)
            ax_traj.set_aspect('equal')
            ax_traj.set_xlabel('X (m)')
            ax_traj.set_ylabel('Y (m)')
            ax_traj.set_title('Top-Down Trajectory (XY)', fontsize=11)
            if yaw_control_active:
                # 在轨迹图添加箭头图例说明
                from matplotlib.lines import Line2D
                custom_handles = [
                    Line2D([0], [0], color='orange', lw=2, label='Heading'),
                    Line2D([0], [0], color='limegreen', lw=1.5, linestyle='--', label='To Target'),
                    Line2D([0], [0], color='magenta', lw=2, label='Yaw-Rotated'),
                ]
                ax_traj.legend(handles=ax_traj.get_legend_handles_labels()[0] + custom_handles,
                              fontsize=7, loc='upper left')
            else:
                ax_traj.legend(fontsize=8, loc='upper left')
            ax_traj.grid(True, alpha=0.3)

            # ---- Bottom-right: Detailed status panel ----
            ax_info = fig.add_subplot(gs[1, 1])
            ax_info.axis('off')

            v_t = rec['v'][t, sample_idx]
            a_t = rec['a'][t, sample_idx]
            spd = rec['speed'][t, sample_idx]
            dst = rec['dist_to_obs'][t, sample_idx]
            coll = rec['collision'][t, sample_idx]
            d2t = rec['dist_to_target'][t, sample_idx]
            depth_pct = rec['depth_valid_pct'][t, sample_idx]
            att_z = rec['attitude_z'][t, sample_idx]
            init_d2t = float(np.linalg.norm(p_start - p_target))
            progress = max(0, (1.0 - d2t / max(init_d2t, 0.01))) * 100
            total_coll = int(rec['collision'][:t+1, sample_idx].sum())

            # 偏航控制状态
            yaw_off_deg = np.degrees(all_yaw_offset[t])
            yaw_rate_deg = np.degrees(all_yaw_rate[t])

            info_lines = [
                f"Step: {t}/{T}    Time: {t * self.ctl_dt:.2f} s",
                f"=========================",
                f"  POSITION (ROS)",
                f"    X: {p_t[0]:+8.3f} m",
                f"    Y: {p_t[1]:+8.3f} m",
                f"    Z: {p_t[2]:+8.3f} m  (height)",
                f"  VELOCITY",
                f"    Vx:{v_t[0]:+7.3f}  Vy:{v_t[1]:+7.3f}  Vz:{v_t[2]:+7.3f}",
                f"    Speed: {spd:.3f} m/s",
                f"  ACCELERATION",
                f"    Ax:{a_t[0]:+7.3f}  Ay:{a_t[1]:+7.3f}  Az:{a_t[2]:+7.3f}",
                f"=========================",
                f"  OBSTACLE",
                f"    Min Dist:  {dst:.3f} m",
                f"    Margin:    {margin:.3f} m",
                f"    Collision: {'!! YES !!' if coll else 'No'}  (total: {total_coll})",
                f"  NAVIGATION",
                f"    To Target: {d2t:.2f} m",
                f"    Progress:  {progress:.1f}%",
                f"    Target: ({p_target[0]:.1f}, {p_target[1]:.1f}, {p_target[2]:.1f})",
                f"  ATTITUDE",
                f"    Heading: ({hdg[0]:+.2f}, {hdg[1]:+.2f}, {hdg[2]:+.2f})",
                f"    Tilt Z:  ({att_z[0]:+.2f}, {att_z[1]:+.2f}, {att_z[2]:+.2f})",
                f"  YAW CONTROL {'[ACTIVE]' if yaw_control_active else '[N/A 旧模型]'}",
                *([
                    f"    Offset:  {yaw_off_deg:+7.2f}°",
                    f"    Rate:    {yaw_rate_deg:+7.2f}°/s",
                    f"    |Offset|: {'▓' * min(int(abs(yaw_off_deg) / 3), 20)}{'░' * max(0, 20 - min(int(abs(yaw_off_deg) / 3), 20))} {abs(yaw_off_deg):.1f}°",
                ] if yaw_control_active else [
                    f"    (此模型无偏航控制头)",
                ]),
                f"  SENSOR",
                f"    Depth Valid: {depth_pct:.1f}%",
            ]
            info_text = "\n".join(info_lines)

            facecolor = '#FFE0E0' if coll else 'lightyellow'
            textcolor = 'red' if coll else 'black'
            ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        color=textcolor,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=facecolor, alpha=0.9))
            ax_info.set_title('Flight Status', fontsize=11)

            yaw_title = f' | Yaw {np.degrees(all_yaw_offset[t]):+.1f}°' if yaw_control_active else ''
            fig.suptitle(
                f'Episode {episode_idx} | Step {t}/{T} | '
                f'Speed {spd:.2f} m/s | Obs {dst:.2f} m | Target {d2t:.1f} m'
                f'{yaw_title}',
                fontsize=12, y=0.99
            )
            fig.subplots_adjust(top=0.95, bottom=0.05, left=0.04, right=0.98)

            # Render figure to numpy array
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(buf)
            plt.close(fig)

        writer.close()
        print(f"  [Saved] 综合视频: {comp_path}")
        return comp_path

    # ================================================================
    # 输出每步 CSV 日志
    # ================================================================
    def save_csv_log(self, rec, episode_idx, sample_idx=0, save_dir=None):
        """输出每步详细数据的 CSV 日志"""
        if save_dir is None:
            save_dir = self.args.output_dir

        T = rec['p'].shape[0]
        p_target = rec['p_target'][sample_idx]
        margin = rec['margin'][sample_idx]

        csv_path = os.path.join(save_dir, f'episode_{episode_idx:03d}_log.csv')
        with open(csv_path, 'w') as f:
            f.write('step,time_s,'
                    'pos_x,pos_y,pos_z,'
                    'vel_x,vel_y,vel_z,speed,'
                    'acc_x,acc_y,acc_z,'
                    'heading_x,heading_y,heading_z,'
                    'dist_to_obs,margin,collision,'
                    'dist_to_target,progress_pct,'
                    'depth_valid_pct\n')
            init_d2t = float(np.linalg.norm(rec['p_start'][sample_idx] - p_target))
            for t in range(T):
                p = rec['p'][t, sample_idx]
                v = rec['v'][t, sample_idx]
                a = rec['a'][t, sample_idx]
                h = rec['heading'][t, sample_idx]
                spd = rec['speed'][t, sample_idx]
                dst = rec['dist_to_obs'][t, sample_idx]
                coll = int(rec['collision'][t, sample_idx])
                d2t = rec['dist_to_target'][t, sample_idx]
                prog = max(0, (1.0 - d2t / max(init_d2t, 0.01))) * 100
                dpct = rec['depth_valid_pct'][t, sample_idx]
                f.write(f'{t},{t * self.ctl_dt:.4f},'
                        f'{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},'
                        f'{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{spd:.4f},'
                        f'{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},'
                        f'{h[0]:.4f},{h[1]:.4f},{h[2]:.4f},'
                        f'{dst:.4f},{margin:.4f},{coll},'
                        f'{d2t:.4f},{prog:.2f},'
                        f'{dpct:.2f}\n')
        print(f"  [Saved] CSV 日志: {csv_path}")
        return csv_path

    # ================================================================
    # 汇总统计
    # ================================================================
    def print_summary(self, all_records):
        """打印所有 episode 的统计摘要"""
        print("\n" + "=" * 70)
        print("                    可视化验证统计摘要")
        print("=" * 70)

        total_steps = 0
        total_collisions = 0
        total_success = 0
        total_reached = 0
        total_no_collision = 0
        speeds = []
        min_dists = []
        final_dists = []
        depth_pcts = []
        max_speeds = []
        init_dists = []

        for i, rec in enumerate(all_records):
            T = rec['p'].shape[0]
            B = rec['p'].shape[1]
            for b in range(B):
                n_coll = int(rec['collision'][:, b].sum())
                avg_spd = float(rec['speed'][:, b].mean())
                max_spd = float(rec['speed'][:, b].max())
                min_d = float(rec['dist_to_obs'][:, b].min())
                final_d = float(rec['dist_to_target'][-1, b])
                init_d = float(np.linalg.norm(rec['p_start'][b] - rec['p_target'][b]))
                reached = bool((rec['dist_to_target'][:, b] <= self.args.arrival_threshold).any())
                no_collision = (n_coll == 0)
                success = no_collision and reached
                avg_depth = float(rec['depth_valid_pct'][:, b].mean())

                total_steps += T
                total_collisions += n_coll
                total_reached += int(reached)
                total_no_collision += int(no_collision)
                total_success += int(success)
                speeds.append(avg_spd)
                max_speeds.append(max_spd)
                min_dists.append(min_d)
                final_dists.append(final_d)
                init_dists.append(init_d)
                depth_pcts.append(avg_depth)

                progress = max(0, (1 - final_d / max(init_d, 0.01))) * 100
                print(f"  Episode {i}, Drone {b}: "
                      f"碰撞={n_coll}/{T} | "
                      f"平均/峰值速度={avg_spd:.2f}/{max_spd:.2f} m/s | "
                      f"最小距离={min_d:.3f} m | "
                        f"终端距离={final_d:.2f} m ({progress:.0f}% 完成) | "
                        f"到达={reached} | 成功={success} | "
                      f"深度覆盖={avg_depth:.1f}%")

        print("-" * 70)
        n_eps = len(speeds)
        print(f"  总 Episode 数: {len(all_records)} ({n_eps} 个轨迹)")
        print(f"  碰撞率: {total_collisions}/{total_steps} "
              f"({100 * total_collisions / max(total_steps, 1):.2f}%)")
        print(f"  平均速度: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} m/s"
              f" (峰值 {np.max(max_speeds):.2f} m/s)")
        print(f"  最小障碍距离: {np.min(min_dists):.3f} m"
              f" (均值 {np.mean(min_dists):.3f} m)")
        print(f"  终端到目标距离: {np.mean(final_dists):.2f} ± {np.std(final_dists):.2f} m"
              f" (初始 {np.mean(init_dists):.1f} m)")
        avg_prog = np.mean([max(0, 1 - fd / max(id_, 0.01))
                            for fd, id_ in zip(final_dists, init_dists)]) * 100
        print(f"  平均完成进度: {avg_prog:.1f}%")
        print(f"  无碰撞率: {total_no_collision}/{n_eps} "
              f"({100 * total_no_collision / max(n_eps, 1):.2f}%)")
        print(f"  到达率: {total_reached}/{n_eps} "
              f"({100 * total_reached / max(n_eps, 1):.2f}%)")
        print(f"  成功率(无碰撞且到达): {total_success}/{n_eps} "
              f"({100 * total_success / max(n_eps, 1):.2f}%)")
        print(f"  深度图平均覆盖率: {np.mean(depth_pcts):.1f}%")
        print("=" * 70)


# ================================================================
# 主入口
# ================================================================

def main():
    args = parse_args()

    # 打印参数
    print("=" * 60)
    print("  无人机飞行可视化验证")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes:   {args.num_episodes}")
    print(f"  Timesteps:  {args.timesteps}")
    print(f"  随机场景:   {args.random_scene}")
    print(f"  输出目录:   {args.output_dir}")
    print(f"  模型分辨率: {args.image_height}x{args.image_width}")
    print(f"  可视化分辨率: {args.viz_height}x{args.viz_width}")
    print(f"  可视化 FOV:  {args.viz_fov:.0f}°")
    print("=" * 60)

    runner = EvalRunner(args)
    all_records = []

    for ep in range(args.num_episodes):
        print(f"\n{'─' * 40}")
        print(f"  运行 Episode {ep}/{args.num_episodes}")
        print(f"{'─' * 40}")

        rec = runner.run_episode(ep)
        all_records.append(rec)

        # 对每个 batch 样本生成可视化
        for b in range(args.batch_size):
            runner.plot_trajectory(rec, ep, sample_idx=b)
            runner.save_csv_log(rec, ep, sample_idx=b)

            if not args.no_video and args.save_video:
                runner.save_video(rec, ep, sample_idx=b)
                runner.save_composite_video(rec, ep, sample_idx=b)

            if args.save_frames:
                runner.save_frames(rec, ep, sample_idx=b)

    # 汇总
    runner.print_summary(all_records)
    print(f"\n所有输出已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
