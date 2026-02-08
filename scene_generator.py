"""
随机场景生成器模块

本模块提供随机场景合成功能，从基本几何体（球、圆柱、圆环、椎体等）
随机组合生成训练场景，并提供碰撞安全的出生点/目标点采样方法。

设计思路：
- 参考项目 (DiffPhysDrone) 在 CUDA 端用参数化体素/球/圆柱实现随机场景，
  每个 batch 样本看到不同的障碍物布局。
- 本项目使用 PyTorch3D 网格渲染，因此场景必须是统一的 Meshes 对象。
  但同一 batch 内所有样本共享同一场景（因为网格通过 extend(B) 广播），
  所以每 episode 随机生成一个场景即可提供泛化能力。
- 地板始终存在，障碍物从原语库中随机抽样放置。
- 安全出生点/目标点通过点云最近邻距离检测实现。

作者: Kirk
"""

import os
import math
import random
import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import sample_points_from_meshes


# ============================================================
# 基本几何体原语库
# ============================================================

# 默认原语路径 (相对于项目根目录)
DEFAULT_PRIMITIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base_model")

# 原语注册表：名称 → (文件名, 默认缩放范围, 是否为地板)
# 缩放范围为 (scale_min, scale_max)，在三个轴上独立随机缩放
PRIMITIVE_REGISTRY = {
    "sphere":   ("球1_1.obj",         (0.5, 3.0),  False),
    "cylinder": ("圆柱体2_2_2.obj",   (0.5, 3.0),  False),
    "torus":    ("圆环5_5_1.obj",     (0.5, 3.0),  False),
    "cone":     ("椎体2_2_2.obj",     (0.5, 3.0),  False),
    "floor":    ("地板.obj",          (1.0, 1.0),  True),
}


def _load_primitive_mesh(name, device, primitive_dir=None):
    """
    加载单个原语网格。

    Args:
        name: 原语名称 (PRIMITIVE_REGISTRY 中的 key)
        device: 计算设备
        primitive_dir: 原语文件目录，None 使用默认路径

    Returns:
        Meshes: 加载的网格（已居中到原点，unit bbox）
    """
    if primitive_dir is None:
        primitive_dir = DEFAULT_PRIMITIVE_DIR

    filename, _, _ = PRIMITIVE_REGISTRY[name]
    path = os.path.join(primitive_dir, filename)

    mesh = load_objs_as_meshes([path], device=device)

    # 确保有纹理
    if mesh.textures is None:
        verts = mesh.verts_list()[0]
        verts_rgb = torch.ones_like(verts)[None]
        mesh.textures = TexturesVertex(verts_features=verts_rgb)

    return mesh


def _center_mesh(mesh):
    """
    将网格居中到原点（基于包围盒中心）。

    Args:
        mesh: Meshes 对象（单个网格）

    Returns:
        Meshes: 居中后的新网格
    """
    verts = mesh.verts_packed()  # (V, 3)
    center = (verts.max(0).values + verts.min(0).values) / 2.0
    new_verts = verts - center
    return mesh.update_padded(new_verts.unsqueeze(0))


def _transform_mesh(mesh, scale, rotation_y, translation):
    """
    对网格施加缩放、Y轴旋转和平移变换。

    Args:
        mesh: Meshes 对象（单个网格）
        scale: (3,) tensor 或 float，XYZ 缩放因子
        rotation_y: float，绕 Z 轴旋转角度（弧度，ROS 坐标系中 Z 朝上）
        translation: (3,) tensor，平移向量

    Returns:
        Meshes: 变换后的新网格
    """
    verts = mesh.verts_packed()  # (V, 3)
    device = verts.device

    # 缩放
    if isinstance(scale, (float, int)):
        scale = torch.tensor([scale, scale, scale], device=device)
    verts = verts * scale

    # 绕 Z 轴旋转 (ROS 坐标系 Z 朝上)
    cos_a = math.cos(rotation_y)
    sin_a = math.sin(rotation_y)
    R = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], device=device, dtype=torch.float32)
    verts = verts @ R.T

    # 平移
    verts = verts + translation

    # 重建网格（保留纹理和面片）
    new_mesh = mesh.update_padded(verts.unsqueeze(0))
    return new_mesh


# ============================================================
# 场景生成
# ============================================================

class SceneGenerator:
    """
    随机场景生成器。

    每次调用 generate() 时，从原语库中随机抽取障碍物，
    随机变换（缩放、旋转、平移）后合并为一个场景网格。

    属性：
        device: 计算设备
        primitive_meshes: dict，预加载的原语网格缓存
        arena_range: float，场景活动区域的半径范围 (X/Y)
        obstacle_z_range: tuple，障碍物高度范围 (min_z, max_z)
        num_obstacles_range: tuple，每场景障碍物数量范围 (min, max)
    """

    def __init__(self,
                 device,
                 primitive_dir=None,
                 arena_range=10.0,
                 obstacle_z_range=(0.0, 8.0),
                 num_obstacles_range=(5, 15),
                 obstacle_scale_range=(0.5, 3.0),
                 floor_scale=(1.0, 1.0, 1.0),
                 include_floor=True,
                 seed=None):
        """
        Args:
            device: 计算设备
            primitive_dir: 原语文件目录
            arena_range: 场景 X/Y 范围 [-arena_range, arena_range]
            obstacle_z_range: 障碍物底部高度范围 (会保证 Z >= 0)
            num_obstacles_range: 障碍物数量范围 (min, max)，含两端
            obstacle_scale_range: 障碍物缩放范围
            floor_scale: 地板缩放因子 (x, y, z)
            include_floor: 是否包含地板
            seed: 随机种子（仅用于 Python random 模块）
        """
        self.device = device
        self.arena_range = arena_range
        self.obstacle_z_range = obstacle_z_range
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_scale_range = obstacle_scale_range
        self.floor_scale = floor_scale
        self.include_floor = include_floor

        if seed is not None:
            random.seed(seed)

        # 预加载所有原语网格
        self.primitive_meshes = {}
        obstacle_names = [k for k, (_, _, is_floor) in PRIMITIVE_REGISTRY.items() if not is_floor]
        for name in obstacle_names:
            self.primitive_meshes[name] = _center_mesh(
                _load_primitive_mesh(name, device=device, primitive_dir=primitive_dir)
            )
        if include_floor:
            self.primitive_meshes["floor"] = _load_primitive_mesh(
                "floor", device=device, primitive_dir=primitive_dir
            )

        self.obstacle_names = obstacle_names

    @torch.no_grad()
    def generate(self, num_obstacles=None):
        """
        生成一个随机场景。

        Returns:
            scene_mesh (Meshes): 合并后的场景网格（所有障碍物 + 地板）
            obstacle_info (list): 障碍物信息列表，每项为 dict:
                {"name": str, "center": (3,), "scale": (3,), "half_extent": (3,)}
        """
        meshes_to_join = []
        obstacle_info = []

        # 1) 地板
        if self.include_floor and "floor" in self.primitive_meshes:
            floor_mesh = self.primitive_meshes["floor"]
            floor_scale = torch.tensor(self.floor_scale, device=self.device)
            floor_transformed = _transform_mesh(
                floor_mesh,
                scale=floor_scale,
                rotation_y=0.0,
                translation=torch.zeros(3, device=self.device)
            )
            meshes_to_join.append(floor_transformed)

        # 2) 随机障碍物
        if num_obstacles is None:
            lo, hi = self.num_obstacles_range
            num_obstacles = random.randint(lo, hi)

        for _ in range(num_obstacles):
            # 随机选择原语类型
            name = random.choice(self.obstacle_names)
            base_mesh = self.primitive_meshes[name]

            # 随机缩放 (XYZ 独立，但不会差太多)
            s_lo, s_hi = self.obstacle_scale_range
            base_scale = random.uniform(s_lo, s_hi)
            # 各轴在 base_scale 基础上加一点扰动 (±30%)
            sx = base_scale * random.uniform(0.7, 1.3)
            sy = base_scale * random.uniform(0.7, 1.3)
            sz = base_scale * random.uniform(0.7, 1.3)
            scale = torch.tensor([sx, sy, sz], device=self.device)

            # 随机旋转 (绕 Z 轴)
            rot_z = random.uniform(0, 2 * math.pi)

            # 随机位置 (X/Y 均匀分布，Z 基于障碍物高度)
            tx = random.uniform(-self.arena_range, self.arena_range)
            ty = random.uniform(-self.arena_range, self.arena_range)
            # 障碍物中心 Z：确保底部不低于 0
            z_lo, z_hi = self.obstacle_z_range
            tz = random.uniform(z_lo, z_hi)
            translation = torch.tensor([tx, ty, tz], device=self.device)

            transformed = _transform_mesh(base_mesh, scale, rot_z, translation)
            meshes_to_join.append(transformed)

            # 记录障碍物信息（用于碰撞检测参考）
            # 计算变换后的包围盒半范围
            t_verts = transformed.verts_packed()
            half_ext = (t_verts.max(0).values - t_verts.min(0).values) / 2.0
            center = (t_verts.max(0).values + t_verts.min(0).values) / 2.0
            obstacle_info.append({
                "name": name,
                "center": center,
                "scale": scale,
                "half_extent": half_ext,
            })

        # 3) 合并所有网格为一个场景
        if len(meshes_to_join) == 0:
            raise ValueError("No meshes to join! Check include_floor and num_obstacles settings.")

        scene_mesh = join_meshes_as_scene(meshes_to_join)

        return scene_mesh, obstacle_info


# ============================================================
# 安全出生点/目标点采样
# ============================================================

@torch.no_grad()
def sample_safe_points(
    obstacle_pcd,
    num_points,
    arena_range=8.0,
    z_range=(1.0, 6.0),
    min_clearance=1.0,
    max_attempts=50,
    device=None,
):
    """
    采样碰撞安全的点位（出生点或目标点）。

    使用拒绝采样：随机生成候选点，检查到最近障碍物点云的距离，
    拒绝距离小于 min_clearance 的候选点。

    Args:
        obstacle_pcd: (1, N, 3) 或 (N, 3) 障碍物点云
        num_points: 需要的安全点数
        arena_range: X/Y 采样范围 [-arena_range, arena_range]
        z_range: Z 采样范围 (min, max)
        min_clearance: 最小安全距离（到最近障碍物表面）
        max_attempts: 最大采样轮数，超过则放宽约束
        device: 计算设备

    Returns:
        points: (num_points, 3) 安全点位置
    """
    from pytorch3d.ops import knn_points

    if device is None:
        device = obstacle_pcd.device

    # 归一化 obstacle_pcd 形状
    if obstacle_pcd.dim() == 2:
        obstacle_pcd = obstacle_pcd.unsqueeze(0)  # (1, N, 3)

    z_lo, z_hi = z_range
    accepted = []  # 收集已接受的点

    for attempt in range(max_attempts):
        # 每轮多生成一些候选点以加速
        n_needed = num_points - len(accepted)
        if n_needed <= 0:
            break

        n_candidates = max(n_needed * 4, 64)  # 多生成 4 倍候选
        candidates = torch.zeros(n_candidates, 3, device=device)
        candidates[:, 0] = (torch.rand(n_candidates, device=device) - 0.5) * 2 * arena_range
        candidates[:, 1] = (torch.rand(n_candidates, device=device) - 0.5) * 2 * arena_range
        candidates[:, 2] = torch.rand(n_candidates, device=device) * (z_hi - z_lo) + z_lo

        # KNN 查询最近障碍物距离
        candidates_expanded = candidates.unsqueeze(0)  # (1, n_candidates, 3)
        pcd_expanded = obstacle_pcd.expand(1, -1, -1)
        result = knn_points(candidates_expanded, pcd_expanded, K=1)
        dists = result.dists.squeeze(0).squeeze(-1).sqrt()  # (n_candidates,)

        # 根据当前轮次逐步放宽安全距离
        # 前 70% 的轮次使用完整 min_clearance，之后线性放宽到 min_clearance * 0.3
        progress = attempt / max(max_attempts - 1, 1)
        if progress < 0.7:
            current_clearance = min_clearance
        else:
            # 从 min_clearance 线性衰减到 min_clearance * 0.3
            t = (progress - 0.7) / 0.3
            current_clearance = min_clearance * (1.0 - 0.7 * t)

        safe_mask = dists > current_clearance
        safe_points = candidates[safe_mask]

        if safe_points.shape[0] > 0:
            n_take = min(safe_points.shape[0], n_needed)
            accepted.append(safe_points[:n_take])

    if len(accepted) == 0:
        # 极端退化：完全找不到安全点，返回高空随机点 (Z=z_hi)
        print("[WARNING] sample_safe_points: 无法找到安全点，使用高空后备位置")
        fallback = torch.zeros(num_points, 3, device=device)
        fallback[:, 0] = (torch.rand(num_points, device=device) - 0.5) * 2 * arena_range
        fallback[:, 1] = (torch.rand(num_points, device=device) - 0.5) * 2 * arena_range
        fallback[:, 2] = z_hi
        return fallback

    result = torch.cat(accepted, dim=0)[:num_points]

    # 如果仍不够（理论上不太可能因为有退化逻辑），填充最后一个点
    if result.shape[0] < num_points:
        pad_n = num_points - result.shape[0]
        result = torch.cat([result, result[-1:].expand(pad_n, -1)], dim=0)

    return result


@torch.no_grad()
def sample_safe_targets(
    obstacle_pcd,
    spawn_points,
    arena_range=8.0,
    z_range=(1.5, 6.0),
    min_clearance=1.0,
    min_distance=3.0,
    max_distance=8.0,
    max_attempts=50,
    device=None,
):
    """
    为每个出生点采样碰撞安全的目标点。

    在出生点周围的环形区域内采样，确保：
    1) 目标点不在障碍物内部（距离 > min_clearance）
    2) 目标点与出生点的距离在 [min_distance, max_distance] 范围内

    Args:
        obstacle_pcd: (1, N, 3) 障碍物点云
        spawn_points: (B, 3) 出生点位置
        arena_range: X/Y 场景范围
        z_range: 目标点 Z 范围
        min_clearance: 到障碍物最小安全距离
        min_distance: 到出生点最小距离
        max_distance: 到出生点最大距离
        max_attempts: 最大采样轮数
        device: 计算设备

    Returns:
        targets: (B, 3) 安全目标点
    """
    from pytorch3d.ops import knn_points

    if device is None:
        device = obstacle_pcd.device

    if obstacle_pcd.dim() == 2:
        obstacle_pcd = obstacle_pcd.unsqueeze(0)

    B = spawn_points.shape[0]
    z_lo, z_hi = z_range
    found = torch.zeros(B, dtype=torch.bool, device=device)
    targets = torch.zeros(B, 3, device=device)

    for attempt in range(max_attempts):
        n_remaining = (~found).sum().item()
        if n_remaining == 0:
            break

        remaining_mask = ~found
        remaining_spawn = spawn_points[remaining_mask]  # (R, 3)
        R = remaining_spawn.shape[0]

        # 随机角度 + 距离偏移
        angle = torch.rand(R, device=device) * 2 * math.pi
        dist = torch.rand(R, device=device) * (max_distance - min_distance) + min_distance

        candidates = remaining_spawn.clone()
        candidates[:, 0] = candidates[:, 0] + torch.cos(angle) * dist
        candidates[:, 1] = candidates[:, 1] + torch.sin(angle) * dist
        candidates[:, 2] = candidates[:, 2] + torch.randn(R, device=device) * 2.0
        candidates[:, 2] = candidates[:, 2].clamp(z_lo, z_hi)

        # 限制在场景范围内
        candidates[:, 0] = candidates[:, 0].clamp(-arena_range, arena_range)
        candidates[:, 1] = candidates[:, 1].clamp(-arena_range, arena_range)

        # 检查安全距离
        candidates_expanded = candidates.unsqueeze(0)  # (1, R, 3)
        pcd_expanded = obstacle_pcd.expand(1, -1, -1)
        result = knn_points(candidates_expanded, pcd_expanded, K=1)
        dists = result.dists.squeeze(0).squeeze(-1).sqrt()  # (R,)

        # 逐步放宽约束
        progress = attempt / max(max_attempts - 1, 1)
        if progress < 0.7:
            current_clearance = min_clearance
        else:
            t = (progress - 0.7) / 0.3
            current_clearance = min_clearance * (1.0 - 0.7 * t)

        safe = dists > current_clearance

        # 更新已找到的目标
        remaining_indices = torch.where(remaining_mask)[0]
        for i, idx in enumerate(remaining_indices):
            if safe[i] and not found[idx]:
                targets[idx] = candidates[i]
                found[idx] = True

    # 对于未找到的，使用高空后备
    if not found.all():
        n_missing = (~found).sum().item()
        print(f"[WARNING] sample_safe_targets: {n_missing}/{B} 个目标点使用后备位置")
        missing_idx = torch.where(~found)[0]
        for idx in missing_idx:
            targets[idx] = spawn_points[idx].clone()
            targets[idx, 2] = z_hi  # 移到高空

    return targets
