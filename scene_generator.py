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
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import sample_points_from_meshes


# ============================================================
# 网格简化工具
# ============================================================

def _decimate_mesh(mesh, target_faces, device=None):
    """
    基于顶点聚类 (Vertex Clustering) 的网格简化。

    纯 PyTorch 实现，不依赖 trimesh / open3d。
    原理：将顶点位置量化到均匀网格，同一格子内的顶点合并为一个
    （使用格内均值），面片重映射后去除退化三角形。

    用于在加载原语时预简化高面片网格（如圆环 1152→~200 面），
    从而降低整个场景的面片总数和渲染开销。

    Args:
        mesh: Meshes 对象（单个网格）
        target_faces: 目标面片数（近似值，实际结果可能略有偏差）
        device: 计算设备

    Returns:
        Meshes: 简化后的新网格（保留纹理颜色）
    """
    verts = mesh.verts_packed()           # (V, 3)
    faces = mesh.faces_packed()           # (F, 3)
    if device is None:
        device = verts.device

    num_faces = faces.shape[0]
    if num_faces <= target_faces:
        return mesh  # 已经足够简单，无需简化

    # 自适应网格分辨率：通过二分搜索找到使结果面片数最接近 target 的 grid_res
    bbox_min = verts.min(0).values
    bbox_max = verts.max(0).values
    bbox_size = bbox_max - bbox_min
    bbox_size = bbox_size.clamp(min=1e-6)

    best_res = 4
    best_diff = float('inf')
    lo_res, hi_res = 3, 64

    for _ in range(16):  # 二分搜索
        mid_res = (lo_res + hi_res) // 2
        if mid_res <= lo_res:
            break
        # 试量化
        cell_size = bbox_size / mid_res
        q = ((verts - bbox_min) / cell_size).long().clamp(max=mid_res - 1)
        keys = q[:, 0] * (mid_res * mid_res) + q[:, 1] * mid_res + q[:, 2]
        inv = keys.unique(return_inverse=True)[1]
        test_faces = inv[faces]
        valid = (test_faces[:, 0] != test_faces[:, 1]) & \
                (test_faces[:, 1] != test_faces[:, 2]) & \
                (test_faces[:, 0] != test_faces[:, 2])
        n_valid = valid.sum().item()
        diff = abs(n_valid - target_faces)
        if diff < best_diff:
            best_diff = diff
            best_res = mid_res
        if n_valid < target_faces:
            lo_res = mid_res  # 网格太粗 → 结果面太少 → 需要更细
        else:
            hi_res = mid_res  # 面太多 → 需要更粗

    grid_res = best_res
    cell_size = bbox_size / grid_res

    # 量化顶点到网格
    quantized = ((verts - bbox_min) / cell_size).long().clamp(max=grid_res - 1)
    keys = quantized[:, 0] * (grid_res * grid_res) + quantized[:, 1] * grid_res + quantized[:, 2]

    # 为每个唯一 key 分配新顶点 ID
    unique_keys, inverse = keys.unique(return_inverse=True)
    num_new_verts = unique_keys.shape[0]

    # 新顶点位置 = 格内顶点均值
    new_verts = torch.zeros(num_new_verts, 3, device=device)
    counts = torch.zeros(num_new_verts, device=device)
    new_verts.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), verts)
    counts.scatter_add_(0, inverse, torch.ones(verts.shape[0], device=device))
    new_verts = new_verts / counts.unsqueeze(-1).clamp(min=1)

    # 新顶点颜色 = 格内均值
    has_texture = mesh.textures is not None
    new_colors = None
    if has_texture:
        try:
            vert_colors = mesh.textures.verts_features_packed()  # (V, C)
            new_colors = torch.zeros(num_new_verts, vert_colors.shape[1], device=device)
            new_colors.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, vert_colors.shape[1]), vert_colors)
            new_colors = new_colors / counts.unsqueeze(-1).clamp(min=1)
        except Exception:
            new_colors = torch.ones(num_new_verts, 3, device=device)

    # 重映射面片
    new_faces = inverse[faces]  # (F, 3)

    # 去除退化面（三个顶点中有重复的）
    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
            (new_faces[:, 1] != new_faces[:, 2]) & \
            (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[valid]

    # 向量化去重面片（排序后去重）
    if new_faces.shape[0] > 0:
        sorted_faces, _ = new_faces.sort(dim=1)
        face_keys = sorted_faces[:, 0].long() * (num_new_verts + 1) ** 2 + \
                    sorted_faces[:, 1].long() * (num_new_verts + 1) + \
                    sorted_faces[:, 2].long()
        _, inv_idx, counts_f = face_keys.unique(return_inverse=True, return_counts=True)
        # 使用 scatter_reduce 找到每个 unique key 的第一个出现位置
        perm = torch.arange(new_faces.shape[0], device=device)
        first_occ = torch.full((counts_f.shape[0],), new_faces.shape[0], dtype=torch.long, device=device)
        first_occ.scatter_reduce_(0, inv_idx, perm, reduce='amin')
        new_faces = new_faces[first_occ]

    if new_faces.shape[0] == 0:
        return mesh  # 简化失败，返回原始网格

    # 紧凑化：去掉未使用的顶点
    used_verts = new_faces.unique()
    remap = torch.full((num_new_verts,), -1, dtype=torch.long, device=device)
    remap[used_verts] = torch.arange(used_verts.shape[0], device=device)
    new_verts = new_verts[used_verts]
    new_faces = remap[new_faces]

    # 构建新网格
    if new_colors is not None:
        new_colors = new_colors[used_verts]
        textures = TexturesVertex(verts_features=new_colors.unsqueeze(0))
    else:
        textures = TexturesVertex(verts_features=torch.ones_like(new_verts).unsqueeze(0))

    return Meshes(verts=[new_verts], faces=[new_faces], textures=textures)


def _weld_scene_vertices(mesh, epsilon=1e-4):
    """
    场景级顶点焊接：合并距离小于 epsilon 的重复顶点，去除退化面片。

    在 join_meshes_as_scene 之后调用，可合并重叠物体间的近似重复顶点，
    并清理退化/重复面片，减少场景总面片数。

    Args:
        mesh: Meshes 对象（合并后的场景网格）
        epsilon: 焊接距离阈值（默认 0.1mm）

    Returns:
        Meshes: 焊接后的新网格
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    device = verts.device

    # 用量化实现近似顶点合并（O(V) 复杂度）
    quantized = (verts / epsilon).round().long()
    keys = quantized[:, 0] * 1000003 + quantized[:, 1] * 1009 + quantized[:, 2]
    unique_keys, inverse = keys.unique(return_inverse=True)

    if unique_keys.shape[0] == verts.shape[0]:
        return mesh  # 无重复顶点

    num_new = unique_keys.shape[0]
    new_verts = torch.zeros(num_new, 3, device=device)
    counts = torch.zeros(num_new, device=device)
    new_verts.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), verts)
    counts.scatter_add_(0, inverse, torch.ones(verts.shape[0], device=device))
    new_verts = new_verts / counts.unsqueeze(-1).clamp(min=1)

    # 处理纹理
    new_colors = None
    if mesh.textures is not None:
        try:
            vert_colors = mesh.textures.verts_features_packed()
            new_colors = torch.zeros(num_new, vert_colors.shape[1], device=device)
            new_colors.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, vert_colors.shape[1]), vert_colors)
            new_colors = new_colors / counts.unsqueeze(-1).clamp(min=1)
        except Exception:
            new_colors = torch.ones(num_new, 3, device=device)

    new_faces = inverse[faces]

    # 去退化面
    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
            (new_faces[:, 1] != new_faces[:, 2]) & \
            (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[valid]

    if new_faces.shape[0] == 0:
        return mesh

    # 紧凑化
    used = new_faces.unique()
    remap = torch.full((num_new,), -1, dtype=torch.long, device=device)
    remap[used] = torch.arange(used.shape[0], device=device)
    new_verts = new_verts[used]
    new_faces = remap[new_faces]

    if new_colors is not None:
        new_colors = new_colors[used]
        textures = TexturesVertex(verts_features=new_colors.unsqueeze(0))
    else:
        textures = TexturesVertex(verts_features=torch.ones_like(new_verts).unsqueeze(0))

    return Meshes(verts=[new_verts], faces=[new_faces], textures=textures)


# ============================================================
# 坐标系转换工具
# ============================================================

def obj_to_ros(pos):
    """
    OBJ 坐标系 (Y-up) 与 ROS 坐标系 (Z-up) 之间的转换。

    映射关系（自逆变换，双向适用）：
      OBJ (x, y, z) ↔ ROS (-x, z, y)

    Args:
        pos: (..., 3) tensor，源坐标系中的位置

    Returns:
        (..., 3) tensor，目标坐标系中的位置
    """
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    return torch.stack([-x, z, y], dim=-1)

# ROS→OBJ 是完全相同的变换（自逆）
ros_to_obj = obj_to_ros


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
    "cube":     ("方块.obj",         (0.5, 3.0),  False),
}


def _load_primitive_mesh(name, device, primitive_dir=None):
    """
    加载单个原语网格。

    Args:
        name: 原型名称 (PRIMITIVE_REGISTRY 中的 key)
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


def _random_rotation_matrix(device, max_tilt=torch.pi / 4, generator=None):
    """
    生成随机旋转矩阵：完整 yaw 旋转 + 受限 pitch/roll。

    用于给障碍物添加多轴旋转，使场景更自然、更具挑战性。
    Yaw 在 [0, 2π) 范围完全随机，Pitch/Roll 在 [-max_tilt, max_tilt] 范围。

    Args:
        device: 计算设备
        max_tilt: pitch/roll 最大倾斜角（弧度），默认 π/4（45°）

    Returns:
        (3, 3) 旋转矩阵 (float32)
    """
    yaw = torch.rand((), device=device, generator=generator) * (2.0 * torch.pi)
    pitch = (torch.rand((), device=device, generator=generator) * 2.0 - 1.0) * max_tilt
    roll = (torch.rand((), device=device, generator=generator) * 2.0 - 1.0) * max_tilt

    cy, cp, cr = yaw.cos(), pitch.cos(), roll.cos()
    sy, sp, sr = yaw.sin(), pitch.sin(), roll.sin()

    # OBJ 坐标系: Y-up，Yaw 绕 Y 轴, Pitch 绕 X 轴, Roll 绕 Z 轴
    z = torch.zeros((), device=device)
    o = torch.ones((), device=device)
    Ry = torch.stack([torch.stack([cy, z, sy]), torch.stack([z, o, z]), torch.stack([-sy, z, cy])])
    Rx = torch.stack([torch.stack([o, z, z]), torch.stack([z, cp, -sp]), torch.stack([z, sp, cp])])
    Rz = torch.stack([torch.stack([cr, -sr, z]), torch.stack([sr, cr, z]), torch.stack([z, z, o])])
    return Ry @ Rx @ Rz


def _transform_mesh(mesh, scale, rotation_y=0.0, translation=None, rotation_matrix=None):
    """
    对网格施加缩放、旋转和平移变换。

    支持两种旋转方式：
      1. rotation_matrix (3×3 tensor): 完整旋转矩阵（优先使用）
      2. rotation_y (float): 仅绕 OBJ Y 轴旋转（兼容旧代码）

    注意：本函数在 OBJ 坐标系中工作（Y 轴朝上）。

    Args:
        mesh: Meshes 对象（单个网格）
        scale: (3,) tensor 或 float，XYZ 缩放因子
        rotation_y: float，绕 OBJ Y 轴旋转角度（弧度），当 rotation_matrix=None 时使用
        translation: (3,) tensor，OBJ 坐标系中的平移向量
        rotation_matrix: (3,3) tensor，完整旋转矩阵，优先于 rotation_y

    Returns:
        Meshes: 变换后的新网格
    """
    verts = mesh.verts_packed()  # (V, 3)
    device = verts.device

    # 缩放
    if isinstance(scale, (float, int)):
        scale = torch.tensor([scale, scale, scale], device=device)
    verts = verts * scale

    # 旋转
    if rotation_matrix is not None:
        verts = verts @ rotation_matrix.T
    else:
        cos_a = torch.tensor(rotation_y).cos().item()
        sin_a = torch.tensor(rotation_y).sin().item()
        R = torch.tensor([
            [ cos_a, 0, sin_a],
            [ 0,     1, 0    ],
            [-sin_a, 0, cos_a]
        ], device=device, dtype=torch.float32)
        verts = verts @ R.T

    # 平移
    if translation is not None:
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

    每次调用 generate() 时，从原型库中随机抽取障碍物，
    随机变换（缩放、旋转、平移）后合并为一个场景网格。

    坐标系约定：
      本模块在 OBJ 坐标系中工作（Y 轴朝上），与加载的 .obj 文件一致。
      OBJ ↔ ROS 映射: ROS (x,y,z) → OBJ (-x, z, y)  (自逆变换)
      - OBJ X/Z: 水平平面
      - OBJ Y: 高度（上方向）
      - 地板表面位于 OBJ Y ≈ 0

    属性：
        device: 计算设备
        primitive_meshes: dict，预加载的原语网格缓存
        arena_range: float，场景活动区域的半径范围 (水平面 X/Z)
        obstacle_z_range: tuple，障碍物高度范围 (min_height, max_height)，
                          在 OBJ 坐标系中映射到 Y 轴
        num_obstacles_range: tuple，每场景障碍物数量范围 (min, max)
        primitive_half_heights: dict，预计算的各原语居中后 Y 方向半高度
    """

    def __init__(self,
                 device,
                 primitive_dir=None,
                 arena_range=6.0,
                 obstacle_z_range=(0.0, 5.0),
                 num_obstacles_range=(20, 40),
                 obstacle_scale_range=(0.3, 1.5),
                 floor_scale=None,
                 include_floor=True,
                 concentration=0.0,
                 grid_jitter=True,
                 enable_3d_rotation=True,
                 max_tilt=torch.pi / 4,
                 ground_clearance=0.05,
                 ground_ratio=0.6,
                 cluster_ratio=0.3,
                 cluster_spread=1.5,
                 normalize_primitives=True,
                 max_faces_per_primitive=200,
                 weld_scene=True,
                 seed=None):
        """
        Args:
            device: 计算设备
            primitive_dir: 文件目录
            arena_range: 场景水平范围 [-arena_range, arena_range]（OBJ X/Z 平面）
            obstacle_z_range: 障碍物高度范围 (min, max)。
                              语义为"高度"，内部映射到 OBJ Y 轴。
                              会自动保证障碍物底部不低于地板 (OBJ Y >= 0)。
            num_obstacles_range: 障碍物数量范围 (min, max)，含两端
            obstacle_scale_range: 障碍物缩放范围
            floor_scale: 地板缩放因子 (x, y, z)，默认 None 自动根据 arena_range 计算
            include_floor: 是否包含地板
            concentration: 障碍物向中心集中的比例 (0=均匀, 1=全部集中)。
                           仅在 grid_jitter=False 时生效。默认 0.0（纯均匀分布）。
            grid_jitter: 使用网格抖动分布（推荐）。将场景分为网格，每格放置0或1个
                         障碍物，确保全场均匀覆盖（包括边缘区域）。默认 True。
            enable_3d_rotation: 启用三轴旋转（yaw + 受限 pitch/roll），默认 True。
                                False 则仅绕 Y 轴旋转（兼容旧行为）。
            max_tilt: 三轴旋转时 pitch/roll 最大倾斜角（弧度），默认 π/4。
            ground_clearance: 障碍物底部距地面最小间距（OBJ Y），默认 0.05m。
            ground_ratio: 接地物体比例 (0.0~1.0)。接地物体底部紧贴地面，
                          Y 轴缩放偏大形成柱子/墙壁，迫使无人机绕行而非从下方穿越。
                          默认 0.6（60% 物体接地）。
            cluster_ratio: 簇生物体比例 (0.0~1.0)。这部分物体不使用网格位置，
                           而是放置在已有物体附近，允许重叠以形成复合形状，
                           提高模型对大型/异形障碍物的泛化能力。默认 0.3。
            cluster_spread: 簇生物体相对于父物体的最大水平偏移距离 (m)，默认 1.5。
                            值越小，组合越紧密；值越大，组合越松散。
            normalize_primitives: 将所有原语归一化到单位包围盒 (max extent = 1.0)，
                                  确保不同形状的缩放行为一致。默认 True。
                                  归一化后 scale=1.0 表示最大维度为 1m。
            max_faces_per_primitive: 每个原语的最大面片数，超过则在加载时自动简化。
                                     默认 200，可显著降低场景总面片数（尤其是圆环 1152→~200、
                                     球 960→~200）。设为 0 或 None 禁用简化。
            weld_scene: 在 join_meshes_as_scene 后进行顶点焊接和退化面清理。
                        可合并重叠簇生物体间的重复顶点，默认 True。
            seed: 随机种子（仅用于 Python random 模块）
        """
        self.device = device
        self.arena_range = arena_range
        self.obstacle_z_range = obstacle_z_range
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_scale_range = obstacle_scale_range
        self.floor_scale = floor_scale
        self.include_floor = include_floor
        self.concentration = concentration
        self.grid_jitter = grid_jitter
        self.enable_3d_rotation = enable_3d_rotation
        self.max_tilt = max_tilt
        self.ground_clearance = ground_clearance
        self.ground_ratio = ground_ratio
        self.cluster_ratio = cluster_ratio
        self.cluster_spread = cluster_spread
        self.normalize_primitives = normalize_primitives
        self.max_faces_per_primitive = max_faces_per_primitive
        self.weld_scene = weld_scene

        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

        # 预加载所有原语网格
        self.primitive_meshes = {}
        self.primitive_half_heights = {}  # 各原语居中后 Y 方向半高度
        obstacle_names = [k for k, (_, _, is_floor) in PRIMITIVE_REGISTRY.items() if not is_floor]
        for name in obstacle_names:
            centered = _center_mesh(
                _load_primitive_mesh(name, device=device, primitive_dir=primitive_dir)
            )

            # 可选：归一化到单位包围盒（max extent = 1.0）
            if normalize_primitives:
                verts = centered.verts_packed()
                max_extent = (verts.max(0).values - verts.min(0).values).max().item()
                if max_extent > 1e-6:
                    normalized_verts = verts / max_extent
                    centered = centered.update_padded(normalized_verts.unsqueeze(0))

            # 简化高面片原语（在20 加载时一次性完成，不影响 generate 速度）
            if max_faces_per_primitive and max_faces_per_primitive > 0:
                n_before = centered.faces_packed().shape[0]
                if n_before > max_faces_per_primitive:
                    centered = _decimate_mesh(centered, max_faces_per_primitive, device=device)
                    n_after = centered.faces_packed().shape[0]
                    print(f"  [Decimate] {name}: {n_before} → {n_after} faces "
                          f"({(1 - n_after/n_before)*100:.0f}% 减少)")

            self.primitive_meshes[name] = centered
            # 预计算 Y 方向半高度（居中后 Y_max ≈ -Y_min ≈ half_height）
            verts_y = centered.verts_packed()[:, 1]
            self.primitive_half_heights[name] = verts_y.max().item()

        if include_floor:
            floor_raw = _load_primitive_mesh(
                "floor", device=device, primitive_dir=primitive_dir
            )
            # 居中地板 XZ（保持 Y 不变，使顶面仍在 Y≈0）
            verts = floor_raw.verts_packed()
            center = (verts.min(0).values + verts.max(0).values) / 2
            center[1] = 0.0  # Y 轴不偏移
            floor_raw = floor_raw.update_padded((verts - center).unsqueeze(0))
            self.primitive_meshes["floor"] = floor_raw

            # 自动计算地板缩放：覆盖 arena 并留 20% 余量
            if self.floor_scale is None:
                floor_extent = (verts.max(0).values - verts.min(0).values)[[0, 2]].max().item()
                s = 2.0 * self.arena_range * 1.2 / floor_extent
                self.floor_scale = (s, 1.0, s)

        self.obstacle_names = obstacle_names

    def _rand(self, *size):
        shape = size if size else ()
        return torch.rand(shape, device=self.device, generator=self.generator)

    def _randn(self, *size):
        shape = size if size else ()
        return torch.randn(shape, device=self.device, generator=self.generator)

    def _rand_scalar(self, low=0.0, high=1.0):
        return float((self._rand() * (high - low) + low).item())

    def _rand_int(self, low, high):
        return int(torch.randint(low, high, (), device=self.device, generator=self.generator).item())

    def _rand_bool(self, p_true):
        return bool((self._rand() < p_true).item())

    def _grid_jitter_positions(self, num_obstacles):
        """
        网格抖动位置生成：将场景划分为均匀网格，随机选取网格并在格内抖动。

        确保障碍物均匀覆盖整个场景（包括边缘区域），避免中心聚集。

        Args:
            num_obstacles: 需要的障碍物数量

        Returns:
            list of (x, z) 位置元组（OBJ 坐标系水平面）
        """
        arena = self.arena_range
        # 网格边数：略多于 sqrt(num_obstacles)，保证有足够的格子
        grid_side = max(
            int(torch.ceil(torch.sqrt(torch.tensor(float(num_obstacles) * 1.2, device=self.device))).item()),
            3,
        )
        cell_size = 2.0 * arena / grid_side

        coords = torch.arange(grid_side, device=self.device, dtype=torch.float32)
        cx, cz = torch.meshgrid(coords, coords, indexing='ij')
        centers = torch.stack([
            -arena + (cx.reshape(-1) + 0.5) * cell_size,
            -arena + (cz.reshape(-1) + 0.5) * cell_size,
        ], dim=-1)

        perm = torch.randperm(centers.shape[0], device=self.device, generator=self.generator)
        selected = centers[perm[:num_obstacles]]

        jitter = (self._rand(num_obstacles, 2) * 0.8 - 0.4) * cell_size
        positions_t = (selected + jitter).clamp(-arena + 0.05, arena - 0.05)
        positions = [tuple(p.tolist()) for p in positions_t]

        # 如果需要的障碍物多于网格数，额外随机填充
        while len(positions) < num_obstacles:
            px = self._rand_scalar(-arena, arena)
            pz = self._rand_scalar(-arena, arena)
            positions.append((px, pz))

        return positions

    def _random_positions(self, num_obstacles):
        """
        随机位置生成（旧模式兼容）：支持可选的中心集中分布。

        Args:
            num_obstacles: 障碍物数量

        Returns:
            list of (x, z) 位置元组
        """
        arena = self.arena_range
        sigma = arena / 3.0
        positions = []
        for _ in range(num_obstacles):
            if self._rand_bool(self.concentration):
                tx = float(self._randn().mul(sigma).clamp(-arena, arena).item())
                tz = float(self._randn().mul(sigma).clamp(-arena, arena).item())
            else:
                tx = self._rand_scalar(-arena, arena)
                tz = self._rand_scalar(-arena, arena)
            positions.append((tx, tz))
        return positions

    @torch.no_grad()
    def generate(self, num_obstacles=None):
        """
        生成一个随机场景。

        放置策略（两层架构）：
        1. **主体障碍物** (n_primary)：通过网格抖动均匀覆盖场景。
        2. **簇生障碍物** (n_cluster)：放置在已有障碍物附近（cluster_spread 范围内），
           允许重叠以形成复合形状，提高模型对大型/异形障碍物的泛化能力。

        接地机制：
        - 每个障碍物以 ground_ratio 概率"接地"——底部紧贴地面，
          Y 轴缩放偏大（1.0–3.0×），形成柱子/墙壁/岩石等地面障碍物。
        - 非接地物体照旧在 [height_lo, height_hi] 范围内随机悬浮。
        - 这迫使无人机学习绕行机动，而非从所有物体下方穿越。

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
            num_obstacles = self._rand_int(lo, hi + 1)

        # 分离主体与簇生障碍物
        n_primary = max(int(num_obstacles * (1 - self.cluster_ratio)), 1)
        n_cluster = num_obstacles - n_primary

        # 生成主体水平位置
        if self.grid_jitter:
            primary_positions = self._grid_jitter_positions(n_primary)
        else:
            primary_positions = self._random_positions(n_primary)

        height_lo, height_hi = self.obstacle_z_range

        # 已放置障碍物中心列表（供簇生物体参考）
        placed_centers = []

        def _place_one(px, pz, force_grounded=None):
            """放置单个障碍物，返回无。直接修改外层 meshes_to_join / obstacle_info。"""
            name = self.obstacle_names[self._rand_int(0, len(self.obstacle_names))]
            base_mesh = self.primitive_meshes[name]
            base_verts = base_mesh.verts_packed()  # (V, 3), 已居中 (归一化后)

            # 决定是否接地
            is_grounded = force_grounded if force_grounded is not None \
                else self._rand_bool(self.ground_ratio)

            # 随机缩放 (XYZ 独立，基础值 ±30% 扰动)
            s_lo, s_hi = self.obstacle_scale_range
            base_scale = self._rand_scalar(s_lo, s_hi)
            sx = base_scale * self._rand_scalar(0.7, 1.3)
            sz = base_scale * self._rand_scalar(0.7, 1.3)
            if is_grounded:
                # 接地物体：Y 轴缩放偏大，形成柱子/墙壁/树桩等
                sy = base_scale * self._rand_scalar(1.0, 3.0)
            else:
                sy = base_scale * self._rand_scalar(0.7, 1.3)
            scale = torch.tensor([sx, sy, sz], device=self.device)

            # 缩放
            verts = base_verts * scale

            # 旋转
            if self.enable_3d_rotation:
                R = _random_rotation_matrix(self.device, self.max_tilt, generator=self.generator)
                verts = verts @ R.T
            else:
                rot_y = self._rand() * (2.0 * torch.pi)
                cos_a = rot_y.cos()
                sin_a = rot_y.sin()
                z = torch.zeros((), device=self.device)
                o = torch.ones((), device=self.device)
                Ry = torch.stack([
                    torch.stack([cos_a, z, sin_a]),
                    torch.stack([z, o, z]),
                    torch.stack([-sin_a, z, cos_a]),
                ])
                verts = verts @ Ry.T

            # 动态计算旋转后实际 Y 范围（精确防止地下放置）
            y_min_local = verts[:, 1].min().item()

            if is_grounded:
                # 接地：底部紧贴地面
                ty = self.ground_clearance - y_min_local
            else:
                # 悬浮：在 [min_ty, height_hi] 范围内随机
                min_ty = self.ground_clearance - y_min_local  # y_min_local 通常为负
                min_ty = max(min_ty, height_lo)
                ty = self._rand_scalar(min_ty, height_hi) if min_ty < height_hi else min_ty

            # 平移
            translation = torch.tensor([px, ty, pz], device=self.device)
            verts = verts + translation

            # 创建变换后的网格
            transformed = base_mesh.update_padded(verts.unsqueeze(0))
            meshes_to_join.append(transformed)

            # 记录障碍物信息（用于碰撞检测参考）
            half_ext = (verts.max(0).values - verts.min(0).values) / 2.0
            center = (verts.max(0).values + verts.min(0).values) / 2.0
            obstacle_info.append({
                "name": name,
                "center": center,
                "scale": scale,
                "half_extent": half_ext,
            })
            placed_centers.append((px, ty, pz))

        # --- 主体障碍物：均匀网格分布 ---
        for px, pz in primary_positions:
            _place_one(px, pz)

        # --- 簇生障碍物：在已有物体附近组合，形成复合形状 ---
        for _ in range(n_cluster):
            if placed_centers:
                parent = placed_centers[self._rand_int(0, len(placed_centers))]
                dx = self._rand_scalar(-self.cluster_spread, self.cluster_spread)
                dz = self._rand_scalar(-self.cluster_spread, self.cluster_spread)
                cx = max(-self.arena_range + 0.05,
                         min(self.arena_range - 0.05, parent[0] + dx))
                cz = max(-self.arena_range + 0.05,
                         min(self.arena_range - 0.05, parent[2] + dz))
                _place_one(cx, cz)
            else:
                px = self._rand_scalar(-self.arena_range, self.arena_range)
                pz = self._rand_scalar(-self.arena_range, self.arena_range)
                _place_one(px, pz)

        # 3) 合并所有网格为一个场景
        if len(meshes_to_join) == 0:
            raise ValueError("No meshes to join! Check include_floor and num_obstacles settings.")

        scene_mesh = join_meshes_as_scene(meshes_to_join)

        # 场景级拓扑优化：合并重复顶点、清理退化面
        if self.weld_scene:
            v_before = scene_mesh.verts_packed().shape[0]
            f_before = scene_mesh.faces_packed().shape[0]
            scene_mesh = _weld_scene_vertices(scene_mesh, epsilon=1e-4)
            v_after = scene_mesh.verts_packed().shape[0]
            f_after = scene_mesh.faces_packed().shape[0]
            if v_before != v_after or f_before != f_after:
                pass  # 静默优化，可取消注释以查看效果
                # print(f"  [Weld] verts {v_before}→{v_after}, faces {f_before}→{f_after}")

        return scene_mesh, obstacle_info


# ============================================================
# 跨地图出生/目标点采样
# ============================================================

def _greedy_accept_separated(candidates, existing, min_dist):
    """
    贪心接受：从 candidates 中选出互相间距 ≥ min_dist 且离 existing 也足够远的点。

    核心优化：在 GPU 上一次性算完所有距离矩阵，再在 CPU 上做贪心选取，
    避免原来每个候选调 .item() 的 GPU→CPU 同步瓶颈。

    Args:
        candidates: (C, 3) GPU tensor, 候选点
        existing:   (M, 3) GPU tensor, 已接受的点 (M 可为 0)
        min_dist:   float, 最小间距

    Returns:
        accepted_indices: list[int], 在 candidates 中被接受的索引
    """
    C = candidates.shape[0]
    if C == 0:
        return []

    # 1) 若有 existing，过滤掉离 existing 太近的候选 (全 GPU)
    if existing.shape[0] > 0:
        d_to_ex = torch.cdist(candidates, existing)       # (C, M) GPU
        far_mask = d_to_ex.min(dim=1).values >= min_dist   # (C,) GPU
        valid_idx = torch.where(far_mask)[0]
        if valid_idx.shape[0] == 0:
            return []
        candidates_sub = candidates[valid_idx]
    else:
        valid_idx = torch.arange(C, device=candidates.device)
        candidates_sub = candidates

    # 2) 候选间两两距离 (全 GPU，一次性)
    pair_dist_gpu = torch.cdist(candidates_sub, candidates_sub)  # (C', C')

    # 3) 转 CPU 做贪心 (矩阵很小 ≤ 256×256，转移 <0.1ms)
    pair_dist = pair_dist_gpu.cpu()
    Cs = pair_dist.shape[0]
    accepted_local = []
    for i in range(Cs):
        ok = True
        for j in accepted_local:
            if pair_dist[i, j].item() < min_dist:
                ok = False
                break
        if ok:
            accepted_local.append(i)

    # 映射回原始 candidates 索引
    valid_idx_cpu = valid_idx.cpu()
    return [int(valid_idx_cpu[i]) for i in accepted_local]


@torch.no_grad()
def sample_cross_map_spawn_target(
    obstacle_pcd,
    num_points,
    arena_range=6.0,
    z_range=(1.0, 3.0),
    min_clearance=1.0,
    min_inter_distance=0.0,
    max_attempts=50,
    device=None,
):
    """
    采样跨地图的出生/目标点对（OBJ 坐标系，Y-up）。

    策略：
    - 为每个 batch 元素随机选取一个"穿越方向"角度 θ
    - 出生点在 θ 方向的"负侧"（靠近一个边缘）
    - 目标点在 θ 方向的"正侧"（靠近对面边缘）
    - 确保无人机必须穿越场景中央区域，不能绕行边缘
    - 两点均通过碰撞检测保证安全

    Args:
        obstacle_pcd: (1, N, 3) 或 (N, 3) 障碍物点云（OBJ 坐标系）
        num_points: 需要的点对数（= batch_size）
        arena_range: 水平场景范围
        z_range: 高度范围 (min, max)，映射到 OBJ Y
        min_clearance: 到障碍物最小安全距离
        max_attempts: 最大采样轮数
        device: 计算设备

    Returns:
        spawn_points: (num_points, 3) 出生点（OBJ 坐标系）
        target_points: (num_points, 3) 目标点（OBJ 坐标系）
    """
    from pytorch3d.ops import knn_points

    if device is None:
        device = obstacle_pcd.device
    if obstacle_pcd.dim() == 2:
        obstacle_pcd = obstacle_pcd.unsqueeze(0)

    height_lo, height_hi = z_range

    # 每个 batch 元素的穿越方向
    theta = torch.rand(num_points, device=device) * 2 * torch.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    spawn = torch.zeros(num_points, 3, device=device)
    target = torch.zeros(num_points, 3, device=device)
    spawn_found = torch.zeros(num_points, dtype=torch.bool, device=device)
    target_found = torch.zeros(num_points, dtype=torch.bool, device=device)

    R = arena_range

    for attempt in range(max_attempts):
        # 逐步放宽安全距离
        progress = attempt / max(max_attempts - 1, 1)
        if progress < 0.7:
            current_clearance = min_clearance
        else:
            t = (progress - 0.7) / 0.3
            current_clearance = min_clearance * (1.0 - 0.7 * t)

        # ---- 出生点：在穿越方向的"负侧" ----
        if not spawn_found.all():
            idx = torch.where(~spawn_found)[0]
            n = idx.shape[0]
            n_cand = max(n * 8, 64)

            along = -(torch.rand(n_cand, device=device) * 0.7 + 0.2) * R
            perp = (torch.rand(n_cand, device=device) - 0.5) * 1.6 * R
            idx_rep = idx.repeat((n_cand + n - 1) // n)[:n_cand]
            cx = along * cos_t[idx_rep] - perp * sin_t[idx_rep]
            cz = along * sin_t[idx_rep] + perp * cos_t[idx_rep]
            cy = torch.rand(n_cand, device=device) * (height_hi - height_lo) + height_lo
            cx = cx.clamp(-R * 0.95, R * 0.95)
            cz = cz.clamp(-R * 0.95, R * 0.95)
            candidates = torch.stack([cx, cy, cz], dim=-1)

            dists = knn_points(candidates.unsqueeze(0), obstacle_pcd, K=1).dists.squeeze(0).squeeze(-1).sqrt()
            safe = dists > current_clearance

            if min_inter_distance > 0:
                # 收集所有安全候选（不分 batch 元素），做向量化贪心
                safe_cands = candidates[safe]
                if safe_cands.shape[0] > 0:
                    existing = spawn[spawn_found]
                    accepted_idx = _greedy_accept_separated(safe_cands, existing, min_inter_distance)
                    # 分配接受的候选点到对应 batch 元素
                    safe_idx_rep = idx_rep[safe]
                    for ai in accepted_idx:
                        gid = safe_idx_rep[ai].item()
                        if not spawn_found[gid]:
                            spawn[gid] = safe_cands[ai]
                            spawn_found[gid] = True
            else:
                for k, global_idx in enumerate(idx):
                    mask_k = (idx_rep == global_idx) & safe
                    safe_cands = candidates[mask_k]
                    if safe_cands.shape[0] > 0 and not spawn_found[global_idx]:
                        spawn[global_idx] = safe_cands[0]
                        spawn_found[global_idx] = True

        # ---- 目标点：在穿越方向的"正侧" ----
        if not target_found.all():
            idx = torch.where(~target_found)[0]
            n = idx.shape[0]
            n_cand = max(n * 8, 64)

            along = (torch.rand(n_cand, device=device) * 0.7 + 0.2) * R
            perp = (torch.rand(n_cand, device=device) - 0.5) * 1.6 * R
            idx_rep = idx.repeat((n_cand + n - 1) // n)[:n_cand]
            cx = along * cos_t[idx_rep] - perp * sin_t[idx_rep]
            cz = along * sin_t[idx_rep] + perp * cos_t[idx_rep]
            cy = torch.rand(n_cand, device=device) * (height_hi - height_lo) + height_lo
            cx = cx.clamp(-R * 0.95, R * 0.95)
            cz = cz.clamp(-R * 0.95, R * 0.95)
            candidates = torch.stack([cx, cy, cz], dim=-1)

            dists = knn_points(candidates.unsqueeze(0), obstacle_pcd, K=1).dists.squeeze(0).squeeze(-1).sqrt()
            safe = dists > current_clearance

            if min_inter_distance > 0:
                safe_cands = candidates[safe]
                if safe_cands.shape[0] > 0:
                    existing = target[target_found]
                    accepted_idx = _greedy_accept_separated(safe_cands, existing, min_inter_distance)
                    safe_idx_rep = idx_rep[safe]
                    for ai in accepted_idx:
                        gid = safe_idx_rep[ai].item()
                        if not target_found[gid]:
                            target[gid] = safe_cands[ai]
                            target_found[gid] = True
            else:
                for k, global_idx in enumerate(idx):
                    mask_k = (idx_rep == global_idx) & safe
                    safe_cands = candidates[mask_k]
                    if safe_cands.shape[0] > 0 and not target_found[global_idx]:
                        target[global_idx] = safe_cands[0]
                        target_found[global_idx] = True

        if spawn_found.all() and target_found.all():
            break

    # 后备：未找到的点使用默认位置
    if not spawn_found.all():
        missing = ~spawn_found
        n_miss = missing.sum().item()
        print(f"[WARNING] sample_cross_map: {n_miss} spawn points using fallback")
        spawn[missing, 0] = -R * 0.7
        spawn[missing, 1] = height_hi
        spawn[missing, 2] = 0.0

    if not target_found.all():
        missing = ~target_found
        n_miss = missing.sum().item()
        print(f"[WARNING] sample_cross_map: {n_miss} target points using fallback")
        target[missing, 0] = R * 0.7
        target[missing, 1] = height_hi
        target[missing, 2] = 0.0

    return spawn, target


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
    min_inter_distance=0.0,
    max_attempts=50,
    device=None,
):
    """
    采样碰撞安全的点位（出生点或目标点），在 OBJ 坐标系中工作。

    使用拒绝采样：随机生成候选点，检查到最近障碍物点云的距离，
    拒绝距离小于 min_clearance 的候选点。

    坐标系约定 (OBJ, Y-up):
      - index 0 (X): 水平，范围 [-arena_range, arena_range]
      - index 1 (Y): 高度（上方向），范围 z_range
      - index 2 (Z): 水平，范围 [-arena_range, arena_range]

    返回值在 OBJ 坐标系中，调用方负责转换到 ROS 坐标系。

    Args:
        obstacle_pcd: (1, N, 3) 或 (N, 3) 障碍物点云（OBJ 坐标系）
        num_points: 需要的安全点数
        arena_range: 水平采样范围 [-arena_range, arena_range]（OBJ X/Z）
        z_range: 高度采样范围 (min, max)，语义为"高度"，映射到 OBJ Y 轴
        min_clearance: 最小安全距离（到最近障碍物表面）
        max_attempts: 最大采样轮数，超过则放宽约束
        device: 计算设备

    Returns:
        points: (num_points, 3) 安全点位置（OBJ 坐标系）
    """
    from pytorch3d.ops import knn_points

    if device is None:
        device = obstacle_pcd.device

    # 归一化 obstacle_pcd 形状
    if obstacle_pcd.dim() == 2:
        obstacle_pcd = obstacle_pcd.unsqueeze(0)  # (1, N, 3)

    height_lo, height_hi = z_range
    accepted = []  # 收集已接受的点

    for attempt in range(max_attempts):
        # 每轮多生成一些候选点以加速
        n_needed = num_points - len(accepted)
        if n_needed <= 0:
            break

        n_candidates = max(n_needed * 4, 64)  # 多生成 4 倍候选
        candidates = torch.zeros(n_candidates, 3, device=device)
        # OBJ X/Z = 水平面, OBJ Y = 高度
        candidates[:, 0] = (torch.rand(n_candidates, device=device) - 0.5) * 2 * arena_range
        candidates[:, 1] = torch.rand(n_candidates, device=device) * (height_hi - height_lo) + height_lo
        candidates[:, 2] = (torch.rand(n_candidates, device=device) - 0.5) * 2 * arena_range

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
            if min_inter_distance > 0:
                # 向量化贪心：GPU 批量距离 + CPU 贪心选取
                existing = torch.stack(accepted) if accepted else safe_points.new_zeros(0, 3)
                idxs = _greedy_accept_separated(safe_points, existing, min_inter_distance)
                for i in idxs:
                    if len(accepted) >= num_points:
                        break
                    accepted.append(safe_points[i])
            else:
                n_take = min(safe_points.shape[0], n_needed)
                for j in range(n_take):
                    accepted.append(safe_points[j])

    if len(accepted) == 0:
        # 极端退化：完全找不到安全点，返回高空随机点 (OBJ Y=height_hi)
        print("[WARNING] sample_safe_points: 无法找到安全点，使用高空后备位置")
        fallback = torch.zeros(num_points, 3, device=device)
        fallback[:, 0] = (torch.rand(num_points, device=device) - 0.5) * 2 * arena_range
        fallback[:, 1] = height_hi  # OBJ Y = 高度
        fallback[:, 2] = (torch.rand(num_points, device=device) - 0.5) * 2 * arena_range
        return fallback

    result = torch.stack(accepted[:num_points])

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
    min_inter_distance=0.0,
    max_attempts=50,
    device=None,
):
    """
    为每个出生点采样碰撞安全的目标点（OBJ 坐标系，Y-up）。

    在出生点周围的环形区域内采样，确保：
    1) 目标点不在障碍物内部（距离 > min_clearance）
    2) 目标点与出生点的距离在 [min_distance, max_distance] 范围内

    坐标系约定 (OBJ, Y-up):
      - index 0 (X): 水平
      - index 1 (Y): 高度（上方向），范围 z_range
      - index 2 (Z): 水平
      水平偏移在 XZ 平面，高度偏移在 Y 轴。

    Args:
        obstacle_pcd: (1, N, 3) 障碍物点云（OBJ 坐标系）
        spawn_points: (B, 3) 出生点位置（OBJ 坐标系）
        arena_range: 水平场景范围（OBJ X/Z）
        z_range: 目标点高度范围，映射到 OBJ Y
        min_clearance: 到障碍物最小安全距离
        min_distance: 到出生点最小距离
        max_distance: 到出生点最大距离
        max_attempts: 最大采样轮数
        device: 计算设备

    Returns:
        targets: (B, 3) 安全目标点（OBJ 坐标系）
    """
    from pytorch3d.ops import knn_points

    if device is None:
        device = obstacle_pcd.device

    if obstacle_pcd.dim() == 2:
        obstacle_pcd = obstacle_pcd.unsqueeze(0)

    B = spawn_points.shape[0]
    height_lo, height_hi = z_range
    found = torch.zeros(B, dtype=torch.bool, device=device)
    targets = torch.zeros(B, 3, device=device)

    for attempt in range(max_attempts):
        n_remaining = (~found).sum().item()
        if n_remaining == 0:
            break

        remaining_mask = ~found
        remaining_spawn = spawn_points[remaining_mask]  # (R, 3)
        R = remaining_spawn.shape[0]

        # 随机角度 + 距离偏移（在 OBJ XZ 水平面内）
        angle = torch.rand(R, device=device) * 2 * torch.pi
        dist = torch.rand(R, device=device) * (max_distance - min_distance) + min_distance

        candidates = remaining_spawn.clone()
        candidates[:, 0] = candidates[:, 0] + torch.cos(angle) * dist   # OBJ X (水平)
        candidates[:, 2] = candidates[:, 2] + torch.sin(angle) * dist   # OBJ Z (水平)
        candidates[:, 1] = candidates[:, 1] + torch.randn(R, device=device) * 2.0  # OBJ Y (高度偏移)
        candidates[:, 1] = candidates[:, 1].clamp(height_lo, height_hi)

        # 限制在场景水平范围内（OBJ X/Z）
        candidates[:, 0] = candidates[:, 0].clamp(-arena_range, arena_range)
        candidates[:, 2] = candidates[:, 2].clamp(-arena_range, arena_range)

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

        # 更新已找到的目标 — 向量化贪心
        remaining_indices = torch.where(remaining_mask)[0]
        safe_cand = candidates[safe]
        safe_rem_idx = remaining_indices[safe]
        if safe_cand.shape[0] > 0:
            if min_inter_distance > 0:
                existing = targets[found] if found.any() else safe_cand.new_zeros(0, 3)
                accepted_idxs = _greedy_accept_separated(safe_cand, existing, min_inter_distance)
                for ai in accepted_idxs:
                    idx = safe_rem_idx[ai]
                    if not found[idx]:
                        targets[idx] = safe_cand[ai]
                        found[idx] = True
            else:
                for i in range(safe_cand.shape[0]):
                    idx = safe_rem_idx[i]
                    if not found[idx]:
                        targets[idx] = safe_cand[i]
                        found[idx] = True

    # 对于未找到的，使用高空后备
    if not found.all():
        n_missing = (~found).sum().item()
        print(f"[WARNING] sample_safe_targets: {n_missing}/{B} 个目标点使用后备位置")
        missing_idx = torch.where(~found)[0]
        for idx in missing_idx:
            targets[idx] = spawn_points[idx].clone()
            targets[idx, 1] = height_hi  # OBJ Y = 高空

    return targets
