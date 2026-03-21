"""
无人机渲染器模块

本模块提供了一个基于 PyTorch3D 的可微分 3D 渲染器类 `DroneRenderer`，
用于加载无人机网格模型并生成 RGB 图像和深度图，支持批量渲染和坐标系转换。

设计原则：
- DroneRenderer 作为统一底座，支持不同分辨率/FOV 的渲染需求。
- 通过 create_variant() 方法共享网格和光照，仅更换分辨率/FOV/光栅化参数。
- 消除可视化脚本中独立 HighResRenderer 的硬编码重复。

作者: KirkChinese
版本: 2.0 (统一底座)
日期: 2026-01-10
"""

import torch
from pytorch3d.ops import sample_points_from_meshes, knn_points, SubdivideMeshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    PerspectiveCameras,
    PointLights,
    TexturesVertex,
    look_at_view_transform
)

# ============================================================
# 公共工具：FOV ↔ 焦距转换 (GPU-safe, 避免 math 模块)
# ============================================================
_PI = torch.tensor(torch.pi)  # CPU 常量，仅用于初始化

def hfov_to_focal(hfov_deg: float, image_width: int) -> float:
    """从水平 FOV (度) 计算焦距 (像素)。仅初始化调用。"""
    half_rad = torch.deg2rad(torch.tensor(hfov_deg / 2.0))
    return float(image_width / 2.0 / half_rad.tan())

def focal_to_hfov(focal: float, image_width: int) -> float:
    """从焦距 (像素) 计算水平 FOV (度)。"""
    return float(2.0 * torch.rad2deg(torch.atan(torch.tensor(image_width / 2.0 / focal))))

def focal_to_vfov(focal: float, image_height: int) -> float:
    """从焦距 (像素) 计算垂直 FOV (度)。"""
    return float(2.0 * torch.rad2deg(torch.atan(torch.tensor(image_height / 2.0 / focal))))


def build_cam_mount_R(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, device=None, batch_size=1):
    """
    构造相机安装旋转矩阵（机体坐标系内）。

    旋转约定（航空惯例，相对于机体 X-forward / Y-left / Z-up）：
    - pitch: 正值 → 相机向下倾斜（俯视）
    - roll:  正值 → 相机向右倾斜
    - yaw:   正值 → 相机向右偏转

    顺序: R = Rz(yaw) @ Rx(roll) @ Ry(-pitch)
    （pitch 取负号使得"正值=向下"与航空惯例一致）

    Args:
        roll_deg, pitch_deg, yaw_deg: 相机安装角 (度)。可为 float 或 (B,) Tensor。
        device: 目标设备。
        batch_size: 若角度为标量则扩展到此 batch 大小。

    Returns:
        R_mount: (B, 3, 3) 旋转矩阵
    """
    _D2R = torch.pi / 180.0

    # 自动推断 batch_size：若任一角度是 Tensor，以其尺寸为准
    for _a in (roll_deg, pitch_deg, yaw_deg):
        if isinstance(_a, torch.Tensor):
            batch_size = _a.shape[0]
            break

    def _to_rad(x):
        if isinstance(x, (int, float)):
            return torch.full((batch_size,), x * _D2R, device=device)
        return x.to(device=device, dtype=torch.float32) * _D2R

    p = _to_rad(pitch_deg)   # pitch
    r = _to_rad(roll_deg)    # roll
    y = _to_rad(yaw_deg)     # yaw

    # Ry(-pitch): 正 pitch → camera looks down
    cp, sp = torch.cos(p), torch.sin(p)
    cr, sr = torch.cos(r), torch.sin(r)
    cy, sy = torch.cos(y), torch.sin(y)
    z = torch.zeros_like(p)
    o = torch.ones_like(p)

    Ry = torch.stack([cp, z, -sp, z, o, z, sp, z, cp], dim=-1).reshape(-1, 3, 3)
    Rx = torch.stack([o, z, z, z, cr, sr, z, -sr, cr], dim=-1).reshape(-1, 3, 3)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], dim=-1).reshape(-1, 3, 3)

    return Rz @ Rx @ Ry

class DroneRenderer:
    """
    可微分无人机 3D 渲染器类。

    该类使用 PyTorch3D 加载无人机网格模型并进行光栅化渲染，支持生成 RGB 图像和深度图。
    提供坐标系转换功能，将 ROS 坐标系下的无人机状态转换为渲染所需的相机参数。
    支持批量渲染，可用于训练可微分无人机控制系统。

    Attributes:
        device (torch.device): 计算设备（CPU 或 GPU）。
        image_size (tuple): 输出图像尺寸 (H, W)。
        H (int): 图像高度。
        W (int): 图像宽度。
        focal_length (tuple): 相机焦距 ((fx, fy),)。
        principal_point (tuple): 相机主点 ((cx, cy),)。
        mesh (Meshes): 加载的 3D 网格模型。
        raster_settings (RasterizationSettings): 光栅化设置。
        lights (PointLights): 场景光照。
        rasterizer (MeshRasterizer): 网格光栅化器。
        shader (SoftPhongShader): Phong 着色器。
        obstacle_pcd (torch.Tensor): 从网格采样的障碍物点云，形状 (1, N, 3)。

    Args:
        mesh_path (str): .obj 模型文件的路径。
        device (torch.device, optional): 计算设备，默认自动检测 CUDA。
        image_size (tuple, optional): 输出图像尺寸 (H, W)，默认 (480, 640)。
        focal_length (float or tuple, optional): 相机焦距，默认 500.0。
        num_samples (int, optional): 点云采样点数，默认 20000。

    Raises:
        FileNotFoundError: 如果 mesh_path 指定的文件不存在。
        ValueError: 如果 image_size 或 focal_length 参数无效。
    """
    def __init__(self,
                 mesh_path, 
                 device=None, 
                 image_size=(480, 640), 
                 focal_length=500.0,
                 principal_point=None,
                 lights_location=[[0.0, 0.0, -3.0]],
                 num_samples=20000,
                 subdivide_times=3,
                 z_clip_value=0.3):
        """
        初始化无人机渲染器。

        加载指定的网格模型，设置渲染参数和组件，包括光栅化器、着色器和障碍物点云。

        Args:
            mesh_path (str): .obj 模型文件的路径。
            device (torch.device, optional): 计算设备，默认自动检测 CUDA。
            image_size (tuple, optional): 输出图像尺寸 (H, W)，默认 (480, 640)。
            focal_length (float or tuple, optional): 相机焦距，默认 500.0。
            principal_point (tuple, optional): 相机主点 (cx, cy)。默认图像中心。
            lights_location (list, optional): 光源位置，默认 [[0, 0, -3.0]]。
            num_samples (int, optional): 点云采样点数，默认 20000。
            subdivide_times (int, optional): 网格细分次数，默认 3。
                设为 0 配合 z_clip_value 可获得等价质量但 10-40x 更快的渲染。
                细分主要用于精细化网格几何，不再是解决面片消失的手段。
            z_clip_value (float, optional): 近平面裁剪值，默认 0.3。
                将跨越近平面的三角形裁剪，防止投影到无穷大后消失。
                设为 0.3 与训练中 depth.clamp(0.3, 24) 对齐，
                避免无人机贴近障碍物时 clip_faces 产生海量碎片导致 OOM。
                设为 None 禁用（不推荐，会导致近处三角形丢失）。

        Raises:
            FileNotFoundError: 如果 mesh_path 指定的文件不存在。
        """
        # 设置计算设备
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.image_size = image_size  # (H, W)
        self.H, self.W = image_size
        self.mesh_path = mesh_path
        
        # 处理焦距参数 — 预计算为 Tensor，避免每帧重复 tuple→tensor 转换
        if isinstance(focal_length, (float, int)) or (torch.is_tensor(focal_length) and focal_length.dim() == 0):
            fl = float(focal_length)
            self.focal_length = torch.tensor([[fl, fl]], dtype=torch.float32, device=self.device)
        else:
            self.focal_length = torch.tensor([focal_length], dtype=torch.float32, device=self.device)
            
        if principal_point is None:
            self.principal_point = torch.tensor([[self.W / 2.0, self.H / 2.0]], dtype=torch.float32, device=self.device)
        else:
            self.principal_point = torch.tensor([principal_point], dtype=torch.float32, device=self.device)
        
        self.image_size_tensor = torch.tensor([[self.H, self.W]], dtype=torch.float32, device=self.device)
        
        # 保存细分次数以便 update_mesh 时使用
        self.subdivide_times = subdivide_times
        
        # 加载网格模型
        self.mesh = self._load_mesh(mesh_path, subdivide_times=subdivide_times)
        
        # 配置光栅化设置
        # bin_size=None → auto：PyTorch3D 自动选择 coarse binning 大小，
        # 配合 max_faces_per_bin 防止 bin 溢出。
        # 相比 bin_size=0（naive O(F×P)），coarse-to-fine 在面片多时显著更快。
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            perspective_correct=True,
            z_clip_value=z_clip_value,
            bin_size=None,           # auto coarse-to-fine（替代 naive）
            max_faces_per_bin=50000, # 防止场景障碍物多时 bin 溢出
        )
        
        # 初始化光照
        self.lights = PointLights(device=self.device, location=lights_location)
        
        # 初始化渲染组件
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, lights=self.lights)
        
        # 从网格采样障碍物点云
        self.obstacle_pcd = sample_points_from_meshes(self.mesh, num_samples=num_samples).to(self.device)
        
        # Mesh.extend() 缓存：训练中 batch size 固定，避免每帧重建
        self._extended_mesh_cache = None
        self._extended_mesh_bs = 0

        # compute_view_matrix 中的规范向量缓存（避免每帧重建 tensor）
        self._fwd_canonical = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3, 1)
        self._up_canonical = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3, 1)

        # 动态场景合成：额外网格（无人机机体、动态障碍物等）在渲染时与静态场景合并
        self._dynamic_meshes = []
        self._dynamic_pcds = []

    # ================================================================
    # 动态场景合成 API
    # ================================================================

    def set_dynamic_meshes(self, meshes, pcds=None):
        """
        设置渲染时需要额外合成的动态网格（如无人机机体、动态障碍物）。

        调用后下一次 render() 会将这些网格与静态场景合并渲染。
        pcds 用于碰撞检测：动态障碍物的点云会被纳入 full_obstacle_pcd。

        Args:
            meshes: list[Meshes]，需要合成的额外网格。
            pcds: list[Tensor(1, N, 3)]，可选，额外网格对应的点云（OBJ 坐标系）。
        """
        self._dynamic_meshes = meshes if meshes else []
        self._dynamic_pcds = pcds if pcds else []
        # 使 extend 缓存失效
        self._extended_mesh_cache = None
        self._extended_mesh_bs = 0

    def clear_dynamic_meshes(self):
        """清除所有动态合成网格。"""
        self.set_dynamic_meshes([], [])

    @property
    def render_mesh(self):
        """
        渲染用网格：静态场景 + 动态额外网格。

        无动态网格时直接返回 self.mesh，避免多余的 join 开销。
        """
        if not self._dynamic_meshes:
            return self.mesh
        from pytorch3d.structures import join_meshes_as_scene
        return join_meshes_as_scene([self.mesh] + self._dynamic_meshes)

    @property
    def full_obstacle_pcd(self):
        """
        碰撞检测用点云：静态障碍物 + 动态障碍物点云。

        不包含无人机机体点云（无人机间碰撞由 inter_drone_distances 单独处理）。
        """
        if not self._dynamic_pcds:
            return self.obstacle_pcd
        return torch.cat([self.obstacle_pcd] + self._dynamic_pcds, dim=1)

    def create_variant(self, image_size=None, hfov_deg=None, focal_length=None,
                       z_clip_value=0.3, max_faces_per_bin=50000):
        """
        创建共享 mesh / lights / obstacle_pcd 的渲染器变体。

        用于同一场景下不同分辨率/FOV 的渲染需求（如高分辨率可视化），
        避免重复加载网格和点云。

        Args:
            image_size: (H, W) 输出分辨率，默认复用当前渲染器设置。
            hfov_deg: 水平 FOV (度)，与 focal_length 二选一。
            focal_length: 直接指定焦距，优先于 hfov_deg。
            z_clip_value: 近平面裁剪值。
            max_faces_per_bin: 光栅化 bin 容量（高分辨率需要更大值）。

        Returns:
            DroneRendererVariant: 共享 mesh 的轻量渲染器。
        """
        if image_size is None:
            image_size = self.image_size
        H, W = image_size

        if focal_length is None:
            if hfov_deg is not None:
                focal_length = hfov_to_focal(hfov_deg, W)
            else:
                focal_length = float(self.focal_length[0, 0])

        return DroneRendererVariant(
            parent=self,
            image_size=image_size,
            focal_length=focal_length,
            z_clip_value=z_clip_value,
            max_faces_per_bin=max_faces_per_bin,
        )

    def _load_mesh(self, mesh_path, subdivide_times=3):
        """
        加载 .obj 格式的 3D 网格模型，并对大面片进行细分处理。

        如果模型无纹理，则创建默认白色纹理以确保渲染正常进行。
        通过网格细分解决 PyTorch3D 对大平面渲染时的精度问题，
        避免在特定视角下（如视角与平面平行时）大面片消失的情况。

        Args:
            mesh_path (str): 模型文件路径。
            subdivide_times (int): 网格细分次数，默认为 3。每次细分会将
                                   每个三角形分割为 4 个小三角形。
                                   注意：细分会增加顶点和面片数量，
                                   n 次细分后面片数量变为原来的 4^n 倍。

        Returns:
            Meshes: 加载的网格对象，包含顶点、法线和纹理信息。

        Raises:
            FileNotFoundError: 如果文件不存在。
            RuntimeError: 如果文件格式不支持或加载失败。
        """
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        
        # 检查并修复纹理：如果无纹理，创建默认白色纹理
        if mesh.textures is None:
            verts = mesh.verts_list()[0]
            verts_rgb = torch.ones_like(verts)[None]  # 默认白色
            mesh.textures = TexturesVertex(verts_features=verts_rgb)
        
        # 对网格进行细分处理，解决大面片渲染精度问题
        # 大面片在视角与其平行时容易出现渲染消失的问题
        if subdivide_times > 0:
            # 获取顶点颜色特征用于细分后的纹理插值
            # 注意：SubdivideMeshes 要求 feats 为 packed 格式 (V, D)，
            # 而 verts_features_padded() 返回 (N, V, D)，会导致维度错误
            verts_rgb = mesh.textures.verts_features_packed()  # (V, 3)
            
            for _ in range(subdivide_times):
                subdivider = SubdivideMeshes(mesh)
                mesh, verts_rgb = subdivider(mesh, feats=verts_rgb)
            
            # 更新细分后的纹理
            # SubdivideMeshes 可能返回 packed (V', D) 或 padded (1, V', D)，需统一处理
            if verts_rgb.dim() == 2:
                verts_rgb = verts_rgb[None]  # packed → padded
            mesh.textures = TexturesVertex(verts_features=verts_rgb)
            
        return mesh

    def update_mesh(self, mesh_path, num_samples=20000, subdivide_times=None):
        """
        更换渲染的网格模型。

        加载新的 .obj 文件并更新内部网格数据，同时重新采样障碍物点云。

        Args:
            mesh_path (str): 新的网格文件路径。
            num_samples (int, optional): 重新采样的点数，默认 20000。
            subdivide_times (int, optional): 网格细分次数。默认使用初始化时的设置。

        Raises:
            FileNotFoundError: 如果文件不存在。
        """
        if subdivide_times is None:
            subdivide_times = self.subdivide_times
        self.mesh_path = mesh_path
        self.mesh = self._load_mesh(mesh_path, subdivide_times=subdivide_times)
        # 重新采样障碍物点云
        self.obstacle_pcd = sample_points_from_meshes(self.mesh, num_samples=num_samples).to(self.device)
        # 使 extend 缓存失效
        self._extended_mesh_cache = None
        self._extended_mesh_bs = 0

    def render(self, R, T, return_tensor=False, return_rgb=True, return_depth=True,clean_depth=False, dt=None):
        """
        渲染给定视角的 RGB 图像和深度图。

        使用 PyTorch3D 进行光栅化渲染，支持批量处理。
        注意：此函数直接接收 PyTorch3D 标准的相机外参 (World-to-View)。
        通常建议先调用 `compute_view_matrix` 从 ROS 状态获取这里的 R 和 T。

        Args:
            R (torch.Tensor): 旋转矩阵 (World -> View)，形状 (3, 3) 或 (B, 3, 3)。
            T (torch.Tensor): 平移向量 (World -> View)，形状 (3,) 或 (B, 3)。
            return_tensor (bool, optional): 是否返回 PyTorch 张量（带梯度），默认 False，返回 NumPy 数组。
            return_rgb (bool, optional): 是否计算并返回 RGB 图像，默认 True。
            return_depth (bool, optional): 是否计算并返回深度图，默认 True。
            clean_depth (bool, optional): 是否清洗深度图，默认 False。
            dt (float, optional): 时间步长，可选，用于后续扩展（如光流计算）。
            
        Returns:
            tuple: (rgb_image, depth_map)
                - rgb_image (torch.Tensor or numpy.ndarray or None): RGB 图像，形状 (B, H, W, 3) 或 (H, W, 3)。
                - depth_map (torch.Tensor or numpy.ndarray or None): 深度图，形状 (B, H, W) 或 (H, W)。
                如果对应标志为 False，则返回 None。

        Raises:
            ValueError: 如果 R 或 T 的形状无效。
        """
        # 确保输入为张量
        if not torch.is_tensor(R):
            R = torch.tensor(R, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(T):
            T = torch.tensor(T, device=self.device, dtype=torch.float32)

        # 处理批量维度
        is_batch = True
        if R.dim() == 2: 
            R = R.unsqueeze(0)
            is_batch = False
        if T.dim() == 1: 
            T = T.unsqueeze(0)
        
        # 创建当前帧的相机
        cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            image_size=self.image_size_tensor,
            in_ndc=False,  # 使用屏幕像素坐标
            R=R,
            T=T,
            device=self.device
        )
        
        # 扩展网格以匹配批量大小（缓存复用，避免每帧重建）
        bs = len(cameras)
        if self._extended_mesh_bs != bs or self._extended_mesh_cache is None:
            self._extended_mesh_cache = self.render_mesh.extend(bs)
            self._extended_mesh_bs = bs
        meshes = self._extended_mesh_cache
        fragments = self.rasterizer(meshes, cameras=cameras)
        
        rgb_images = None
        depth_maps = None

        # 仅在需要 RGB 时执行着色
        if return_rgb:
            images = self.shader(fragments, meshes, cameras=cameras)
            rgb_images = images[..., :3]  # 提取 RGB 通道

        # 提取深度图
        if return_depth:
            depth_maps = fragments.zbuf[..., 0]

        # 如果输入非批量，输出也降维
        if not is_batch:
            if rgb_images is not None:
                rgb_images = rgb_images[0]
            if depth_maps is not None:
                depth_maps = depth_maps[0]

        # 根据 return_tensor 决定输出格式
        if return_tensor:
            if clean_depth: # 清洗深度图
                depth_maps = self.clean_depth_map(depth_maps)
            return rgb_images, depth_maps

        # 转换为 NumPy 数组（detach）
        rgb_out = None
        depth_out = None
        if rgb_images is not None:
            rgb_out = rgb_images.detach().cpu().numpy()
        if depth_maps is not None:
            depth_out = depth_maps.detach().cpu().numpy()
        
        return rgb_out, depth_out

    def render_with_mesh(self, mesh_extended, R, T,
                         return_rgb=True, return_depth=True):
        """
        直接使用预扩展的 mesh 渲染，跳过 extend 缓存机制。
        用于 per-group 渲染优化：调用方预先构建好 mesh.extend(G)
        避免每组重复 set_dynamic_meshes → join → extend 的开销。
        """
        cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            image_size=self.image_size_tensor,
            in_ndc=False, R=R, T=T, device=self.device,
        )
        fragments = self.rasterizer(mesh_extended, cameras=cameras)
        rgb_images = None
        depth_maps = None
        if return_rgb:
            images = self.shader(fragments, mesh_extended, cameras=cameras)
            rgb_images = images[..., :3]
        if return_depth:
            depth_maps = fragments.zbuf[..., 0]
        return rgb_images, depth_maps

    def compute_view_matrix(self, p_ros, R_ros, camera_pitch_deg=None,
                            cam_offset_body=None, cam_mount_R=None):
        """
        将无人机在 ROS 坐标系下的状态转换为 PyTorch3D 渲染器所需的外参 (R, T)。

        该函数自动处理坐标系转换、相机安装偏移和 LookAt 变换计算。

        Args:
            p_ros (torch.Tensor): 无人机位置 (ROS World Frame ENU)，形状 (B, 3)。
            R_ros (torch.Tensor): 无人机姿态旋转矩阵 (Body -> ROS World)，形状 (B, 3, 3)。
            camera_pitch_deg (float|Tensor, optional): 相机安装俯仰角度 (度)。
                正值 → 向下（俯视），负值 → 向上（仰视）。
                当 cam_mount_R 为 None 时使用，默认 10.0。
            cam_offset_body (list or Tensor, optional): 相机在机体坐标系下的偏移 [x, y, z]，默认 [0.1, 0, 0]。
            cam_mount_R (Tensor, optional): 相机安装旋转矩阵 (3,3) 或 (B,3,3)。
                由 build_cam_mount_R() 生成；提供后 camera_pitch_deg 被忽略。

        Returns:
            tuple: (R_view, T_view) — World-to-View 变换，形状 (B,3,3), (B,3)。
        """
        B = p_ros.shape[0]
        device = p_ros.device
        
        # ---- 平移偏移 ----
        if cam_offset_body is None:
            cam_offset_body = [0.0, 0.0, 0.0]
        
        if not torch.is_tensor(cam_offset_body):
            cam_offset_body = torch.tensor(cam_offset_body, device=device, dtype=torch.float32)
        
        if cam_offset_body.dim() == 1:
            cam_offset_body = cam_offset_body.view(1, 3, 1).expand(B, -1, -1)
        elif cam_offset_body.dim() == 2:
            cam_offset_body = cam_offset_body.unsqueeze(2)
            
        p_cam_ros = p_ros + torch.bmm(R_ros, cam_offset_body).squeeze(2)
        
        # ---- 安装旋转矩阵 ----
        if cam_mount_R is not None:
            # 使用调用方提供的完整安装旋转矩阵
            R_mount = cam_mount_R
            if R_mount.dim() == 2:
                R_mount = R_mount.unsqueeze(0).expand(B, -1, -1)
        else:
            # 从 camera_pitch_deg 构造（仅 pitch，向后兼容）
            if camera_pitch_deg is None:
                camera_pitch_deg = 10.0
            R_mount = build_cam_mount_R(
                pitch_deg=camera_pitch_deg, device=device, batch_size=B,
            )
        
        # 计算 Look At 指向向量和 Up 向量 (在机体坐标系下)
        forward_canonical = self._fwd_canonical.expand(B, -1, -1)
        up_canonical = self._up_canonical.expand(B, -1, -1)
        
        # 应用安装旋转
        look_dir_body = torch.bmm(R_mount, forward_canonical)
        up_dir_body = torch.bmm(R_mount, up_canonical)
        
        # 转换到世界坐标系
        p_target_ros = p_cam_ros + torch.bmm(R_ros, look_dir_body).squeeze(2)  # 目标点距离相机 1.0 米
        up_vec_ros = torch.bmm(R_ros, up_dir_body).squeeze(2)
        
        # 转换所有向量到 PyTorch3D World 坐标系
        p_cam_pt3d = transform_pos_ros2pt3d(p_cam_ros)
        p_at_pt3d = transform_pos_ros2pt3d(p_target_ros)
        up_vec_pt3d = transform_pos_ros2pt3d(up_vec_ros)
        
        # 使用 look_at_view_transform 计算 World-to-View 变换
        R_view, T_view = look_at_view_transform(
            eye=p_cam_pt3d,
            at=p_at_pt3d,
            up=up_vec_pt3d,
            device=device
        )
        
        return R_view, T_view

    @staticmethod
    def clean_depth_map(depth, min_dist=0.2, max_dist=10.0):
        """
        清洗深度图，移除无效或超出范围的深度值。

        将背景像素（通常为 -1）和超出距离范围的像素设置为 NaN，便于后续处理。

        Args:
            depth (torch.Tensor or numpy.ndarray): 输入深度图。
            min_dist (float, optional): 最小有效距离，默认 0.2。
            max_dist (float, optional): 最大有效距离，默认 10.0。

        Returns:
            torch.Tensor or numpy.ndarray: 清洗后的深度图，形状与输入相同。

        Raises:
            TypeError: 如果 depth 不是张量或数组。
        """
        if torch.is_tensor(depth):
            depth = depth.clone()
        else:
            depth = depth.copy()
        # 背景通常为 -1
        invalid_mask = (depth == -1) | (depth > max_dist) | (depth < min_dist)
        depth[invalid_mask] = float('nan')
        return depth


class DroneRendererVariant:
    """
    共享 mesh/lights/obstacle_pcd 的轻量渲染器变体。

    由 DroneRenderer.create_variant() 创建，适用于同一场景下
    不同分辨率/FOV 的渲染需求（如高分辨率可视化输出）。
    """

    def __init__(self, parent, image_size, focal_length, z_clip_value=0.3,
                 max_faces_per_bin=50000):
        self.parent = parent
        self.device = parent.device
        self.image_size = image_size
        H, W = image_size

        fl = float(focal_length)
        self.focal_length = torch.tensor([[fl, fl]], dtype=torch.float32, device=self.device)
        self.principal_point = torch.tensor(
            [[W / 2.0, H / 2.0]], dtype=torch.float32, device=self.device)
        self.image_size_tensor = torch.tensor(
            [[H, W]], dtype=torch.float32, device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            z_clip_value=z_clip_value,
            max_faces_per_bin=max_faces_per_bin,
        )
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, lights=parent.lights)

    @property
    def mesh(self):
        return self.parent.mesh

    @property
    def render_mesh(self):
        return self.parent.render_mesh

    @property
    def obstacle_pcd(self):
        return self.parent.obstacle_pcd

    @property
    def full_obstacle_pcd(self):
        return self.parent.full_obstacle_pcd

    @property
    def lights(self):
        return self.parent.lights

    def compute_view_matrix(self, p_ros, R_ros, camera_pitch_deg=None,
                            cam_offset_body=None, cam_mount_R=None):
        """委托给 parent。"""
        return self.parent.compute_view_matrix(
            p_ros, R_ros, camera_pitch_deg, cam_offset_body, cam_mount_R)

    @torch.no_grad()
    def render(self, R, T, return_tensor=True, return_rgb=True, return_depth=True, **_kw):
        """渲染 (默认 no_grad，用于可视化)。"""
        if not torch.is_tensor(R):
            R = torch.tensor(R, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(T):
            T = torch.tensor(T, device=self.device, dtype=torch.float32)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() == 1:
            T = T.unsqueeze(0)

        cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            image_size=self.image_size_tensor,
            in_ndc=False, R=R, T=T, device=self.device,
        )
        mesh = self.parent.render_mesh.extend(len(cameras))
        fragments = self.rasterizer(mesh, cameras=cameras)

        rgb_images = None
        depth_maps = None
        if return_rgb:
            images = self.shader(fragments, mesh, cameras=cameras)
            rgb_images = images[..., :3]
        if return_depth:
            depth_maps = fragments.zbuf[..., 0]

        if not return_tensor:
            if rgb_images is not None:
                rgb_images = rgb_images.detach().cpu().numpy()
            if depth_maps is not None:
                depth_maps = depth_maps.detach().cpu().numpy()
        return rgb_images, depth_maps


# ============================================================
# 无人机网格工具
# ============================================================

def compute_drone_safety_radius(drone_mesh_path, device=None, aero_margin=0.05):
    """
    从无人机网格文件计算安全半径 = 包围球半径 + 空气动力学干扰边距。

    Args:
        drone_mesh_path: .obj 文件路径。
        device: 计算设备。
        aero_margin: 空气动力学干扰边距 (m)，默认 0.05。

    Returns:
        float: 安全半径 (米)。
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = load_objs_as_meshes([drone_mesh_path], device=device)
    verts = mesh.verts_packed()  # (V, 3)
    centroid = verts.mean(dim=0)
    bounding_radius = (verts - centroid).norm(dim=1).max().item()
    return bounding_radius + aero_margin


def load_drone_mesh(drone_mesh_path, device=None, scale=1.0, max_faces=500):
    """
    加载无人机网格并返回 Meshes 对象 + 包围球信息。

    面片数超过 max_faces 时自动简化，避免多机渲染时面片爆炸。
    例: 原始 drone.obj 有 8924 面，16 机 = 142K 面，简化到 500 面后仅 8K 面。

    Args:
        drone_mesh_path: .obj 文件路径。
        device: 计算设备。
        scale: 缩放因子 (随机化用)。
        max_faces: 最大面片数，超过则自动简化。默认 500。

    Returns:
        tuple: (mesh, centroid, bounding_radius)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = load_objs_as_meshes([drone_mesh_path], device=device)
    if mesh.textures is None:
        verts = mesh.verts_list()[0]
        verts_rgb = torch.ones_like(verts)[None]
        mesh.textures = TexturesVertex(verts_features=verts_rgb)
    # 包围球信息必须在简化之前计算，保证物理碰撞半径不受简化影响
    verts = mesh.verts_packed()
    centroid = verts.mean(dim=0)
    bounding_radius = float((verts - centroid).norm(dim=1).max().item())
    # 面片简化（渲染用低模，碰撞半径保持原始精度）
    n_faces = mesh.faces_packed().shape[0]
    if max_faces and max_faces > 0 and n_faces > max_faces:
        from scene_generator import _decimate_mesh
        mesh = _decimate_mesh(mesh, max_faces, device=device)
        n_after = mesh.faces_packed().shape[0]
        print(f"  [DroneMesh] 简化: {n_faces} → {n_after} faces "
              f"({(1 - n_after/n_faces)*100:.0f}% 减少)")
    return mesh, centroid, bounding_radius


def transform_pos_ros2pt3d(pos):
    """
    将位置向量从 ROS (ENU) 坐标系转换到 PyTorch3D 渲染坐标系。

    ROS (ENU): +X: East, +Y: North, +Z: Up
    PyTorch3D World: +X: West, +Y: Up, +Z: North
    映射: (x, y, z) -> (-x, z, y)

    Args:
        pos (torch.Tensor): ROS 坐标系下的位置，形状 (..., 3)。

    Returns:
        torch.Tensor: PyTorch3D 坐标系下的位置，形状 (..., 3)。

    Raises:
        ValueError: 如果 pos 的最后一个维度不是 3。
    """
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    return torch.stack([-x, z, y], dim=-1)

def transform_rot_ros2pt3d(R):
    """
    将旋转矩阵 (Body -> World) 从 ROS 坐标系转换到 PyTorch3D 坐标系。

    假设 Body Frame 在两种定义下相对网格一致。通过变换每个基向量来实现转换。

    Args:
        R (torch.Tensor): ROS 坐标系下的旋转矩阵，形状 (..., 3, 3)。

    Returns:
        torch.Tensor: PyTorch3D 坐标系下的旋转矩阵，形状 (..., 3, 3)。

    Raises:
        ValueError: 如果 R 的最后两个维度不是 (3, 3)。
    """
    # 变换基向量 (R 的列向量)
    r0 = transform_pos_ros2pt3d(R[..., 0])  # Body X 轴
    r1 = transform_pos_ros2pt3d(R[..., 1])  # Body Y 轴
    r2 = transform_pos_ros2pt3d(R[..., 2])  # Body Z 轴
    return torch.stack([r0, r1, r2], dim=-1)

if __name__ == "__main__":
    # 简单的测试代码：加载模型、渲染并可视化
    import os
    import matplotlib.pyplot as plt
    
    # 假设数据路径
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "sample/sample.obj")
    
    if os.path.exists(obj_filename):
        # 初始化渲染器
        renderer = DroneRenderer(obj_filename)
        
        # 设置固定视角进行测试
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=180, device=renderer.device)
        
        # 渲染图像
        rgb, depth = renderer.render(R, T)
        clean_depth = renderer.clean_depth_map(depth)
        
        print(f"渲染RGB形状: {rgb.shape}")
        print(f"渲染深度图形状: {depth.shape}")
        
        # 可视化 (仅在直接运行时)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        if torch.is_tensor(rgb):
            plt.imshow(rgb.cpu().numpy())
        else:
            plt.imshow(rgb)
        plt.title("RGB")
        plt.subplot(1, 2, 2)
        if torch.is_tensor(clean_depth):
            plt.imshow(clean_depth.cpu().numpy(), cmap="viridis")
        else:
            plt.imshow(clean_depth, cmap="viridis")
        plt.title("Depth")
        plt.show()
    else:
        print(f"测试文件未找到: {obj_filename}")
