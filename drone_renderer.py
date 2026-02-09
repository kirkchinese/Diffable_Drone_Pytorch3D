"""
无人机渲染器模块

本模块提供了一个基于 PyTorch3D 的可微分 3D 渲染器类 `DroneRenderer`，
用于加载无人机网格模型并生成 RGB 图像和深度图，支持批量渲染和坐标系转换。

主要功能：
- 加载和渲染 .obj 格式的 3D 网格模型。
- 计算相机视图矩阵，支持 ROS 坐标系到 PyTorch3D 坐标系的转换。
- 生成 RGB 和深度图像，支持梯度传播。
- 提供障碍物距离计算和深度图清洗功能。

依赖库：
- torch: PyTorch 张量计算和自动微分。
- pytorch3d: 3D 渲染和几何操作库。
- numpy: 数值计算和数组操作。
- math: 数学函数和常量。

作者: KirkChinese
版本: 你记了吗？
日期: 2026-01-10
"""

import torch
import numpy as np
import math
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
                 z_clip_value=0.01):
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
            z_clip_value (float, optional): 近平面裁剪值，默认 0.01。
                将跨越 z=0 的三角形沿近平面裁剪，防止投影到无穷大后消失。
                这是解决"大面片消失"问题的正确方案，比细分更高效。
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
        # bin_size=0 → naive 光栅化：跳过 coarse binning 阶段，
        # 对小分辨率 (48x64) 速度相当且不会 bin 溢出。
        # 场景障碍物多时（20-40 个，数万面片），默认 binning 容量不够会报
        # "Bin size was too small in the coarse rasterization phase" 错误。
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            perspective_correct=True,
            z_clip_value=z_clip_value,  # 近平面裁剪: 解决跨越 z=0 的三角形投影异常
            bin_size=0,  # naive 光栅化，避免 bin 溢出
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
            self._extended_mesh_cache = self.mesh.extend(bs)
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

    def compute_view_matrix(self, p_ros, R_ros, camera_pitch_deg=-10.0, cam_offset_body=None):
        """
        将无人机在 ROS 坐标系下的状态转换为 PyTorch3D 渲染器所需的外参 (R, T)。

        该函数自动处理坐标系转换、相机安装偏移和 LookAt 变换计算。

        Args:
            p_ros (torch.Tensor): 无人机位置 (ROS World Frame ENU)，形状 (B, 3)。
            R_ros (torch.Tensor): 无人机姿态旋转矩阵 (Body -> ROS World)，形状 (B, 3, 3)。
            camera_pitch_deg (float, optional): 相机安装俯仰角 (度)，默认 -10.0。
                正值表示相机向下倾斜 (俯视)，负值表示仰视。
            cam_offset_body (list or torch.Tensor, optional): 相机在机体坐标系下的偏移 [x, y, z]，默认 [0.1, 0, 0]。
            
        Returns:
            tuple: (R_view, T_view)
                - R_view (torch.Tensor): World-to-View 旋转矩阵，形状 (B, 3, 3)。
                - T_view (torch.Tensor): World-to-View 平移向量，形状 (B, 3)。

        Raises:
            ValueError: 如果输入张量的形状不符合要求。
        """
        B = p_ros.shape[0]
        device = p_ros.device
        
        # 计算相机在 ROS World 下的位置
        # Camera Offset in Body Frame: Default [0.1, 0, 0]
        if cam_offset_body is None:
            cam_offset_body = [0.1, 0.0, 0.0]
        
        if not torch.is_tensor(cam_offset_body):
            cam_offset_body = torch.tensor(cam_offset_body, device=device, dtype=torch.float32)
        
        # Ensure shape (B, 3, 1)
        if cam_offset_body.dim() == 1:
            cam_offset_body = cam_offset_body.view(1, 3, 1).repeat(B, 1, 1)
        elif cam_offset_body.dim() == 2:
            cam_offset_body = cam_offset_body.unsqueeze(2)
            
        p_cam_ros = p_ros + torch.bmm(R_ros, cam_offset_body).squeeze(2)
        
        # 定义相机相对于机体的旋转矩阵 (R_mount)
        # 使用旋转矩阵标准化表达，便于扩展 (如增加Yaw/Roll偏置)
        # 这里的 pitch 是绕机体 Y 轴旋转。正 Pitch 代表向下看，即 Forward 向量向 -Z 偏转。
        # 支持 per-sample 的 Tensor 输入（用于训练时的相机角度随机化）
        if isinstance(camera_pitch_deg, (int, float)):
            pitch_rad = torch.full((B,), camera_pitch_deg * math.pi / 180, device=device)
        else:
            # Tensor 输入 (B,) — per-sample 俯仰角
            pitch_rad = camera_pitch_deg.to(device=device, dtype=torch.float32) * (math.pi / 180)
        c = torch.cos(pitch_rad)
        s = torch.sin(pitch_rad)
        zeros = torch.zeros(B, device=device)
        ones = torch.ones(B, device=device)
        R_mount = torch.stack([
            c,     zeros, s,
            zeros, ones,  zeros,
            -s,    zeros, c
        ], dim=-1).reshape(B, 3, 3)
        
        # 计算 Look At 指向向量和 Up 向量 (在机体坐标系下)
        forward_canonical = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1).repeat(B, 1, 1)
        up_canonical = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1).repeat(B, 1, 1)
        
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
