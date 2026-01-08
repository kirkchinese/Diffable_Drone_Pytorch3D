import torch
import numpy as np
import math
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
    def __init__(self, mesh_path, device=None, image_size=(480, 640), focal_length=500.0):
        """
        [渲染器类] 负责加载无人机网格模型并生成 RGB 和深度图。
        使用 PyTorch3D 的光栅化渲染管线。
        
        Args:
            mesh_path (str): .obj 模型文件的绝对路径或相对路径。
            device (torch.device, optional): 计算设备 (cuda/cpu)。如果为 None，自动检测。
            image_size (tuple): 输出图像大小 (Height, Width)。默认 (480, 640)。
            focal_length (float or tuple): 相机焦距 (像素单位)。
                                           如果为 float，假设 fx=fy。
                                           默认 500.0。
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.image_size = image_size # (H, W)
        self.H, self.W = image_size
        
        # 处理焦距
        if isinstance(focal_length, (float, int)):
            self.focal_length = ((focal_length, focal_length),)
        else:
            self.focal_length = (focal_length,)
            
        self.principal_point = ((self.W/2, self.H/2),)
        
        # 加载网格
        self.mesh = self._load_mesh(mesh_path)
        
        # 光栅化设置
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            perspective_correct=True 
        )
        
        # 初始化光照 (暂时设定为点光源)
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        
        # 初始化组件 (不绑定相机，渲染时动态传入)
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        # SoftPhongShader 需要在 forward 中接收 cameras
        self.shader = SoftPhongShader(device=self.device, lights=self.lights)

    def _load_mesh(self, mesh_path):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        
        # 检查并修复纹理
        if mesh.textures is None:
            # print("模型无纹理，创建默认白色纹理...")
            verts = mesh.verts_list()[0]
            verts_rgb = torch.ones_like(verts)[None] 
            mesh.textures = TexturesVertex(verts_features=verts_rgb)
            
        return mesh

    def update_mesh(self, mesh_path):
        """更换渲染的模型"""
        self.mesh = self._load_mesh(mesh_path)

    def render(self, R, T, return_tensor=False, return_rgb=True, return_depth=True):
        """
        [核心渲染接口] 渲染给定视角的图像和深度图。
        
        注意：此函数直接接收 PyTorch3D 标准的相机外参 (World-to-View)。
        通常建议先调用 `compute_view_matrix` 从 ROS 状态获取这里的 R 和 T。
        
        Args:
            R (torch.Tensor): 旋转矩阵 (World -> View)。Shape: (3, 3) 或 (B, 3, 3)。
            T (torch.Tensor): 平移向量 (World -> View Translation)。Shape: (3,) 或 (B, 3)。
            return_tensor (bool): True 返回 Tensor(带梯度)，False 返回 detached numpy。
            return_rgb (bool): 是否计算并返回 RGB 图像。
            return_depth (bool): 是否计算并返回深度图。
            
        Returns:
            rgb_image, depth_map 的元组。
            如果没有请求某项，对应返回值为 None。
        """
        # 转换为 Tensor
        if not torch.is_tensor(R):
            R = torch.tensor(R, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(T):
            T = torch.tensor(T, device=self.device, dtype=torch.float32)

        # 确保输入维度
        is_batch = True
        if R.dim() == 2: 
            R = R.unsqueeze(0)
            is_batch = False
        if T.dim() == 1: 
            T = T.unsqueeze(0)
        
        # 创建当前帧的相机
        # 注意: R, T 这里应该是 World-to-View 的变换 (PyTorch3D 约定)
        cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            image_size=((self.H, self.W),),
            in_ndc=False, # 使用屏幕像素坐标
            R=R,
            T=T,
            device=self.device
        )
        
        # 扩展 mesh 以匹配 batch size
        meshes = self.mesh.extend(len(cameras))
        fragments = self.rasterizer(meshes, cameras=cameras)
        
        rgb_images = None
        depth_maps = None

        # 仅在需要 RGB 时执行 Shader
        if return_rgb:
            # 着色
            images = self.shader(fragments, meshes, cameras=cameras)
            # 提取 RGB 
            rgb_images = images[..., :3]

        if return_depth:
            depth_maps = fragments.zbuf[..., 0]

        if not is_batch:
            # 如果输入是单个，输出也降维为单个
            if rgb_images is not None:
                rgb_images = rgb_images[0]
            if depth_maps is not None:
                depth_maps = depth_maps[0]

        if return_tensor:
            return rgb_images, depth_maps

        # 转为 numpy (detach)
        rgb_out = None
        depth_out = None
        if rgb_images is not None:
             rgb_out = rgb_images.detach()
        if depth_maps is not None:
             depth_out = depth_maps.detach()
        
        return rgb_out, depth_out

    def compute_view_matrix(self, p_ros, R_ros, camera_pitch_deg=-10.0):
        """
        [跨模块接口] 将无人机在 ROS 坐标系下的状态，转换为 PyTorch3D 渲染器所需的外参 (R, T)。
        
        此函数自动处理了：
        1. 坐标系手性转换 (ROS ENU -> PyTorch3D World)。
        2. 相机相对于机身的安装位置 (Offset) 和安装角度 (Mount Rotation)。
        3. LookAt 变换计算。
        
        Args:
            p_ros (torch.Tensor): 无人机位置 (ROS World Frame Enu)。
                                  Shape: [B, 3]
            R_ros (torch.Tensor): 无人机姿态旋转矩阵 (Body -> ROS World Frame)。
                                  Shape: [B, 3, 3]
            camera_pitch_deg (float): 相机安装俯仰角 (度)。
                                      正值表示相机向下倾斜 (俯视, Pitch Down)。
                                      注意: 参考项目 DiffPhysDrone 通常使用正值表示仰视 (Pitch Up)。
                                      如果要完全复现参考项目行为，请设置 camera_pitch_deg = -10.0。
                                      默认 10.0 度 (俯视)。
            
        Returns:
            tuple: (R_view, T_view)
                - R_view: World-to-View 旋转矩阵 [B, 3, 3]。可直接传给 `render()`。
                - T_view: World-to-View 平移向量 [B, 3]。可直接传给 `render()`。
        """
        B = p_ros.shape[0]
        device = p_ros.device
        
        # 计算相机在 ROS World 下的位置
        # Camera Offset in Body Frame: [0.1, 0, 0]
        cam_offset_body = torch.tensor([0.1, 0.0, 0.0], device=device).view(1, 3, 1).repeat(B, 1, 1)
        p_cam_ros = p_ros + torch.bmm(R_ros, cam_offset_body).squeeze(2)
        
        # 定义相机相对于机体的旋转矩阵 (R_mount)
        # 使用旋转矩阵标准化表达，便于扩展 (如增加Yaw/Roll偏置)
        # 这里的 pitch 是绕机体 Y 轴旋转。正 Pitch 代表向下看，即 Forward 向量向 -Z 偏转。
        pitch_rad = math.radians(camera_pitch_deg)
        c = math.cos(pitch_rad)
        s = math.sin(pitch_rad)
        
        R_mount = torch.tensor([
            [c,   0.0, s],
            [0.0, 1.0, 0.0],
            [-s,  0.0, c]
        ], device=device).view(1, 3, 3).repeat(B, 1, 1)
        
        # 计算 Look At 指向向量和 Up 向量 (在机体坐标系下)
        # 原始 Camera Frame 定义: Forward=+X, Up=+Z (与 Body Frame 一致)
        forward_canonical = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1).repeat(B, 1, 1)
        up_canonical = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1).repeat(B, 1, 1)
        
        # 应用安装旋转
        look_dir_body = torch.bmm(R_mount, forward_canonical)
        up_dir_body = torch.bmm(R_mount, up_canonical)
        
        # 转换到世界坐标系
        # Target Point = Camera Pos + R_drone @ look_dir_body
        # 注意: 这里 look_dir_body 长度为 1，直接作为方向向量
        # 我们让 look_at 点距离相机 1.0 米
        p_target_ros = p_cam_ros + torch.bmm(R_ros, look_dir_body).squeeze(2)
        
        # Up Vector World = R_drone @ up_dir_body
        up_vec_ros = torch.bmm(R_ros, up_dir_body).squeeze(2)
        
        # 转换所有向量到 PyTorch3D World 坐标系
        p_cam_pt3d = transform_pos_ros2pt3d(p_cam_ros)
        p_at_pt3d = transform_pos_ros2pt3d(p_target_ros)
        up_vec_pt3d = transform_pos_ros2pt3d(up_vec_ros)
        
        # 使用 look_at_view_transform 计算 E (World-to-View)
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
        清洗深度图，处理背景和无效值。
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
    
    ROS (ENU): 
        +X: East
        +Y: North
        +Z: Up
    PyTorch3D World (用于渲染): 
        +X: West (Left if looking North) - 注意这里实际上是反向的 East
        +Y: Up
        +Z: North (Into Screen)
        
    Mapping:
        (x, y, z) -> (-x, z, y)
    
    解释:
    这种映射构建了一个 PyTorch3D 世界坐标系，其中：
    - 原点与 ROS 重合。
    - 世界的 +Y 轴朝上 (与 ROS +Z 一致)。
    - 世界的 +Z 轴指向北方 (与 ROS +Y 一致)。
    - 世界的 +X 轴指向西方 (与 ROS -X 一致)。
      (根据右手定则: Y(Up) x Z(North) = X(East/Right). 但 PyTorch3D View 习惯 +X is Left. 
       如果我们看向北 (+Z), East 是右侧. PyTorch3D View +X 是左侧.
       所以 ROS +X (East/Right) 映射为 Pt3D -X (Right in View Frame 意义下).)
    """
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    return torch.stack([-x, z, y], dim=-1)

def transform_rot_ros2pt3d(R):
    """
    将旋转矩阵 (Body -> World) 从 ROS 坐标系转换到 PyTorch3D 坐标系。
    假设 Body Frame 在两种定义下相对网格是一致的 (Mesh is static).
    
    R_pt3d = T_coord @ R_ros
    """
    # R: [..., 3, 3]
    # 变换基向量 (R 的列向量)
    r0 = transform_pos_ros2pt3d(R[..., 0]) # Body X axis in Pt3D World
    r1 = transform_pos_ros2pt3d(R[..., 1]) # Body Y axis in Pt3D World
    r2 = transform_pos_ros2pt3d(R[..., 2]) # Body Z axis in Pt3D World
    return torch.stack([r0, r1, r2], dim=-1)

if __name__ == "__main__":
    # 简单的测试代码
    import os
    import matplotlib.pyplot as plt
    
    # 假设数据路径
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "sample/sample.obj")
    
    if os.path.exists(obj_filename):
        renderer = DroneRenderer(obj_filename)
        
        # 设置视角
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=180, device=renderer.device)
        
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
