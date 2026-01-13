"""
动态场景无人机渲染器模块

扩展自 DroneRenderer，支持：
1. 多障碍物场景合成
2. 障碍物动态移动/旋转
3. 随机化障碍物配置

作者: KirkChinese + AI Assistant
日期: 2026-01-12
"我这个人类来解释一下情况防止我后面忘记怎么回事。
这个模块继承自 DroneRenderer，增加了对动态障碍物的支持。
主要新增了 DynamicObstacle 类来封装单个障碍物的网格和状态，
以及 DynamicSceneRenderer 类来管理多个动态障碍物并进行场景合成渲染。
原因是我不知道怎么实现一个动态的场景渲染器，所以我让AI帮我写了这个模块。
AI是这样教导这个模块的使用的：
<start>
## 总结

我为你创建了动态场景支持系统，主要包含：

### 新文件

1. **drone_renderer_dynamic.py** - 动态场景渲染器
2. **dynamic_scene_test.ipynb** - 测试 notebook

### 核心类

| 类名 | 功能 |
|------|------|
| `DynamicObstacle` | 单个动态障碍物（位置/速度/旋转） |
| `DynamicSceneRenderer` | 继承 DroneRenderer，支持多障碍物合成 |
| `DynamicDroneSimulator` | 包装 DroneSimulator，自动更新障碍物 |

### 关键特性

1. **多 Mesh 合成**：使用 `join_meshes_as_scene` 将静态背景 + 动态障碍物合并
2. **基本几何体**：内置 `sphere` 和 `cube`，可直接添加
3. **自定义 Mesh**：支持加载任意 .obj 文件作为障碍物
4. **障碍物运动**：支持线速度和角速度
5. **点云同步更新**：碰撞检测点云实时更新
6. **随机化接口**：`randomize_obstacles()` 一键生成随机场景

### 使用示例

```python
# 创建动态渲染器
renderer = DynamicSceneRenderer(
    static_mesh_path='./data/sample/sample4.obj',
    device=device
)

# 添加移动球体
renderer.add_primitive_obstacle(
    primitive_type='sphere',
    position=torch.tensor([1.0, 0.0, 0.0]),
    velocity=torch.tensor([0.0, 0.5, 0.0]),  # Y方向移动
    scale=0.3
)

# 在训练循环中
for step in range(timesteps):
    renderer.step_obstacles(dt)  # 更新障碍物位置
    rgb, depth = renderer.render(...)  # 渲染
```

### 与训练集成

可以在每个 episode 开始时调用 `randomize_obstacles()` 来生成不同的动态场景，让无人机学习应对变化的环境！

<end>
咱只能说AI写的代码质量还不错，逻辑清晰，注释详细，基本符合我的需求。虽然我还没做测试，但是就先这么放着，等我测试完训练的效果再说吧。
回头我再整合到训练代码里去。
"
"""

import torch
import numpy as np
import math
from typing import List, Optional, Tuple, Union

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    PerspectiveCameras,
    PointLights,
    TexturesVertex,
)
from pytorch3d.transforms import Transform3d, Rotate, Translate

from drone_renderer import DroneRenderer


class DynamicObstacle:
    """
    动态障碍物类
    
    封装单个障碍物的网格、位置、速度等状态
    """
    def __init__(self, 
                 mesh: Meshes,
                 position: torch.Tensor = None,
                 velocity: torch.Tensor = None,
                 rotation: torch.Tensor = None,
                 angular_velocity: torch.Tensor = None,
                 scale: float = 1.0,
                 device: torch.device = None):
        """
        Args:
            mesh: PyTorch3D Meshes 对象
            position: 位置 (3,) 默认原点
            velocity: 线速度 (3,) 默认静止
            rotation: 旋转矩阵 (3, 3) 默认单位矩阵
            angular_velocity: 角速度 (3,) 默认不旋转
            scale: 缩放比例
            device: 计算设备
        """
        self.device = device or mesh.device
        self.original_mesh = mesh
        self.scale = scale
        
        # 状态初始化 - 确保所有 tensor 都在正确的设备上
        self.position = position.to(self.device) if position is not None else torch.zeros(3, device=self.device)
        self.velocity = velocity.to(self.device) if velocity is not None else torch.zeros(3, device=self.device)
        self.rotation = rotation.to(self.device) if rotation is not None else torch.eye(3, device=self.device)
        self.angular_velocity = angular_velocity.to(self.device) if angular_velocity is not None else torch.zeros(3, device=self.device)
        
        # 缓存变换后的网格
        self._transformed_mesh = None
        self._needs_update = True
        
    def step(self, dt: float):
        """
        更新障碍物状态
        
        Args:
            dt: 时间步长
        """
        # 更新位置
        self.position = self.position + self.velocity * dt
        
        # 更新旋转 (简化的欧拉积分)
        if self.angular_velocity.norm() > 1e-6:
            angle = self.angular_velocity.norm() * dt
            axis = self.angular_velocity / (self.angular_velocity.norm() + 1e-8)
            # Rodrigues 公式
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=self.device)
            delta_R = torch.eye(3, device=self.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
            self.rotation = delta_R @ self.rotation
            
        self._needs_update = True
        
    def get_transformed_mesh(self) -> Meshes:
        """获取变换后的网格"""
        if self._needs_update or self._transformed_mesh is None:
            self._transformed_mesh = self._apply_transform()
            self._needs_update = False
        return self._transformed_mesh
    
    def _apply_transform(self) -> Meshes:
        """应用变换到网格"""
        verts = self.original_mesh.verts_packed()  # (V, 3)
        
        # 应用缩放
        verts = verts * self.scale
        
        # 应用旋转
        verts = verts @ self.rotation.T
        
        # 应用平移
        verts = verts + self.position
        
        # 重建 mesh
        faces = self.original_mesh.faces_packed()
        textures = self.original_mesh.textures
        
        new_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        return new_mesh
    
    def set_position(self, position: torch.Tensor):
        """设置位置"""
        self.position = position.to(self.device)
        self._needs_update = True
        
    def set_velocity(self, velocity: torch.Tensor):
        """设置速度"""
        self.velocity = velocity.to(self.device)


class DynamicSceneRenderer(DroneRenderer):
    """
    动态场景渲染器
    
    继承自 DroneRenderer，增加动态障碍物支持
    """
    
    def __init__(self,
                 static_mesh_path: str,
                 device: torch.device = None,
                 image_size: Tuple[int, int] = (480, 640),
                 focal_length: float = 500.0,
                 principal_point: Tuple[float, float] = None,
                 lights_location: List[List[float]] = [[0.0, 0.0, -3.0]],
                 num_samples: int = 20000,
                 subdivide_times: int = 3):
        """
        初始化动态场景渲染器
        
        Args:
            static_mesh_path: 静态背景网格路径
            subdivide_times: 网格细分次数，默认 3。用于解决大面片渲染问题。
            其他参数同 DroneRenderer
        """
        # 调用父类初始化
        super().__init__(
            mesh_path=static_mesh_path,
            device=device,
            image_size=image_size,
            focal_length=focal_length,
            principal_point=principal_point,
            lights_location=lights_location,
            num_samples=num_samples,
            subdivide_times=subdivide_times
        )
        
        # 保存静态网格
        self.static_mesh = self.mesh
        self.static_pcd = self.obstacle_pcd.clone()
        
        # 动态障碍物列表
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
        # 动态障碍物点云 (用于碰撞检测)
        self.dynamic_pcd = None
        
    def add_obstacle(self, 
                     mesh_or_path: Union[Meshes, str],
                     position: torch.Tensor = None,
                     velocity: torch.Tensor = None,
                     scale: float = 1.0,
                     num_samples: int = 1000,
                     subdivide_times: int = 0) -> int:
        """
        添加动态障碍物
        
        Args:
            mesh_or_path: Meshes 对象或 .obj 文件路径
            position: 初始位置
            velocity: 初始速度
            scale: 缩放比例
            num_samples: 点云采样数
            subdivide_times: 网格细分次数，默认为 0（动态障碍物通常较小，不需要细分）
            
        Returns:
            障碍物索引
        """
        # 加载网格
        if isinstance(mesh_or_path, str):
            mesh = self._load_mesh(mesh_or_path, subdivide_times=subdivide_times)
        else:
            mesh = mesh_or_path
            
        # 创建动态障碍物
        obstacle = DynamicObstacle(
            mesh=mesh,
            position=position,
            velocity=velocity,
            scale=scale,
            device=self.device
        )
        
        self.dynamic_obstacles.append(obstacle)
        
        # 更新合成场景
        self._update_scene()
        
        return len(self.dynamic_obstacles) - 1
    
    def add_primitive_obstacle(self,
                               primitive_type: str = 'sphere',
                               position: torch.Tensor = None,
                               velocity: torch.Tensor = None,
                               scale: float = 1.0,
                               **kwargs) -> int:
        """
        添加基本几何体障碍物
        
        Args:
            primitive_type: 'sphere', 'cube', 'cylinder'
            position: 初始位置
            velocity: 初始速度
            scale: 缩放比例
            
        Returns:
            障碍物索引
        """
        from pytorch3d.utils import ico_sphere
        
        if primitive_type == 'sphere':
            level = kwargs.get('level', 2)
            mesh = ico_sphere(level=level, device=self.device)
        elif primitive_type == 'cube':
            # 创建立方体网格
            mesh = self._create_cube_mesh()
        else:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")
            
        # 添加默认纹理
        verts = mesh.verts_list()[0]
        color = kwargs.get('color', [0.8, 0.2, 0.2])  # 默认红色
        verts_rgb = torch.tensor(color, device=self.device).expand(verts.shape[0], 3)[None]
        mesh.textures = TexturesVertex(verts_features=verts_rgb)
        
        return self.add_obstacle(mesh, position, velocity, scale)
    
    def _create_cube_mesh(self) -> Meshes:
        """创建单位立方体网格"""
        # 立方体顶点
        verts = torch.tensor([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], device=self.device, dtype=torch.float32)
        
        # 立方体面 (三角形)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # 前
            [4, 6, 5], [4, 7, 6],  # 后
            [0, 4, 5], [0, 5, 1],  # 下
            [2, 6, 7], [2, 7, 3],  # 上
            [0, 3, 7], [0, 7, 4],  # 左
            [1, 5, 6], [1, 6, 2],  # 右
        ], device=self.device, dtype=torch.int64)
        
        return Meshes(verts=[verts], faces=[faces])
    
    def remove_obstacle(self, index: int):
        """移除指定障碍物"""
        if 0 <= index < len(self.dynamic_obstacles):
            del self.dynamic_obstacles[index]
            self._update_scene()
            
    def clear_dynamic_obstacles(self):
        """清除所有动态障碍物"""
        self.dynamic_obstacles.clear()
        self._update_scene()
        
    def step_obstacles(self, dt: float):
        """
        更新所有动态障碍物状态
        
        Args:
            dt: 时间步长
        """
        for obstacle in self.dynamic_obstacles:
            obstacle.step(dt)
        self._update_scene()
        
    def set_obstacle_position(self, index: int, position: torch.Tensor):
        """设置指定障碍物位置"""
        if 0 <= index < len(self.dynamic_obstacles):
            self.dynamic_obstacles[index].set_position(position)
            self._update_scene()
            
    def set_obstacle_velocity(self, index: int, velocity: torch.Tensor):
        """设置指定障碍物速度"""
        if 0 <= index < len(self.dynamic_obstacles):
            self.dynamic_obstacles[index].set_velocity(velocity)
    
    def _update_scene(self):
        """更新合成场景"""
        if len(self.dynamic_obstacles) == 0:
            # 没有动态障碍物，使用静态网格
            self.mesh = self.static_mesh
            self.obstacle_pcd = self.static_pcd
            self.dynamic_pcd = None
        else:
            # 合成场景
            meshes_list = [self.static_mesh]
            dynamic_pcds = []
            
            for obstacle in self.dynamic_obstacles:
                transformed_mesh = obstacle.get_transformed_mesh()
                meshes_list.append(transformed_mesh)
                
                # 采样动态障碍物点云
                pcd = sample_points_from_meshes(transformed_mesh, num_samples=500)
                dynamic_pcds.append(pcd)
            
            # 合并所有网格
            self.mesh = join_meshes_as_scene(meshes_list)
            
            # 合并点云
            if dynamic_pcds:
                self.dynamic_pcd = torch.cat(dynamic_pcds, dim=1)  # (1, N_total, 3)
                self.obstacle_pcd = torch.cat([self.static_pcd, self.dynamic_pcd], dim=1)
            else:
                self.obstacle_pcd = self.static_pcd
                
    def randomize_obstacles(self, 
                            num_obstacles: int = 3,
                            position_range: Tuple[float, float] = (-2.0, 2.0),
                            velocity_range: Tuple[float, float] = (-0.5, 0.5),
                            scale_range: Tuple[float, float] = (0.3, 1.0)):
        """
        随机化动态障碍物配置
        
        Args:
            num_obstacles: 障碍物数量
            position_range: 位置范围
            velocity_range: 速度范围
            scale_range: 缩放范围
        """
        self.clear_dynamic_obstacles()
        
        for _ in range(num_obstacles):
            # 随机位置
            pos = torch.rand(3, device=self.device) * (position_range[1] - position_range[0]) + position_range[0]
            
            # 随机速度
            vel = torch.rand(3, device=self.device) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
            
            # 随机缩放
            scale = torch.rand(1, device=self.device).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
            
            # 随机选择形状
            primitive = np.random.choice(['sphere', 'cube'])
            color = torch.rand(3).tolist()
            
            self.add_primitive_obstacle(
                primitive_type=primitive,
                position=pos,
                velocity=vel,
                scale=scale,
                color=color
            )


class DynamicDroneSimulator:
    """
    支持动态障碍物的无人机仿真器
    
    包装 DynamicSceneRenderer，提供与 DroneSimulator 兼容的接口
    """
    
    def __init__(self, 
                 base_simulator,  # DroneSimulator 实例
                 dynamic_renderer: DynamicSceneRenderer = None):
        """
        Args:
            base_simulator: 基础的 DroneSimulator
            dynamic_renderer: 动态场景渲染器 (可选，如果不传则从 base 创建)
        """
        self.base = base_simulator
        
        if dynamic_renderer is not None:
            self.renderer = dynamic_renderer
            # 替换基础仿真器的渲染器
            self.base.renderer = dynamic_renderer
        else:
            # 从基础仿真器创建动态渲染器
            self.renderer = DynamicSceneRenderer(
                static_mesh_path=self.base.renderer.mesh_path if hasattr(self.base.renderer, 'mesh_path') else './data/sample/sample4.obj',
                device=self.base.device,
                image_size=self.base.renderer.image_size,
                focal_length=self.base.renderer.focal_length[0][0],
            )
            self.base.renderer = self.renderer
            
    def step(self, *args, **kwargs):
        """执行仿真步进，同时更新动态障碍物"""
        # 更新动态障碍物
        dt = kwargs.get('dt', self.base.dt)
        self.renderer.step_obstacles(dt)
        
        # 调用基础仿真器的 step
        return self.base.step(*args, **kwargs)
    
    def reset(self):
        """重置仿真环境，可选择是否随机化动态障碍物"""
        self.base.reset()
        
    def randomize_dynamic_obstacles(self, **kwargs):
        """随机化动态障碍物"""
        self.renderer.randomize_obstacles(**kwargs)
        
    def __getattr__(self, name):
        """代理到基础仿真器"""
        return getattr(self.base, name)
