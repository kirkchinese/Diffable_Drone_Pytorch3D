"""
可微分3D激光雷达传感器模拟

基于 PyTorch3D 渲染的深度缓冲，通过角度域重采样将透视投影深度图
转换为球面坐标系下的距离测量值，模拟真实多线激光雷达的空间采集模式。

关键设计:
  - 零额外渲染开销：复用已有深度图，仅需一次 grid_sample
  - GPU 全程运行：采样网格预计算后所有操作在 GPU 上完成
  - 可微分：bilinear 插值保持梯度流

技术细节:
  - 输入: 透视投影深度图 (B, H, W)，zbuf 值 = 沿光轴距离
  - 输出: 距离图像 (B, num_beams, points_per_beam)，值 = 沿光束的真实距离
  - 转换: range = depth / cos(θ)，θ 为光束偏离光轴的角度
"""

import torch
import torch.nn.functional as F


class LiDARSensor:
    """基于深度渲染的多线激光雷达传感器模拟器。

    模拟类似 Velodyne VLP-16 的多线激光扫描仪：
    - 垂直方向分布 num_beams 条扫描线
    - 每条线在水平方向均匀采样 points_per_beam 个距离值
    - FOV 受限于前视深度相机的视场角

    使用方式::

        lidar = LiDARSensor(num_beams=16, points_per_beam=64)
        lidar.setup(focal_length=38.5, image_height=48, image_width=64)

        # 训练循环中
        range_img = lidar.depth_to_range_image(depth)   # (B, 16, 64)
        model_input = lidar.preprocess(range_img)        # (B, 1, 16, 64)
    """

    def __init__(self,
                 num_beams: int = 16,
                 points_per_beam: int = 64,
                 max_range: float = 24.0,
                 min_range: float = 0.3,
                 device: str = 'cuda'):
        self.num_beams = num_beams
        self.points_per_beam = points_per_beam
        self.max_range = max_range
        self.min_range = min_range
        self.device = device

        # 延迟初始化：需要相机内参
        self._grid: torch.Tensor | None = None
        self._cos_correction: torch.Tensor | None = None
        self._focal_length: float = 0.0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def setup(self, focal_length: float,
              image_height: int, image_width: int) -> 'LiDARSensor':
        """根据相机内参预计算 LiDAR 采样网格。

        LiDAR 光束在角度空间均匀分布，深度图在像素空间均匀分布。
        此方法预计算每条光束对应的亚像素坐标，后续帧直接复用。

        Args:
            focal_length: 像素焦距 (fx ≈ fy)
            image_height: 深度图高度 (H)
            image_width:  深度图宽度 (W)

        Returns:
            self（支持链式调用）
        """
        self._focal_length = focal_length
        fx = fy = focal_length
        cx = image_width / 2.0
        cy = image_height / 2.0

        # 计算在 grid_sample(align_corners=True) 下确保 grid ∈ [-1,1] 的
        # 最大对称光束角度。
        # pixel = fx * tan(az) + cx, grid_x = 2*pixel/(W-1) - 1
        # 需要 pixel ∈ [0, W-1]：
        #   tan(az) ∈ [-cx/fx, (W-1-cx)/fx]
        # 对称化：取两侧绝对值较小者
        az_half = min(
            torch.atan(torch.tensor(cx / fx)).item(),
            torch.atan(torch.tensor((image_width - 1 - cx) / fx)).item(),
        )
        el_half = min(
            torch.atan(torch.tensor(cy / fy)).item(),
            torch.atan(torch.tensor((image_height - 1 - cy) / fy)).item(),
        )

        # LiDAR 光束角度：在安全 FOV 内均匀采样
        az = torch.linspace(-az_half, az_half,
                            self.points_per_beam)
        el = torch.linspace(el_half, -el_half,
                            self.num_beams)          # 上→下

        el_grid, az_grid = torch.meshgrid(el, az, indexing='ij')  # (V, H_l)

        # 角度 → 像素坐标
        u = fx * az_grid.tan() + cx
        v = fy * el_grid.tan() + cy

        # 归一化到 [-1, 1]（grid_sample 要求，align_corners=True）
        u_norm = 2.0 * u / (image_width - 1) - 1.0
        v_norm = 2.0 * v / (image_height - 1) - 1.0

        # 采样网格 (1, V, H_l, 2)
        self._grid = (torch.stack([u_norm, v_norm], dim=-1)
                      .unsqueeze(0).to(self.device))

        # 深度→距离修正因子: range = depth / cos(θ)
        # cos(θ) = 1 / sqrt(1 + tan²(az) + tan²(el))
        cos_theta = 1.0 / torch.sqrt(
            1.0 + az_grid.tan() ** 2 + el_grid.tan() ** 2)
        self._cos_correction = cos_theta.unsqueeze(0).to(self.device)

        self._image_height = image_height
        self._image_width = image_width
        return self

    # ------------------------------------------------------------------
    # 核心转换
    # ------------------------------------------------------------------

    def depth_to_range_image(self, depth: torch.Tensor) -> torch.Tensor:
        """将透视投影深度图转换为 LiDAR 距离图像。

        Args:
            depth: (B, H, W) 渲染深度图，zbuf 值，背景 = -1

        Returns:
            range_img: (B, num_beams, points_per_beam)
                       有效值 ∈ [min_range, max_range]，背景/超范围 = -1
        """
        assert self._grid is not None, "必须先调用 setup()"

        B = depth.shape[0]

        # grid_sample 需要 (B, C, H, W) 输入
        depth_4d = depth.unsqueeze(1)                         # (B, 1, H, W)
        grid = self._grid.expand(B, -1, -1, -1)              # (B, V, H_l, 2)

        # 双线性插值采样
        sampled = F.grid_sample(
            depth_4d, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(1)                                          # (B, V, H_l)

        # 标记无效区域
        invalid = sampled <= 0

        # 深度 → 距离修正
        range_img = sampled / self._cos_correction            # (B, V, H_l)

        # 范围裁剪 + 标记超限
        out_of_range = (range_img < self.min_range) | (range_img > self.max_range)
        range_img = range_img.clamp(self.min_range, self.max_range)
        range_img = range_img.masked_fill(invalid | out_of_range, -1.0)

        return range_img

    def depth_to_point_cloud(self, depth: torch.Tensor) -> torch.Tensor:
        """将深度图转换为 LiDAR 光束交点的3D点云（相机坐标系）。

        用于可视化和调试，不用于训练输入。

        Args:
            depth: (B, H, W) 深度图

        Returns:
            points: (B, num_beams * points_per_beam, 3)
                    无效点坐标 = (0, 0, 0)
        """
        range_img = self.depth_to_range_image(depth)          # (B, V, H_l)
        B = range_img.shape[0]

        # 光束方向向量 (球面坐标 → 笛卡尔)
        hfov_half = torch.atan(torch.tensor(
            self._image_width / 2.0 / self._focal_length))
        vfov_half = torch.atan(torch.tensor(
            self._image_height / 2.0 / self._focal_length))
        az = torch.linspace(-hfov_half.item(), hfov_half.item(),
                            self.points_per_beam, device=self.device)
        el = torch.linspace(vfov_half.item(), -vfov_half.item(),
                            self.num_beams, device=self.device)
        el_grid, az_grid = torch.meshgrid(el, az, indexing='ij')

        # PyTorch3D 相机坐标系: +X 右, +Y 下, +Z 前
        dx = az_grid.sin() * el_grid.cos()
        dy = -el_grid.sin()
        dz = az_grid.cos() * el_grid.cos()
        dirs = torch.stack([dx, dy, dz], dim=-1)              # (V, H_l, 3)
        dirs = dirs.reshape(1, -1, 3)                          # (1, N, 3)

        valid = range_img > 0                                  # (B, V, H_l)
        range_flat = range_img.reshape(B, -1, 1)              # (B, N, 1)
        points = range_flat * dirs                             # (B, N, 3)
        points = points.masked_fill(
            ~valid.reshape(B, -1, 1).expand_as(points), 0.0)

        return points

    # ------------------------------------------------------------------
    # 预处理（与 preprocess_depth_for_model 对齐）
    # ------------------------------------------------------------------

    def preprocess(self, range_img: torch.Tensor,
                   noise_std: float = 0.0) -> torch.Tensor:
        """将距离图像归一化为模型输入格式。

        变换与深度图预处理一致：inverse distance + 背景置零。

        Args:
            range_img: (B, num_beams, points_per_beam)，背景 = -1
            noise_std: 测量噪声标准差

        Returns:
            (B, 1, num_beams, points_per_beam) 归一化后的距离图输入
        """
        bg_mask = range_img < 0
        x = range_img.clamp(self.min_range, self.max_range)
        x = 3.0 / x - 0.6              # inverse distance, 同 depth 预处理
        x = x.masked_fill(bg_mask, 0.0)
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x.unsqueeze(1)           # (B, 1, V, H_l)
