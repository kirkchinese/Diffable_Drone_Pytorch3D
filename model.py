import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, img_feat, hx

class Model_bigger(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        # Input is expected to be 48x64 (original resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),  # 1, 48, 64 -> 32, 24, 32
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), # 32, 24, 32 -> 64, 12, 16 
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), # 64, 12, 16 -> 128, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), # 128, 6, 8 -> 256, 3, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(256*3*4, 256, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 256)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(256, 256)
        self.fc = nn.Linear(256, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, img_feat, hx


class Model_adaptive(nn.Module):
    """
    自适应分辨率模型 - 可接受任意大小的深度图输入
    
    使用 AdaptiveAvgPool2d 将任意尺寸特征图池化到固定大小，
    从而实现对任意输入分辨率的支持。
    
    结构与参考项目保持一致：CNN特征提取 + GRU时序记忆 + 全连接输出
    
    GRU的作用：
    1. 时序记忆：记住之前帧中看到的障碍物信息（解决部分可观测问题）
    2. 隐式速度估计：从连续帧中估计障碍物相对运动
    3. 控制平滑：避免动作剧烈抖动
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256, adaptive_size=(4, 6)) -> None:
        """
        Args:
            dim_obs: 里程计观测维度 (位置、速度、姿态等)
            dim_action: 动作输出维度
            hidden_dim: GRU隐藏层维度
            adaptive_size: 自适应池化目标尺寸 (H, W)，决定了特征图被压缩到的固定大小
        """
        super().__init__()
        self.adaptive_size = adaptive_size
        self.hidden_dim = hidden_dim
        
        # CNN 特征提取器 - 逐步下采样
        # 每层 stride=2 将分辨率减半，共 5 层 -> 总下采样 32x
        # 例如 640x480 -> 320x240 -> 160x120 -> 80x60 -> 40x30 -> 20x15
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
        )
        
        # 自适应池化：将任意大小特征图池化到固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_size)
        
        # 特征映射到隐藏维度
        pool_feat_dim = 256 * adaptive_size[0] * adaptive_size[1]
        self.stem_fc = nn.Linear(pool_feat_dim, hidden_dim, bias=False)
        
        # 里程计观测投影
        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        
        # GRU 循环层 - 时序记忆
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        
        self.act = nn.LeakyReLU(0.05)
    
    def reset(self):
        """重置隐状态（新 episode 时调用）"""
        pass
    
    def forward(self, x: torch.Tensor, v: torch.Tensor, hx=None):
        """
        Args:
            x: 深度图 (B, 1, H, W) - 任意分辨率
            v: 里程计观测 (B, dim_obs)
            hx: GRU隐状态 (B, hidden_dim) 或 None
            
        Returns:
            act: 动作输出 (B, dim_action)
            aux: 辅助输出 (预留，当前为 None)
            hx: 更新后的隐状态 (B, hidden_dim)
        """
        # CNN 特征提取
        feat = self.conv_layers(x)           # (B, 256, H', W')
        feat = self.adaptive_pool(feat)      # (B, 256, adaptive_size[0], adaptive_size[1])
        feat = feat.flatten(1)               # (B, 256 * adaptive_size[0] * adaptive_size[1])
        img_feat = self.stem_fc(feat)        # (B, hidden_dim)
        
        # 融合图像特征和里程计信息
        fused = self.act(img_feat + self.v_proj(v))
        
        # GRU 时序更新
        hx = self.gru(fused, hx)
        
        # 输出动作
        act = self.fc(self.act(hx))
        
        return act, img_feat, hx


class Model_640x480(nn.Module):
    """
    专为 640x480 分辨率设计的模型（固定分辨率版本）
    
    如果你需要针对其他分辨率调整，修改方法：
    1. 计算每层卷积输出尺寸: out = floor((in + 2*padding - kernel) / stride + 1)
    2. 修改 stem 最后的 nn.Linear 输入维度
    
    640x480 经过 5 层 stride=2 卷积后:
    640x480 -> 320x240 -> 160x120 -> 80x60 -> 40x30 -> 20x15
    最终特征图尺寸: 256 通道 x 20 x 15 = 76800
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 固定分辨率 CNN 特征提取
        self.stem = nn.Sequential(
            # 640x480 -> 320x240
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            # 320x240 -> 160x120
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            # 160x120 -> 80x60
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            # 80x60 -> 40x30
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            # 40x30 -> 20x15
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(256 * 20 * 15, hidden_dim, bias=False),  # 76800 -> hidden_dim
        )
        
        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, img_feat, hx


# ================================================================
# 注意力模型 — CBAM-style 通道+空间注意力
# ================================================================

class _ChannelAttention(nn.Module):
    """通道注意力：squeeze-excitation 变体"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = x.mean(dim=(2, 3))          # (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _SpatialAttention(nn.Module):
    """空间注意力：用 max/avg 池化后 1×1 融合"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        w = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * w


class Model_attention(nn.Module):
    """
    注意力增强模型 — 在 CNN 特征上施加通道+空间注意力 (CBAM)。

    与 Model_bigger 同分辨率 (48×64)，使用 AdaptiveAvgPool 兼容任意尺寸。
    注意力帮助模型聚焦深度图中障碍物密集区域。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
        )
        self.ca = _ChannelAttention(256)
        self.sa = _SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.stem_fc = nn.Linear(256 * 3 * 4, hidden_dim, bias=False)

        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        feat = self.conv(x)
        feat = self.ca(feat)
        feat = self.sa(feat)
        feat = self.pool(feat).flatten(1)
        img_feat = self.stem_fc(feat)
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        return self.fc(self.act(hx)), img_feat, hx


# ================================================================
# 多尺度特征金字塔模型
# ================================================================

class Model_multiscale(nn.Module):
    """
    多尺度特征金字塔模型 — 不同层级的 CNN 特征分别池化后拼接。

    低层特征捕获近距离 fine-grained 障碍物边缘，高层特征捕获远处全局布局。
    兼容任意输入分辨率。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,  32, 3, 2, 1, bias=False), nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.LeakyReLU(0.05))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.LeakyReLU(0.05))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.LeakyReLU(0.05))

        # 每级特征独立池化到 2×2
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool4 = nn.AdaptiveAvgPool2d((2, 2))

        # 64*4 + 128*4 + 256*4 = 1792
        cat_dim = (64 + 128 + 256) * 2 * 2
        self.stem_fc = nn.Linear(cat_dim, hidden_dim, bias=False)

        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        multi = torch.cat([
            self.pool2(f2).flatten(1),
            self.pool3(f3).flatten(1),
            self.pool4(f4).flatten(1),
        ], dim=1)
        img_feat = self.stem_fc(multi)
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        return self.fc(self.act(hx)), img_feat, hx


# ================================================================
# 残差模型 — ResBlock + LSTM
# ================================================================

class _ResBlock(nn.Module):
    """带下采样的残差块 (stride=2 时 shortcut 用 1×1 conv)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class Model_residual(nn.Module):
    """
    残差网络模型 — ResBlock 堆叠 + LSTM 时序记忆。

    与 GRU 模型对比，使用 LSTM 提供更强的长时记忆能力（独立的遗忘门）。
    残差连接缓解深层网络梯度消失。兼容任意输入分辨率。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            _ResBlock(1,   32, stride=2),
            _ResBlock(32,  64, stride=2),
            _ResBlock(64,  128, stride=2),
            _ResBlock(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.stem_fc = nn.Linear(256 * 3 * 4, hidden_dim, bias=False)

        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        img_feat = self.stem_fc(feat)
        fused = self.act(img_feat + self.v_proj(v))
        # hx 是 (h, c) 元组；首次传 None 时 LSTMCell 自动初始化
        if hx is not None and not isinstance(hx, tuple):
            # 兼容外部传入单张量的情况：视为 h，c 初始化为零
            hx = (hx, torch.zeros_like(hx))
        if hx is None:
            h, c = self.lstm(fused)
        else:
            h, c = self.lstm(fused, hx)
        act = self.fc(self.act(h))
        return act, img_feat, (h, c)


# ================================================================
# 轻量级模型 — 深度可分离卷积 (MobileNet-style)
# ================================================================

class _DepthwiseSeparable(nn.Module):
    """深度可分离卷积：depthwise + pointwise"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class Model_lightweight(nn.Module):
    """
    轻量级模型 — 深度可分离卷积 + 小隐层维度。

    参数量约为 Model_bigger 的 1/4，适合快速迭代实验或边缘部署验证。
    兼容任意输入分辨率。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),  # 首层用普通卷积（单通道不适合 depthwise）
            nn.LeakyReLU(0.05),
            _DepthwiseSeparable(16, 32, stride=2),
            _DepthwiseSeparable(32, 64, stride=2),
            _DepthwiseSeparable(64, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.stem_fc = nn.Linear(128 * 3 * 4, hidden_dim, bias=False)

        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        img_feat = self.stem_fc(feat)
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        return self.fc(self.act(hx)), img_feat, hx


# ================================================================
# LiDAR 距离图模型 — 处理 (B, 1, num_beams, points_per_beam) 输入
# ================================================================

class Model_lidar(nn.Module):
    """
    激光雷达距离图模型 — 输入为 LiDAR 距离图像 (B, 1, V, H_l)。

    默认 V=16 (光束数), H_l=64 (每光束采样点数)。
    使用 AdaptiveAvgPool2d 兼容其他参数配置。
    隐层维度与 Model_bigger 一致 (256)，便于公平对比。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=(1, 2), padding=1, bias=False),  # V 方向不下采样
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.stem_fc = nn.Linear(256 * 2 * 4, hidden_dim, bias=False)

        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x, v, hx=None):
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        img_feat = self.stem_fc(feat)
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        return self.fc(self.act(hx)), img_feat, hx


class Model_fusion(nn.Module):
    """
    深度-LiDAR 双分支融合模型。

    两路 CNN 分别提取深度图和距离图特征，
    拼接后通过融合层 + GRU 输出动作。
    总参数量与 Model_bigger 可比。
    """
    def __init__(self, dim_obs=10, dim_action=6, hidden_dim=256):
        super().__init__()
        half = hidden_dim // 2  # 每个分支的特征维度

        # --- 深度图分支 (48×64) ---
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
        )
        self.depth_pool = nn.AdaptiveAvgPool2d((3, 4))
        self.depth_fc = nn.Linear(128 * 3 * 4, half, bias=False)

        # --- LiDAR 距离图分支 (16×64) ---
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, (1, 2), 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.LeakyReLU(0.05),
        )
        self.lidar_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.lidar_fc = nn.Linear(128 * 2 * 4, half, bias=False)

        # --- 融合 ---
        self.v_proj = nn.Linear(dim_obs, hidden_dim)
        self.v_proj.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x_depth, x_lidar, v, hx=None):
        """
        Args:
            x_depth: (B, 1, H, W)   预处理后的深度图
            x_lidar: (B, 1, V, H_l) 预处理后的距离图
            v:       (B, dim_obs)    状态向量
            hx:      GRU 隐状态
        """
        d_feat = self.depth_conv(x_depth)
        d_feat = self.depth_pool(d_feat).flatten(1)
        d_feat = self.depth_fc(d_feat)

        l_feat = self.lidar_conv(x_lidar)
        l_feat = self.lidar_pool(l_feat).flatten(1)
        l_feat = self.lidar_fc(l_feat)

        img_feat = torch.cat([d_feat, l_feat], dim=1)         # (B, hidden_dim)
        fused = self.act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)
        return self.fc(self.act(hx)), img_feat, hx


# ================================================================
# CMA-ES 控制器
# ================================================================

class DecayController(nn.Module):
    """
    CMA-ES 优化的梯度衰减控制器。

    接收主网络 CNN 提取的图像特征（detach），输出 per-sample 的梯度衰减因子。
    参数由 CMA-ES 进化搜索，不参与梯度训练。

    输出范围: [decay_min, decay_min + decay_range] 通过 sigmoid 映射。
    默认 [0.2, 1.0]。
    """
    def __init__(self, feat_dim=256, decay_min=0.2, decay_range=0.8):
        super().__init__()
        self.decay_min = decay_min
        self.decay_range = decay_range
        self.linear = nn.Linear(feat_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # 所有参数冻结 — 仅由 CMA-ES 进化
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, img_feat):
        """
        Args:
            img_feat: (B, feat_dim), 从主网络 CNN detach 后的特征
        Returns:
            decay: (B,), 每个 sample 的梯度衰减因子
        """
        raw = self.linear(img_feat)
        decay = self.decay_min + self.decay_range * torch.sigmoid(raw)
        return decay.squeeze(-1)

    def get_params_vector(self):
        """将所有参数展平为一维向量（CMA-ES 接口）"""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_params_vector(self, vector):
        """从一维向量恢复参数（CMA-ES 接口）"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(vector[offset:offset + numel].reshape(p.shape))
            offset += numel

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class LossGuide(nn.Module):
    """
    CMA-ES 进化的损失系数控制器（指导函数）。

    原始参数 → sigmoid[min, max] 映射 → 有界损失权重。
    参数由 CMA-ES 进化搜索，不参与梯度训练。

    零初始化: sigmoid(0) = 0.5 → 每个系数映射到 [min, max] 中点。
    """
    COEFF_NAMES = ['v', 'speed', 'v_pred', 'collide', 'obj_avoidance',
                   'd_acc', 'd_jerk', 'd_snap', 'ground_affinity', 'bias',
                   'lateral', 'drone_collide']

    DEFAULT_BOUNDS = {
        'v':              (0.1, 5.0),
        'speed':          (0.0, 2.0),
        'v_pred':         (0.1, 5.0),
        'collide':        (0.5, 10.0),
        'obj_avoidance':  (0.3, 8.0),
        'd_acc':          (0.001, 0.1),
        'd_jerk':         (0.0001, 0.05),
        'd_snap':         (0.0, 0.01),
        'ground_affinity':(0.0, 1.0),
        'bias':           (0.0, 1.0),
        'lateral':        (0.0, 1.0),
        'drone_collide':  (1.0, 10.0),
    }

    def __init__(self, bounds=None):
        super().__init__()
        b = bounds if bounds is not None else self.DEFAULT_BOUNDS

        mins = torch.tensor([b[n][0] for n in self.COEFF_NAMES])
        maxs = torch.tensor([b[n][1] for n in self.COEFF_NAMES])

        self.register_buffer('mins', mins)
        self.register_buffer('ranges', maxs - mins)

        self.raw = nn.Parameter(torch.zeros(len(self.COEFF_NAMES)))

        # 所有参数冻结 — 仅由 CMA-ES 进化
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self):
        """Returns dict of {name: bounded_coefficient_value}."""
        bounded = self.mins + self.ranges * torch.sigmoid(self.raw)
        return {name: bounded[i] for i, name in enumerate(self.COEFF_NAMES)}

    def get_params_vector(self):
        return self.raw.data.clone()

    def set_params_vector(self, vector):
        self.raw.data.copy_(vector)

    @property
    def num_params(self):
        return self.raw.numel()


# ================================================================
# MetaController — A-网络实现
# ================================================================

class MetaController(nn.Module):
    """
    A-网络 (Meta Controller): 元控制器。

    生物学类比:
      - 参数在个体"生命周期"（= 一轮训练）内固定，仅由种群级进化 (CMA-ES) 更新
      - 观察"个体"（= 策略网络）的训练状态，动态输出学习信号的调制
      - 状态性 (stateful): 内部维护训练统计的指数移动平均，
        使得相同参数在训练的不同阶段产生不同输出

    与 LossGuide 的核心区别:
      - LossGuide: f() → 固定常数 (无输入, 12D 查找表)
      - MetaController: f(training_state) → 动态调制 (有输入, 条件响应网络)

    架构: Linear(N_IN, hidden) + Tanh + Linear(hidden, N_OUT)
    默认 8 输入 × 4 隐层 × 6 输出 = 66 参数, CMA-ES 高效搜索范围内。
    """

    # 输出系数名与边界
    OUTPUT_KEYS = ['v', 'collide', 'obj_avoidance', 'd_acc', 'lateral', 'grad_decay']
    OUTPUT_BOUNDS = {
        'v':              (0.1, 5.0),
        'collide':        (0.5, 15.0),
        'obj_avoidance':  (0.3, 8.0),
        'd_acc':          (0.001, 0.1),
        'lateral':        (0.0, 2.0),
        'grad_decay':     (0.2, 0.8),
    }

    # 输入特征名 (供外部参考)
    INPUT_KEYS = [
        'progress',         # 训练进度 [0, 1]
        'loss_v',           # 速度损失 EMA (log1p 归一化)
        'loss_collide',     # 碰撞损失 EMA (log1p 归一化)
        'loss_obj',         # 障碍物回避损失 EMA (log1p 归一化)
        'success_rate',     # 成功率 EMA [0, 1]
        'collision_rate',   # 碰撞率 EMA [0, 1] (= 1 - collision_free_rate)
        'avg_speed',        # 平均速度 EMA (归一化至 ~[0,1])
        'goal_progress',    # 目标进度 EMA [0, 1]
    ]
    N_INPUT = len(INPUT_KEYS)
    N_OUTPUT = len(OUTPUT_KEYS)

    def __init__(self, hidden: int = 4, ema_alpha: float = 0.1):
        super().__init__()
        bounds = self.OUTPUT_BOUNDS

        mins = torch.tensor([bounds[k][0] for k in self.OUTPUT_KEYS])
        maxs = torch.tensor([bounds[k][1] for k in self.OUTPUT_KEYS])
        self.register_buffer('mins', mins)
        self.register_buffer('ranges', maxs - mins)

        # 小型 MLP — 参数由 CMA-ES 进化
        self.fc1 = nn.Linear(self.N_INPUT, hidden)
        self.fc2 = nn.Linear(hidden, self.N_OUTPUT)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # 所有参数冻结 — 仅由 CMA-ES 进化
        for p in self.parameters():
            p.requires_grad_(False)

        # 训练状态 EMA 追踪 (不是模型参数，不参与 CMA-ES)
        self.ema_alpha = ema_alpha
        self._ema: dict[str, float] = {}

    def reset_ema(self):
        """新一轮训练/评估前重置 EMA 统计"""
        self._ema.clear()

    def update_ema(self, metrics: dict):
        """用 episode 结果更新训练状态 EMA"""
        alpha = self.ema_alpha
        for key in ('loss_v', 'loss_collide', 'loss_obj_avoidance',
                     'success_rate', 'collision_free_rate', 'avg_speed', 'goal_progress'):
            val = metrics.get(key)
            if val is None:
                continue
            v = val.item() if isinstance(val, torch.Tensor) else float(val)
            if key in self._ema:
                self._ema[key] = (1 - alpha) * self._ema[key] + alpha * v
            else:
                self._ema[key] = v

    def _build_input(self, progress: float) -> torch.Tensor:
        """将 EMA 统计 + 训练进度组装为网络输入张量 (GPU-only)"""
        device = self.mins.device
        raw = torch.tensor([
            progress,
            self._ema.get('loss_v', 1.0),
            self._ema.get('loss_collide', 1.0),
            self._ema.get('loss_obj_avoidance', 0.5),
            self._ema.get('success_rate', 0.0),
            1.0 - self._ema.get('collision_free_rate', 0.5),  # collision_rate
            min(self._ema.get('avg_speed', 0.5) / 3.0, 1.0),  # 归一化
            self._ema.get('goal_progress', 0.0),
        ], dtype=torch.float32, device=device)
        # log1p 归一化: 索引 1,2,3 (loss_v, loss_collide, loss_obj)
        raw[1:4] = torch.log1p(raw[1:4])
        return raw

    def forward(self, progress: float) -> dict:
        """
        Args:
            progress: 当前训练迭代 / 总迭代数, [0, 1]
        Returns:
            dict of {name: bounded_value} — loss 系数 + grad_decay
        """
        x = self._build_input(progress)
        h = torch.tanh(self.fc1(x))
        raw = self.fc2(h)
        bounded = self.mins + self.ranges * torch.sigmoid(raw)
        return {k: bounded[i] for i, k in enumerate(self.OUTPUT_KEYS)}

    # ---- CMA-ES 接口 ----
    def get_params_vector(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_params_vector(self, vector):
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vector[offset:offset + n].reshape(p.shape))
            offset += n

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# LossNetwork — A-网络: 进化的损失函数
# ================================================================

class LossNetwork(nn.Module):
    """
    A-网络 (Loss Network): 进化的损失函数。

    生物学类比:
      - 多巴胺/血清素等神经调质系统的"连线方式"由基因决定 (进化固定)
      - 这些系统不直接产生行为, 而是调制突触可塑性 (= B-网络的学习规则)
      - LossNetwork 正是这种调制: 它接收训练信号, 输出标量损失,
        梯度通过它流向策略网络 — 它 **定义** 了策略网络"如何学习"

    与 MetaController 的核心区别:
      - MetaController: 输出系数 → 仍然是预定义损失项的线性组合 Σ c_i × L_i
      - LossNetwork: 本身就是损失函数 → 可以学习任意非线性损失组合,
        包括: 条件权重 (碰撞时降低速度惩罚), 交互项 (speed × collision),
        训练阶段自适应等, 远超线性加权的表达力

    架构: Linear(13, hidden) + LeakyReLU(0.1) + Linear(hidden, 1)
    输入: 12个损失分量 (可微, 在计算图中) + 训练进度 (标量上下文)
    输出: 标量损失值 — 梯度通过此输出回传到策略网络 (B-网络)

    关键设计:
      - requires_grad=False: 参数不参与梯度优化, 仅由 CMA-ES 进化
      - 前向传播在计算图中: 梯度必须流过此网络到达策略网络 (不能 no_grad)
      - LeakyReLU: 正输入为恒等映射, 保证温启动时精确复现线性组合
      - 温启动: 初始行为 = Σ coef_i × loss_i (恒等初始化 + 默认系数)
        CMA-ES 从此起点搜索非线性改进
    """

    LOSS_KEYS = [
        'v', 'lateral', 'speed', 'v_pred', 'collide',
        'obj_avoidance', 'd_acc', 'd_jerk', 'd_snap',
        'ground_affinity', 'bias', 'drone_collide',
    ]
    N_LOSS = len(LOSS_KEYS)  # 12

    def __init__(self, hidden: int = 16, default_coefs: dict = None):
        super().__init__()
        dim_in = self.N_LOSS + 1  # 12 losses + 1 progress
        self.fc1 = nn.Linear(dim_in, hidden)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

        # ---- 温启动初始化 ----
        # fc1: 前 N_LOSS 个隐层神经元 = 损失项的恒等映射
        #   weight[i, i] = 1.0, 其余 = 0
        #   → h_i = LeakyReLU(loss_i) = loss_i (因为 loss_i ≥ 0)
        # fc2: 对应默认系数
        #   weight[0, i] = coef_i
        # 初始行为: output = Σ coef_i × loss_i = 精确等于当前线性组合
        # 额外 (hidden - N_LOSS) 个神经元: 全零, CMA-ES 可激活用于交互项
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)

        if default_coefs is None:
            default_coefs = {}
        for i, key in enumerate(self.LOSS_KEYS):
            if i < hidden:
                self.fc1.weight.data[i, i] = 1.0
                self.fc2.weight.data[0, i] = default_coefs.get(key, 0.0)

        # 所有参数冻结 — 仅由 CMA-ES 进化
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, loss_dict: dict, progress: float) -> torch.Tensor:
        """
        组合各损失分量为标量损失。

        IMPORTANT: 此方法在计算图中执行 — 不要包在 torch.no_grad() 中!
        梯度流: LossNetwork 输出 → 各损失分量 → 轨迹 → 策略网络 (B-网络)

        Args:
            loss_dict: DroneLoss.forward() 返回的 metrics dict,
                       其中 loss_* 条目是在计算图中的张量
            progress: 训练进度 [0, 1]
        Returns:
            标量损失张量 (在计算图中, 可 .backward())
        """
        parts = []
        for key in self.LOSS_KEYS:
            val = loss_dict.get(f'loss_{key}')
            if val is not None and isinstance(val, torch.Tensor):
                parts.append(val)
            else:
                # 确保 fallback 在正确 device 上 (GPU-only)
                parts.append(self.fc1.weight.new_zeros(()))
        # progress 作为上下文输入 (不在计算图中, 但不影响梯度流向损失分量)
        parts.append(self.fc1.weight.new_tensor(progress))
        x = torch.stack(parts)  # (dim_in,)

        h = F.leaky_relu(self.fc1(x), negative_slope=0.1)  # (hidden,)
        return self.fc2(h).squeeze(-1)  # scalar

    # ---- CMA-ES 接口 ----
    def get_params_vector(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_params_vector(self, vector: torch.Tensor):
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vector[offset:offset + n].reshape(p.shape))
            offset += n

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    print("Testing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    # 原始模型
    m = Model().to(device)
    o, _, _ = m(torch.randn(2, 1, 12, 16, device=device), torch.randn(2, 9, device=device))
    print(f"Model (12x16): output={o.shape}, params={count_params(m):,}")

    # bigger
    m = Model_bigger().to(device)
    o, _, _ = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 9, device=device))
    print(f"Model_bigger (48x64): output={o.shape}, params={count_params(m):,}")

    # adaptive — 多分辨率
    m = Model_adaptive(dim_obs=10, dim_action=6).to(device)
    for h, w in [(48, 64), (240, 320), (480, 640)]:
        o, _, _ = m(torch.randn(2, 1, h, w, device=device), torch.randn(2, 10, device=device))
        print(f"Model_adaptive ({h}x{w}): output={o.shape}, params={count_params(m):,}")

    # 640x480
    m = Model_640x480(dim_obs=10, dim_action=6).to(device)
    o, _, _ = m(torch.randn(2, 1, 480, 640, device=device), torch.randn(2, 10, device=device))
    print(f"Model_640x480: output={o.shape}, params={count_params(m):,}")

    # attention
    m = Model_attention(dim_obs=10, dim_action=6).to(device)
    o, _, _ = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 10, device=device))
    print(f"Model_attention (48x64): output={o.shape}, params={count_params(m):,}")

    # multiscale
    m = Model_multiscale(dim_obs=10, dim_action=6).to(device)
    o, _, _ = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 10, device=device))
    print(f"Model_multiscale (48x64): output={o.shape}, params={count_params(m):,}")

    # residual (LSTM)
    m = Model_residual(dim_obs=10, dim_action=6).to(device)
    o, _, hx = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 10, device=device))
    o2, _, _ = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 10, device=device), hx)
    print(f"Model_residual (48x64): output={o.shape}, params={count_params(m):,}")

    # lightweight
    m = Model_lightweight(dim_obs=10, dim_action=6).to(device)
    o, _, _ = m(torch.randn(2, 1, 48, 64, device=device), torch.randn(2, 10, device=device))
    print(f"Model_lightweight (48x64): output={o.shape}, params={count_params(m):,}")
