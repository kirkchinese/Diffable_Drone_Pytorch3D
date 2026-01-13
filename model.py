import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

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
        return act, None, hx

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
        return act, None, hx


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
        
        return act, None, hx


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
        return act, None, hx


if __name__ == '__main__':
    print("Testing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试原始模型
    model1 = Model().to(device)
    x1 = torch.randn(2, 1, 12, 16).to(device)
    v1 = torch.randn(2, 9).to(device)
    out1, _, hx1 = model1(x1, v1)
    print(f"Model (12x16): input={x1.shape}, output={out1.shape}")
    
    # 测试 bigger 模型
    model2 = Model_bigger().to(device)
    x2 = torch.randn(2, 1, 48, 64).to(device)
    v2 = torch.randn(2, 9).to(device)
    out2, _, hx2 = model2(x2, v2)
    print(f"Model_bigger (48x64): input={x2.shape}, output={out2.shape}")
    
    # 测试自适应模型 - 不同分辨率
    model3 = Model_adaptive(dim_obs=10, dim_action=6).to(device)
    for h, w in [(48, 64), (240, 320), (480, 640), (720, 1280)]:
        x3 = torch.randn(2, 1, h, w).to(device)
        v3 = torch.randn(2, 10).to(device)
        out3, _, hx3 = model3(x3, v3)
        print(f"Model_adaptive ({h}x{w}): input={x3.shape}, output={out3.shape}")
    
    # 测试 640x480 专用模型
    model4 = Model_640x480(dim_obs=10, dim_action=6).to(device)
    x4 = torch.randn(2, 1, 480, 640).to(device)
    v4 = torch.randn(2, 10).to(device)
    out4, _, hx4 = model4(x4, v4)
    print(f"Model_640x480: input={x4.shape}, output={out4.shape}")
    
    # 统计参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Model: {count_params(model1):,}")
    print(f"  Model_bigger: {count_params(model2):,}")
    print(f"  Model_adaptive: {count_params(model3):,}")
    print(f"  Model_640x480: {count_params(model4):,}")
