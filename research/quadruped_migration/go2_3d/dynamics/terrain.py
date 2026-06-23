"""平滑地形高度场 + 机体前向 heightmap 采样（E3D-10 感知地形运动）。

设计原则（贯穿全弧的梯度保真）：地形必须**平滑可微**——尖锐台阶=接触不连续=梯度爆炸
(2D-F4/E3D-4a 教训)。故地形 = 若干高斯隆起之和（C∞ 光滑），机器人须抬足跨过；heightmap
是地形高度在机体前向网格点的采样（便宜的可微外感知，proto-depth；full 深度图渲染成本高后置）。

地形接入接触不改 contact_3d：在算接触前把足端 z 减去其 xy 处地形高度（gap=foot_z−terrain_h），
力按原足位施加（缓坡法向≈+z 的近似）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TerrainField:
    """批量平滑地形：高度 = Σ_k h_k·exp(−‖xy−c_k‖²/(2σ_k²))。各张量 batch 维 B。"""
    centers: torch.Tensor   # (B,K,2) 隆起中心 (世界 xy)
    heights: torch.Tensor   # (B,K)   隆起高度 [m]
    sigmas: torch.Tensor    # (B,K)   隆起宽度 [m]

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """xy:(B,N,2) → (B,N) 地形高度（可微）。"""
        d2 = ((xy[:, :, None, :] - self.centers[:, None, :, :]) ** 2).sum(-1)   # (B,N,K)
        return (self.heights[:, None, :] * torch.exp(-d2 / (2 * self.sigmas[:, None, :] ** 2))).sum(-1)


def random_terrain(B, gen, device, dtype, n_bumps=10,
                   h_lo=0.05, h_hi=0.11, x_lo=0.3, x_hi=2.2, y_spread=0.20,
                   sig_lo=0.07, sig_hi=0.12) -> TerrainField:
    """随机平滑隆起地带（机器人从原点沿 +x 走，隆起密铺前方）。高度 0.05–0.11≫h_swing=0.04
    → 盲走(固定摆高 0.04)绊；感知策略(dh 可达 0.08→摆高 0.12)能跨。陡(σ 小)、密(10 隆起/1.9m)。"""
    def u(shape, lo, hi):
        return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device, dtype=dtype)
    cx = u((B, n_bumps), x_lo, x_hi)
    cy = u((B, n_bumps), -y_spread, y_spread)
    centers = torch.stack([cx, cy], dim=-1)
    heights = u((B, n_bumps), h_lo, h_hi)
    sigmas = u((B, n_bumps), sig_lo, sig_hi)
    return TerrainField(centers, heights, sigmas)


def flat_terrain(B, device, dtype) -> TerrainField:
    """平地（高度恒 0）——盲/感知对照的对照地形。"""
    z = torch.zeros(B, 1, device=device, dtype=dtype)
    return TerrainField(torch.zeros(B, 1, 2, device=device, dtype=dtype), z, z + 1.0)


# 机体前向 heightmap 采样网格（体系 xy，机器人前方一片）
def make_grid(nx=6, ny=3, x_lo=-0.30, x_hi=0.50, y_half=0.18, device="cpu", dtype=torch.float32):
    xs = torch.linspace(x_lo, x_hi, nx, device=device, dtype=dtype)
    ys = torch.linspace(-y_half, y_half, ny, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)        # (nx*ny, 2)


def heightmap(state, terrain: TerrainField, grid: torch.Tensor):
    """机体前向 heightmap（B, Ng）：网格点经机器人偏航投到世界 xy，采地形高度，减机体 z
    → 地形相对机体的高度（前方多高）。仅用偏航(不含俯仰)避免姿态噪声污染感知。"""
    from pytorch3d.transforms import quaternion_to_matrix
    R = quaternion_to_matrix(state.q)                                  # (B,3,3)
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])                          # (B,)
    c, s = torch.cos(yaw), torch.sin(yaw)
    gx, gy = grid[:, 0], grid[:, 1]                                    # (Ng,)
    wx = state.p[:, 0:1] + c[:, None] * gx[None] - s[:, None] * gy[None]   # (B,Ng)
    wy = state.p[:, 1:2] + s[:, None] * gx[None] + c[:, None] * gy[None]
    th = terrain.height(torch.stack([wx, wy], dim=-1))                 # (B,Ng)
    return th - state.p[:, 2:3]                                        # 相对机体
