"""
LiDAR 传感器模块验证测试

测试内容:
  1. setup() 网格预计算正确性
  2. depth_to_range_image 形状与数值范围
  3. 背景(-1) 正确传播
  4. 余弦修正 cos(θ) 在光轴处 ≈ 1、边缘处 > 0
  5. 点云输出形状与有效性
  6. preprocess 输出形状与数值分布
  7. 与 Model_lidar / Model_fusion 端到端前向
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lidar_sensor import LiDARSensor
from model import Model_lidar, Model_fusion, Model_bigger


def test_basic_shapes():
    """基本维度验证"""
    print(">>> test_basic_shapes")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 4, 48, 64
    focal = 38.5

    lidar = LiDARSensor(num_beams=16, points_per_beam=64,
                        max_range=24.0, min_range=0.3, device=device)
    lidar.setup(focal_length=focal, image_height=H, image_width=W)

    # 模拟深度图: 均匀 5m 平面
    depth = torch.full((B, H, W), 5.0, device=device)

    range_img = lidar.depth_to_range_image(depth)
    assert range_img.shape == (B, 16, 64), f"shape mismatch: {range_img.shape}"
    print(f"  range_img shape: {range_img.shape}")
    print(f"  range min={range_img.min():.3f} max={range_img.max():.3f}")

    # 光轴处 range ≈ depth (cos ≈ 1)
    center_val = range_img[0, 8, 32].item()
    assert abs(center_val - 5.0) < 0.5, f"光轴处 range={center_val}, 预期≈5.0"
    print(f"  光轴 range={center_val:.3f} (预期≈5.0)")

    # 边缘处 range > depth (cos < 1)
    edge_val = range_img[0, 0, 0].item()
    assert edge_val > 5.0, f"边缘 range={edge_val} 应 > 5.0"
    print(f"  边缘 range={edge_val:.3f} (> 5.0 ✓)")

    print("  PASS\n")


def test_background_propagation():
    """背景值(-1)正确传播"""
    print(">>> test_background_propagation")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 2, 48, 64
    focal = 38.5

    lidar = LiDARSensor(device=device)
    lidar.setup(focal_length=focal, image_height=H, image_width=W)

    depth = torch.full((B, H, W), -1.0, device=device)  # 全背景
    range_img = lidar.depth_to_range_image(depth)
    assert (range_img == -1.0).all(), "全背景 depth 应产生全 -1 range"
    print("  全背景 → 全 -1 ✓")

    # 半背景
    depth2 = torch.full((B, H, W), 5.0, device=device)
    depth2[:, :H//2, :] = -1.0                # 上半部分背景
    range_img2 = lidar.depth_to_range_image(depth2)
    n_invalid = (range_img2 < 0).sum().item()
    n_total = range_img2.numel()
    print(f"  半背景: {n_invalid}/{n_total} 无效值")
    assert 0 < n_invalid < n_total
    print("  PASS\n")


def test_point_cloud():
    """点云输出验证"""
    print(">>> test_point_cloud")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 2, 48, 64
    focal = 38.5

    lidar = LiDARSensor(num_beams=8, points_per_beam=32, device=device)
    lidar.setup(focal_length=focal, image_height=H, image_width=W)

    depth = torch.full((B, H, W), 3.0, device=device)
    pc = lidar.depth_to_point_cloud(depth)
    assert pc.shape == (B, 8 * 32, 3), f"点云形状: {pc.shape}"
    # 所有点距原点应 ≈ 3m (range ≈ 3 / cos(θ))
    dist = pc.norm(dim=-1)
    valid = dist > 0
    assert valid.any(), "应有有效点"
    min_d = dist[valid].min().item()
    max_d = dist[valid].max().item()
    print(f"  点距原点: [{min_d:.2f}, {max_d:.2f}]")
    assert min_d >= 2.5 and max_d <= 6.0, "距离范围异常"
    print("  PASS\n")


def test_preprocess():
    """preprocess 输出验证"""
    print(">>> test_preprocess")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 4, 48, 64
    focal = 38.5

    lidar = LiDARSensor(device=device)
    lidar.setup(focal_length=focal, image_height=H, image_width=W)

    depth = torch.rand(B, H, W, device=device) * 20 + 0.5  # [0.5, 20.5]
    range_img = lidar.depth_to_range_image(depth)
    x = lidar.preprocess(range_img)
    assert x.shape == (B, 1, 16, 64), f"preprocess shape: {x.shape}"
    # inverse distance: 3/0.3 - 0.6 = 9.4 (近), 3/24 - 0.6 = -0.475 (远)
    print(f"  输出范围: [{x.min():.3f}, {x.max():.3f}]")
    print("  PASS\n")


def test_model_lidar():
    """Model_lidar 端到端前向"""
    print(">>> test_model_lidar")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 4

    lidar = LiDARSensor(device=device)
    lidar.setup(focal_length=38.5, image_height=48, image_width=64)

    model = Model_lidar(dim_obs=10, dim_action=6).to(device)

    depth = torch.rand(B, 48, 64, device=device) * 20 + 0.5
    range_img = lidar.depth_to_range_image(depth)
    x = lidar.preprocess(range_img)

    state = torch.randn(B, 10, device=device)
    act, feat, hx = model(x, state, None)
    assert act.shape == (B, 6), f"action shape: {act.shape}"
    print(f"  act shape: {act.shape}, feat: {feat.shape}, hx: {hx.shape}")
    print("  PASS\n")


def test_model_fusion():
    """Model_fusion 端到端前向"""
    print(">>> test_model_fusion")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 4

    lidar = LiDARSensor(device=device)
    lidar.setup(focal_length=38.5, image_height=48, image_width=64)

    model = Model_fusion(dim_obs=10, dim_action=6).to(device)

    depth = torch.rand(B, 48, 64, device=device) * 20 + 0.5

    # 深度图预处理 (与 preprocess_depth_for_model 一致)
    from navigation_utils import preprocess_depth_for_model
    x_depth = preprocess_depth_for_model(depth, 0.3, 24.0)

    # LiDAR 预处理
    range_img = lidar.depth_to_range_image(depth)
    x_lidar = lidar.preprocess(range_img)

    state = torch.randn(B, 10, device=device)
    act, feat, hx = model(x_depth, x_lidar, state, None)
    assert act.shape == (B, 6), f"fusion action shape: {act.shape}"
    print(f"  act shape: {act.shape}, feat: {feat.shape}")
    print("  PASS\n")


def test_cos_correction_monotonicity():
    """余弦修正因子在视场边缘严格大于1"""
    print(">>> test_cos_correction_monotonicity")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lidar = LiDARSensor(device=device)
    lidar.setup(focal_length=38.5, image_height=48, image_width=64)

    cos_corr = lidar._cos_correction.squeeze(0)  # (V, H_l)
    center_cos = cos_corr[8, 32].item()          # 光轴附近
    corner_cos = cos_corr[0, 0].item()            # 角落

    print(f"  cos(光轴)={center_cos:.4f}  cos(角落)={corner_cos:.4f}")
    # 光轴处 cos≈1, 角落处 cos<1 → 1/cos > 1
    assert center_cos > corner_cos, "光轴处 cos 应大于角落"
    assert abs(center_cos - 1.0) < 0.05, f"光轴 cos 应≈1, got {center_cos}"
    assert corner_cos > 0.5, f"角落 cos 不应太小, got {corner_cos}"
    print("  PASS\n")


def test_param_count_comparison():
    """打印各模型参数量用于论文对比"""
    print(">>> 参数量对比")
    models = {
        'Model_bigger (depth)': Model_bigger(dim_obs=10, dim_action=6),
        'Model_lidar':          Model_lidar(dim_obs=10, dim_action=6),
        'Model_fusion':         Model_fusion(dim_obs=10, dim_action=6),
    }
    for name, m in models.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n:,} params ({n/1024:.1f}K)")
    print()


if __name__ == '__main__':
    test_basic_shapes()
    test_background_propagation()
    test_point_cloud()
    test_preprocess()
    test_cos_correction_monotonicity()
    test_model_lidar()
    test_model_fusion()
    test_param_count_comparison()
    print("=== ALL TESTS PASSED ===")
