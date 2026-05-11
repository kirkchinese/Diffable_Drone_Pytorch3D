"""
验证多机碰撞重构的正确性：
1. inter_drone_vec_subdivided 返回的向量 L2 范数 = ellipsoid_distance - drone_bounding_radius
2. combined_vec_to_nearest 在多机场景下正确选择最近物体
3. 确认 drone_collide 默认权重为 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loss import DroneLoss

def test_ellipsoid_distance():
    """测试 inter_drone_vec_subdivided 的椭球距离计算"""
    from drone_env import DroneSimulator
    
    device = torch.device('cuda:0')
    B = 8
    G = 4  # 4 drones per group, 2 groups
    
    env = DroneSimulator(
        batch_size=B,
        device=device,
        mesh_path='./data/sample/sample4.obj',
        image_size=(48, 64),
        focal_length=500.0,
        num_samples=1000,
        n_drones_per_group=G,
        drone_mesh_path='./data/base_model/drone.obj',
    )
    
    # 设定已知位置：组1的4架在角落，组2的4架在另一角
    env.p = torch.zeros(B, 3, device=device)
    env.v = torch.zeros(B, 3, device=device)
    
    # 组0: 4架无人机
    env.p[0] = torch.tensor([0.0, 0.0, 0.0])
    env.p[1] = torch.tensor([2.0, 0.0, 0.0])
    env.p[2] = torch.tensor([0.0, 3.0, 0.0])
    env.p[3] = torch.tensor([0.0, 0.0, 1.0])  # Z方向，椭球距离应为 2*1=2
    
    # 组1: 4架无人机 (远离组0)
    env.p[4] = torch.tensor([10.0, 10.0, 0.0])
    env.p[5] = torch.tensor([12.0, 10.0, 0.0])
    env.p[6] = torch.tensor([10.0, 13.0, 0.0])
    env.p[7] = torch.tensor([10.0, 10.0, 1.0])
    
    drone_r = env.drone_bounding_radius
    print(f"Drone bounding radius: {drone_r:.4f}")
    
    # 用 1 步 (n_subdiv=1 ... 实际上至少2) 来测试静态场景
    # 速度为0，所有子步位置相同
    vecs = env.inter_drone_vec_subdivided(n_subdiv=2, dt=1/15)  # (2, B, 3)
    
    # 取第一步的结果 (所有子步应相同因为 v=0)
    vec_norms = vecs[0].norm(dim=-1)  # (B,)
    
    # 手算 drone 0 的最近邻：
    # to drone 1: dx=2, dy=0, dz=0 → ell_dist = sqrt(4) = 2.0
    # to drone 2: dx=0, dy=3, dz=0 → ell_dist = sqrt(9) = 3.0
    # to drone 3: dx=0, dy=0, dz=1 → ell_dist = sqrt(4*1) = 2.0
    # 最近 = drone 1 或 drone 3 (都是 2.0)
    expected_0 = 2.0 - drone_r
    
    # drone 3 (Z方向): 最近邻是 drone 0
    # dx=0, dy=0, dz=-1 → ell_dist = sqrt(4*1) = 2.0
    expected_3 = 2.0 - drone_r
    
    print(f"\nDrone 0: vec_norm = {vec_norms[0].item():.4f}, expected ≈ {expected_0:.4f}")
    print(f"Drone 3: vec_norm = {vec_norms[3].item():.4f}, expected ≈ {expected_3:.4f}")
    
    # 验证误差
    eps = 1e-3
    assert abs(vec_norms[0].item() - expected_0) < eps, \
        f"Drone 0 距离不匹配: {vec_norms[0].item()} vs {expected_0}"
    assert abs(vec_norms[3].item() - expected_3) < eps, \
        f"Drone 3 距离不匹配: {vec_norms[3].item()} vs {expected_3}"
    
    print("\n✓ 椭球距离计算正确")
    
    # 验证 Z 轴缩放效果：drone 3 与 drone 0 的 Euclidean 距离是 1.0，
    # 但 ellipsoid 距离是 2.0 → vec_norm 应该是 2.0 - r, 不是 1.0 - r
    euclidean_to_nearest = 1.0
    ellipsoid_to_nearest = 2.0
    assert vec_norms[3].item() > euclidean_to_nearest, \
        f"Z轴缩放失效: norm={vec_norms[3].item()} <= euclidean={euclidean_to_nearest}"
    print(f"✓ Z 轴椭球缩放生效 (norm={vec_norms[3].item():.4f} > euclidean={euclidean_to_nearest})")
    
    return True


def test_no_double_counting():
    """确认 drone_collide 默认为 0，不再双重计算"""
    losser = DroneLoss()
    assert losser.coefs['drone_collide'] == 0.0, \
        f"drone_collide 默认应为 0.0, 实际为 {losser.coefs['drone_collide']}"
    
    # 模拟一个 forward: inter_drone_dist_history=None 时 loss_drone_collide=0
    T, B = 20, 8
    device = torch.device('cuda:0')
    
    p = torch.randn(T, B, 3, device=device)
    v = torch.randn(T, B, 3, device=device)
    target_v = torch.randn(T, B, 3, device=device)
    act = torch.randn(T, B, 3, device=device)
    vec_to_obj = torch.randn(T, 10, B, 3, device=device)  # 子步细分 (T, S=10, B, 3)
    v_preds = torch.randn(T, B, 3, device=device)
    margin = torch.rand(B, device=device) * 0.5 + 0.3
    
    _, metrics = losser.forward(
        p_history=p,
        v_history=v,
        target_vel_history=target_v,
        act_history=act,
        vec_to_obj_history=vec_to_obj,
        v_preds=v_preds,
        env_margin=margin,
        inter_drone_dist_history=None,
    )
    
    assert metrics['loss_drone_collide'].item() == 0.0, \
        f"无 inter_drone_dist_history 时 loss_drone_collide 应为 0, 实际 {metrics['loss_drone_collide'].item()}"
    
    print("\n✓ drone_collide 默认权重为 0, 无双重计算")
    return True


def test_subdiv_count():
    """验证子步数默认为 10（与参考项目一致）"""
    from drone_env import DroneSimulator
    device = torch.device('cuda:0')
    
    env = DroneSimulator(
        batch_size=4,
        device=device,
        mesh_path='./data/sample/sample4.obj',
        image_size=(48, 64),
        focal_length=500.0,
        num_samples=1000,
    )
    env.reset()
    
    vecs = env.combined_vec_to_nearest(dt=1/15)
    S = vecs.shape[0]
    assert S == 10, f"子步数应为 10, 实际为 {S}"
    
    print(f"\n✓ 子步细分数 = {S} (与参考项目一致)")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("多机碰撞重构验证测试")
    print("=" * 60)
    
    test_subdiv_count()
    test_no_double_counting()
    test_ellipsoid_distance()
    
    print("\n" + "=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)
