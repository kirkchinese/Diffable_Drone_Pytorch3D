"""
集成测试：验证动态场景合成和无人机间渲染的正确性。

测试项目：
1. DroneRenderer 动态网格合成 (set_dynamic_meshes / render_mesh / full_obstacle_pcd)
2. DroneRendererVariant 透传动态网格
3. DroneSimulator 无人机机体渲染 (_compose_drone_meshes)
4. DroneSimulator 动态障碍物 (randomize_dynamic_obstacles / step)
5. combined_vec_to_nearest 包含无人机间碰撞
6. 训练循环 inter_drone_dist_history 传递到 loss
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()

def test_renderer_dynamic_meshes(device):
    """测试 DroneRenderer 的动态网格合成功能"""
    print("\n=== Test 1: DroneRenderer 动态网格合成 ===")
    from drone_renderer import DroneRenderer

    renderer = DroneRenderer(
        mesh_path='./data/sample/sample.obj',
        device=device,
        image_size=(48, 64),
        focal_length=50.0,
        num_samples=1000,
        subdivide_times=0,
    )

    # 基线：无动态网格
    base_mesh = renderer.render_mesh
    base_pcd = renderer.full_obstacle_pcd
    print(f"  静态 mesh verts: {base_mesh.verts_packed().shape[0]}")
    print(f"  静态 PCD shape: {base_pcd.shape}")
    assert renderer.render_mesh is renderer.mesh, "无动态网格时 render_mesh 应直接返回 mesh"
    assert renderer.full_obstacle_pcd is renderer.obstacle_pcd, "无动态PCD时直接返回 obstacle_pcd"

    # 添加动态网格
    from pytorch3d.utils import ico_sphere
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    sphere = ico_sphere(level=2, device=device)
    verts = sphere.verts_list()[0]
    sphere.textures = TexturesVertex(verts_features=torch.ones(1, verts.shape[0], 3, device=device))
    # 移到 (2, 0, 0)
    sphere_shifted = Meshes(
        verts=[verts + torch.tensor([2.0, 0.0, 0.0], device=device)],
        faces=[sphere.faces_packed()],
        textures=sphere.textures,
    )

    # 假的 PCD for dynamic obstacle (OBJ coords)
    fake_pcd = torch.randn(1, 100, 3, device=device)

    renderer.set_dynamic_meshes([sphere_shifted], [fake_pcd])

    rm = renderer.render_mesh
    fpcd = renderer.full_obstacle_pcd
    print(f"  合成后 render_mesh verts: {rm.verts_packed().shape[0]}")
    print(f"  合成后 full_obstacle_pcd shape: {fpcd.shape}")
    assert rm.verts_packed().shape[0] > base_mesh.verts_packed().shape[0], "合成后顶点应增加"
    assert fpcd.shape[1] == base_pcd.shape[1] + 100, "合成后PCD应包含动态部分"

    # 清除
    renderer.clear_dynamic_meshes()
    assert renderer.render_mesh is renderer.mesh, "清除后应恢复原始 mesh"
    assert renderer.full_obstacle_pcd is renderer.obstacle_pcd, "清除后PCD应恢复"

    print("  ✓ PASS")


def test_variant_proxy(device):
    """测试 DroneRendererVariant 能透传动态网格"""
    print("\n=== Test 2: DroneRendererVariant 透传 ===")
    from drone_renderer import DroneRenderer
    from pytorch3d.utils import ico_sphere
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    renderer = DroneRenderer(
        mesh_path='./data/sample/sample.obj',
        device=device,
        image_size=(48, 64),
        focal_length=50.0,
        num_samples=1000,
        subdivide_times=0,
    )
    variant = renderer.create_variant(image_size=(96, 128), hfov_deg=90.0)

    # 验证 variant 属性代理
    assert variant.mesh is renderer.mesh
    assert variant.render_mesh is renderer.render_mesh
    assert variant.obstacle_pcd is renderer.obstacle_pcd
    assert variant.full_obstacle_pcd is renderer.full_obstacle_pcd

    # 设置动态网格后 variant 也应看到
    sphere = ico_sphere(level=1, device=device)
    verts = sphere.verts_list()[0]
    sphere.textures = TexturesVertex(verts_features=torch.ones(1, verts.shape[0], 3, device=device))
    renderer.set_dynamic_meshes([sphere])

    assert variant.render_mesh.verts_packed().shape[0] > renderer.mesh.verts_packed().shape[0], \
        "variant.render_mesh 应包含动态网格"

    # 渲染测试
    from pytorch3d.renderer import look_at_view_transform
    R, T = look_at_view_transform(dist=5.0, elev=30, azim=0, device=device)
    rgb, depth = variant.render(R, T)
    assert rgb is not None and depth is not None
    print(f"  variant 渲染 RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

    renderer.clear_dynamic_meshes()
    print("  ✓ PASS")


def test_drone_mesh_rendering(device):
    """测试 DroneSimulator 在渲染中合成无人机机体"""
    print("\n=== Test 3: 无人机机体渲染合成 ===")
    from drone_env import DroneSimulator

    # 检查是否有 drone mesh 文件
    drone_mesh_path = './data/base_model/drone.obj'
    if not os.path.exists(drone_mesh_path):
        print(f"  [SKIP] 无人机网格不存在: {drone_mesh_path}")
        return

    env = DroneSimulator(
        batch_size=4,
        dt=1/15,
        mesh_path='./data/sample/sample.obj',
        image_size=(48, 64),
        focal_length=50.0,
        device=device,
        num_samples=1000,
        subdivide_times=0,
        drone_mesh_path=drone_mesh_path,
        n_drones_per_group=4,  # 全组交互
    )
    env.reset()

    # 渲染前先检查 render_mesh
    base_verts = env.renderer.mesh.verts_packed().shape[0]
    _, depth = env.render(camera_pitch=10.0, return_tensor=True, return_rgb=False, return_depth=True)

    # render() 内部调用 _update_render_scene()，应已把无人机网格加入
    rm_verts = env.renderer.render_mesh.verts_packed().shape[0]
    print(f"  静态场景顶点: {base_verts}")
    print(f"  合成后顶点 (含4架无人机): {rm_verts}")
    assert rm_verts > base_verts, "合成后顶点应多于静态场景"
    print(f"  深度图 shape: {depth.shape}")
    assert depth.shape == (4, 48, 64), f"深度图shape应为(4,48,64), 得到{depth.shape}"

    # hires variant 也应能渲染含无人机的场景
    hires = env.renderer.create_variant(image_size=(96, 128), hfov_deg=90.0)
    R_cam, T_cam = env.renderer.compute_view_matrix(
        p_ros=env.p, R_ros=env.R, camera_pitch_deg=10.0,
    )
    rgb_hi, depth_hi = hires.render(R_cam, T_cam)
    assert rgb_hi.shape == (4, 96, 128, 3)
    assert depth_hi.shape == (4, 96, 128)
    print(f"  高分辨率渲染: RGB {rgb_hi.shape}, Depth {depth_hi.shape}")

    print("  ✓ PASS")


def test_dynamic_obstacles(device):
    """测试动态障碍物生成和步进"""
    print("\n=== Test 4: 动态障碍物 ===")
    from drone_env import DroneSimulator

    env = DroneSimulator(
        batch_size=2,
        dt=1/15,
        mesh_path='./data/sample/sample.obj',
        image_size=(48, 64),
        focal_length=50.0,
        device=device,
        num_samples=1000,
        subdivide_times=0,
        enable_dynamic_obstacles=True,
        num_dynamic_obstacles_range=(3, 3),
        dynamic_obstacle_speed_range=(-1.0, 1.0),
        dynamic_obstacle_scale_range=(0.3, 0.5),
    )
    env.reset()

    # 随机化动态障碍物
    env.randomize_dynamic_obstacles(arena_range=5.0)
    assert len(env._dynamic_obstacles) == 3, f"应有3个动态障碍物, 得到{len(env._dynamic_obstacles)}"

    # 渲染应包含动态障碍物
    _, depth0 = env.render(camera_pitch=10.0, return_tensor=True, return_rgb=False, return_depth=True)
    fpcd0 = env.renderer.full_obstacle_pcd.shape[1]
    print(f"  full_obstacle_pcd 采样点: {fpcd0} (含动态障碍物)")
    assert fpcd0 > env.renderer.obstacle_pcd.shape[1], "full_obstacle_pcd 应多于 static obstacle_pcd"

    # 步进后位置应变化
    pos_before = [obs.position.clone() for obs in env._dynamic_obstacles]
    env.step(
        act_cmd=torch.zeros(2, 3, device=device),
        target_pos_vector=torch.ones(2, 3, device=device),
        dt=1/15,
    )
    pos_after = [obs.position.clone() for obs in env._dynamic_obstacles]
    any_moved = False
    for pb, pa in zip(pos_before, pos_after):
        if not torch.allclose(pb, pa, atol=1e-6):
            any_moved = True
    assert any_moved, "step() 后动态障碍物位置应变化"
    print("  ✓ 动态障碍物位置在 step() 后更新")

    # 清除
    env.clear_dynamic_obstacles()
    assert len(env._dynamic_obstacles) == 0
    assert env.renderer.full_obstacle_pcd is env.renderer.obstacle_pcd
    print("  ✓ PASS")


def test_combined_vec_to_nearest(device):
    """测试 combined_vec_to_nearest 融合静态障碍物和无人机间碰撞"""
    print("\n=== Test 5: combined_vec_to_nearest ===")
    from drone_env import DroneSimulator

    env = DroneSimulator(
        batch_size=4,
        dt=1/15,
        mesh_path='./data/sample/sample.obj',
        image_size=(48, 64),
        focal_length=50.0,
        device=device,
        num_samples=1000,
        subdivide_times=0,
        n_drones_per_group=4,
    )
    env.reset()

    # 单纯静态障碍物
    vecs_obs = env.vec_to_obj_subdivided(dt=1/15)
    print(f"  vec_to_obj_subdivided shape: {vecs_obs.shape}")

    # 融合 (含无人机间)
    vecs_combined = env.combined_vec_to_nearest(dt=1/15)
    print(f"  combined_vec_to_nearest shape: {vecs_combined.shape}")
    assert vecs_combined.shape == vecs_obs.shape

    # 无人机间距离
    drone_dist, drone_vec = env.inter_drone_distances()
    print(f"  inter_drone_distances: min={drone_dist.min():.3f}, max={drone_dist.max():.3f}")

    # 当 n_drones_per_group=1 时，combined 应等于 obs-only
    env2 = DroneSimulator(
        batch_size=4, dt=1/15,
        mesh_path='./data/sample/sample.obj',
        image_size=(48, 64), focal_length=50.0,
        device=device, num_samples=1000, subdivide_times=0,
        n_drones_per_group=1,
    )
    env2.reset()
    env2.p = env.p.clone()
    env2.v = env.v.clone()
    vecs_single = env2.combined_vec_to_nearest(dt=1/15)
    vecs_obs_single = env2.vec_to_obj_subdivided(dt=1/15)
    assert torch.allclose(vecs_single, vecs_obs_single, atol=1e-5), \
        "n_drones_per_group=1 时 combined 应等于 obs-only"
    print("  ✓ n_drones_per_group=1 时 combined == obs-only")

    print("  ✓ PASS")


def test_loss_inter_drone(device):
    """测试 inter_drone_dist_history 传递到 loss"""
    print("\n=== Test 6: Loss inter_drone_dist_history ===")
    from loss import DroneLoss

    losser = DroneLoss(coef_drone_collide=5.0)

    T, B = 10, 4
    p = torch.randn(T, B, 3, device=device) * 0.1
    v = torch.randn(T, B, 3, device=device) * 0.5
    target_v = torch.randn(T, B, 3, device=device)
    acts = torch.randn(T + 2, B, 3, device=device)
    vecs = torch.randn(T, B, 3, device=device)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True) * 2.0  # 远离障碍物
    v_preds = torch.randn(T, B, 3, device=device)
    margin = torch.full((B,), 0.3, device=device)

    # 无 inter_drone_dist 时
    loss1, m1 = losser.forward(p, v, target_v, acts, vecs, v_preds, margin)
    assert m1['loss_drone_collide'].item() == 0.0, "无 inter_drone 时 loss_drone_collide 应为 0"

    # 有 inter_drone_dist 时
    drone_dist = torch.rand(T, B, device=device) * 0.5  # 近距离
    loss2, m2 = losser.forward(p, v, target_v, acts, vecs, v_preds, margin,
                       inter_drone_dist_history=drone_dist)
    assert m2['loss_drone_collide'].item() > 0, "有近距离 inter_drone 时 loss_drone_collide 应 > 0"
    print(f"  loss_drone_collide: {m2['loss_drone_collide'].item():.4f}")

    print("  ✓ PASS")


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_renderer_dynamic_meshes(device)
    test_variant_proxy(device)
    test_drone_mesh_rendering(device)
    test_dynamic_obstacles(device)
    test_combined_vec_to_nearest(device)
    test_loss_inter_drone(device)

    print("\n" + "=" * 50)
    print("  所有测试通过 ✓")
    print("=" * 50)


if __name__ == '__main__':
    main()
