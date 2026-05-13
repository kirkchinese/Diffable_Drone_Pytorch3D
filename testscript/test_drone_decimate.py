"""
验证无人机网格减面 + 多机渲染性能
对比减面前后的面片数、渲染时间、bin overflow 情况
"""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_decimate_face_count():
    """验证减面后的面片数"""
    from scene_generator import _decimate_mesh
    from pytorch3d.io import load_objs_as_meshes
    
    device = torch.device('cuda:0')
    mesh = load_objs_as_meshes(['./data/base_model/drone.obj'], device=device)
    n_orig = mesh.faces_packed().shape[0]
    v_orig = mesh.verts_packed().shape[0]
    print(f"原始 drone.obj: {v_orig} verts, {n_orig} faces")
    
    for target in [100, 200, 300, 500]:
        m = _decimate_mesh(mesh, target)
        n = m.faces_packed().shape[0]
        v = m.verts_packed().shape[0]
        print(f"  target={target}: {v} verts, {n} faces ({(1-n/n_orig)*100:.1f}% 减少)")
    print()


def test_render_performance():
    """对比有/无drone mesh的渲染性能"""
    from drone_env import DroneSimulator
    
    device = torch.device('cuda:0')
    B = 16
    T_steps = 50  # 渲染帧数
    
    # ---- 无drone mesh (baseline) ----
    print("=== 无 drone mesh (baseline) ===")
    env_base = DroneSimulator(
        batch_size=B,
        mesh_path='./data/sample/sample4.obj',
        num_samples=50000,
        image_size=(48, 64),
        focal_length=32.0,
        z_clip_value=0.3,
        enable_random_scene=True,
        enable_dynamic_obstacles=True,
        num_dynamic_obstacles_range=(5, 5),
        n_drones_per_group=B,
        device=device,
    )
    env_base.reset()
    env_base.randomize_dynamic_obstacles()
    
    # 预热
    for _ in range(3):
        env_base.render(return_tensor=True, return_rgb=False, return_depth=True)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(T_steps):
        env_base.step(act_cmd=torch.zeros(B, 3, device=device))
        with torch.no_grad():
            _, depth = env_base.render(return_tensor=True, return_rgb=False, return_depth=True)
    torch.cuda.synchronize()
    t_base = (time.perf_counter() - t0) * 1000
    print(f"  {T_steps} 帧, 总计: {t_base:.0f} ms, 平均: {t_base/T_steps:.1f} ms/帧")
    del env_base
    torch.cuda.empty_cache()
    
    # ---- 有 drone mesh (减面后) ----
    print("\n=== 有 drone mesh (减面后, max 300 faces) ===")
    env_drone = DroneSimulator(
        batch_size=B,
        mesh_path='./data/sample/sample4.obj',
        num_samples=50000,
        image_size=(48, 64),
        focal_length=32.0,
        z_clip_value=0.3,
        enable_random_scene=True,
        enable_dynamic_obstacles=True,
        num_dynamic_obstacles_range=(5, 5),
        drone_mesh_path='./data/base_model/drone.obj',
        n_drones_per_group=B,
        device=device,
    )
    env_drone.reset()
    env_drone.randomize_dynamic_obstacles()
    
    # 显示实际面片数
    total_drone_faces = env_drone.drone_mesh.faces_packed().shape[0] * B
    print(f"  每个drone: {env_drone.drone_mesh.faces_packed().shape[0]} faces × {B} drones = {total_drone_faces} total")
    
    # 预热
    for _ in range(3):
        env_drone.render(return_tensor=True, return_rgb=False, return_depth=True)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(T_steps):
        env_drone.step(act_cmd=torch.zeros(B, 3, device=device))
        with torch.no_grad():
            _, depth = env_drone.render(return_tensor=True, return_rgb=False, return_depth=True)
    torch.cuda.synchronize()
    t_drone = (time.perf_counter() - t0) * 1000
    print(f"  {T_steps} 帧, 总计: {t_drone:.0f} ms, 平均: {t_drone/T_steps:.1f} ms/帧")
    
    overhead = (t_drone / t_base - 1) * 100
    print(f"\n=== 开销对比 ===")
    print(f"  无drone: {t_base/T_steps:.1f} ms/帧")
    print(f"  有drone: {t_drone/T_steps:.1f} ms/帧")
    print(f"  额外开销: {overhead:+.1f}%")
    del env_drone
    torch.cuda.empty_cache()


def test_no_bin_overflow():
    """验证不再出现 bin overflow"""
    import warnings
    from drone_env import DroneSimulator
    
    device = torch.device('cuda:0')
    B = 48  # 较大的 batch
    
    print("\n=== Bin overflow 测试 (B=48, 10 dynamic obs) ===")
    env = DroneSimulator(
        batch_size=B,
        mesh_path='./data/sample/sample4.obj',
        num_samples=50000,
        image_size=(48, 64),
        focal_length=32.0,
        z_clip_value=0.3,
        enable_random_scene=True,
        enable_dynamic_obstacles=True,
        num_dynamic_obstacles_range=(10, 10),
        drone_mesh_path='./data/base_model/drone.obj',
        n_drones_per_group=8,
        device=device,
    )
    env.reset()
    env.randomize_dynamic_obstacles()
    
    overflow_count = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for i in range(20):
            env.step(act_cmd=torch.zeros(B, 3, device=device))
            with torch.no_grad():
                _, depth = env.render(return_tensor=True, return_rgb=False, return_depth=True)
            # 检查是否有 bin overflow 警告
            for warning in w:
                if 'Bin size' in str(warning.message):
                    overflow_count += 1
        
    if overflow_count == 0:
        print("  ✅ 20帧渲染无 bin overflow")
    else:
        print(f"  ❌ 检测到 {overflow_count} 次 bin overflow!")
    
    del env
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print(" 无人机网格减面验证")
    print("=" * 60)
    print()
    
    test_decimate_face_count()
    test_render_performance()
    test_no_bin_overflow()
