import torch, time, sys
sys.path.insert(0, '.')
from scene_generator import (
    SceneGenerator, sample_cross_map_spawn_target, sample_safe_points, sample_safe_targets
)
from pytorch3d.ops import sample_points_from_meshes

device = torch.device('cuda:0')
gen = SceneGenerator(device=device)

# Generate a scene to get obstacle_pcd
scene_mesh, obstacle_info = gen.generate()
obstacle_pcd = sample_points_from_meshes(scene_mesh, num_samples=10000)
print(f"obstacle_pcd shape: {obstacle_pcd.shape}")

N = 64
# Geometric cap: sqrt(2 * R^2 / N) = sqrt(72/64) ≈ 1.06
min_dist = (2.0 * 6.0**2 / N) ** 0.5
print(f"Using min_dist={min_dist:.3f}m (geometric cap for N={N}, R=6)")

# Warm-up
for _ in range(2):
    sample_cross_map_spawn_target(
        obstacle_pcd, num_points=N, min_inter_distance=min_dist, device=device
    )
torch.cuda.synchronize()

# Profile sample_cross_map_spawn_target
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = sample_cross_map_spawn_target(
        obstacle_pcd, num_points=N, min_inter_distance=min_dist, device=device
    )
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

avg = sum(times) / len(times)
print(f"sample_cross_map (N={N}, min_dist={min_dist}): "
      f"avg={avg*1000:.1f}ms  min={min(times)*1000:.1f}ms  max={max(times)*1000:.1f}ms")

# Profile sample_safe_points
spawn, _ = result
for _ in range(2):
    sample_safe_points(obstacle_pcd, N, min_inter_distance=min_dist, device=device)
torch.cuda.synchronize()

times2 = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pts = sample_safe_points(obstacle_pcd, N, min_inter_distance=min_dist, device=device)
    torch.cuda.synchronize()
    times2.append(time.perf_counter() - t0)

avg2 = sum(times2) / len(times2)
print(f"sample_safe_points (N={N}, min_dist={min_dist}): "
      f"avg={avg2*1000:.1f}ms  min={min(times2)*1000:.1f}ms  max={max(times2)*1000:.1f}ms")

# Profile sample_safe_targets
for _ in range(2):
    sample_safe_targets(obstacle_pcd, spawn, min_inter_distance=min_dist, device=device)
torch.cuda.synchronize()

times3 = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tgt = sample_safe_targets(obstacle_pcd, spawn, min_inter_distance=min_dist, device=device)
    torch.cuda.synchronize()
    times3.append(time.perf_counter() - t0)

avg3 = sum(times3) / len(times3)
print(f"sample_safe_targets (N={N}, min_dist={min_dist}): "
      f"avg={avg3*1000:.1f}ms  min={min(times3)*1000:.1f}ms  max={max(times3)*1000:.1f}ms")
