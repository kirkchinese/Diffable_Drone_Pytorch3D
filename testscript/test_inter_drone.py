import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_env import DroneSimulator

def test_collision_detection():
    print("Testing inter-drone collision detection...")
    env = DroneSimulator(
        batch_size=2,
        dt=0.02,
        mesh_path="data/sample/sample.obj",
        image_size=(48, 64),
        init_p_range=0.0,  # Put them near origin
        n_drones_per_group=2,  # 同组才互相参与 inter-drone 距离
    )
    
    # Manually place two drones exactly 0.5m apart, very high up to avoid static obstacles from sample.obj
    env.p[0] = torch.tensor([0.0, 0.0, 100.0])
    env.p[1] = torch.tensor([0.0, 0.5, 100.0])
    
    # Set their margins so they are overlapping
    env.margin[0] = 0.3
    env.margin[1] = 0.3
    
    # 椭球间距 sqrt(dx^2+dy^2+4dz^2) = 0.5 (同 z, 仅 y 相差 0.5);
    # inter_drone_distances 减去双方 margin: 0.5 - 0.3 - 0.3 = -0.1 (重叠 = 碰撞)
    dists, vecs = env.inter_drone_distances()
    print(f"Inter-drone distances: {dists.tolist()}")

    # They should be -0.1 (overlapping by 0.1m after subtracting both margins)
    assert abs(dists[0].item() - (-0.1)) < 1e-4, f"Expected -0.1, got {dists[0].item()}"
    assert abs(dists[1].item() - (-0.1)) < 1e-4, f"Expected -0.1, got {dists[1].item()}"
    print("Inter-drone collision detection test passed!")

def test_visual_scaling():
    print("Testing dynamic visual scaling...")
    env = DroneSimulator(
        batch_size=2,
        dt=0.02,
        mesh_path="data/sample/sample.obj",
        drone_mesh_path="data/base_model/drone.obj",
        image_size=(48, 64)
    )
    env.margin[0] = 0.2
    env.margin[1] = 0.5
    
    # Render with inter-drone visibility
    rgb, depth = env.render(return_tensor=True)
    
    # We can check if the renderer used the correct scales by inspecting the code,
    # or just verifying it runs without error
    print("Visual scaling test passed (runs without error)!")

if __name__ == "__main__":
    test_collision_detection()
    test_visual_scaling()
