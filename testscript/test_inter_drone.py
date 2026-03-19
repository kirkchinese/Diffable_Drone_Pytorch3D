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
        init_p_range=0.0  # Put them near origin
    )
    
    # Manually place two drones exactly 0.5m apart, very high up to avoid static obstacles from sample.obj
    env.p[0] = torch.tensor([0.0, 0.0, 100.0])
    env.p[1] = torch.tensor([0.0, 0.5, 100.0])
    
    # Set their margins so they are overlapping
    env.margin[0] = 0.3
    env.margin[1] = 0.3
    
    # The effective distance between them should be:
    # d_ij = 0.5
    # eff_dist = d_ij - margin_j = 0.5 - 0.3 = 0.2
    # So vec_to_obj norm should be 0.2 (if static obstacle is further)
    
    dists, vecs = env._knn_query()
    print(f"Distances to closest object: {dists.tolist()}")
    
    vecs_subdiv = env.vec_to_obj_subdivided(n_subdiv=2)
    dist_subdiv = torch.norm(vecs_subdiv[0], dim=-1)
    print(f"Subdivided distances: {dist_subdiv.tolist()}")
    
    # They should be around 0.2
    assert abs(dists[0].item() - 0.2) < 1e-4, f"Expected 0.2, got {dists[0].item()}"
    assert abs(dists[1].item() - 0.2) < 1e-4, f"Expected 0.2, got {dists[1].item()}"
    print("Collision detection test passed!")

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
