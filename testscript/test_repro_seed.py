#!/usr/bin/env python3
"""Check train.py-style seeding reproduces random scene and spawn sampling."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scene_generator import SceneGenerator, sample_safe_points


def seed_like_train(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_scene_and_spawn(seed):
    seed_like_train(seed)
    device = torch.device("cpu")
    sg = SceneGenerator(
        primitive_dir="./data/base_model",
        device=device,
        arena_range=3.0,
        num_obstacles_range=(4, 4),
        obstacle_scale_range=(0.3, 0.8),
        ground_ratio=0.5,
        cluster_ratio=0.25,
        max_faces_per_primitive=0,
        seed=seed,
    )
    scene_mesh, obstacle_info = sg.generate()
    obstacle_centers = torch.stack([info["center"].cpu() for info in obstacle_info])

    obstacle_pcd = scene_mesh.verts_packed().detach().cpu().unsqueeze(0)
    spawn = sample_safe_points(
        obstacle_pcd=obstacle_pcd,
        num_points=3,
        arena_range=3.0,
        z_range=(1.0, 2.0),
        min_clearance=0.2,
        device=device,
    ).cpu()
    return obstacle_centers, spawn


def main():
    obs0_a, spawn0_a = generate_scene_and_spawn(0)
    obs0_b, spawn0_b = generate_scene_and_spawn(0)
    obs1, spawn1 = generate_scene_and_spawn(1)

    assert torch.equal(obs0_a, obs0_b), "seed=0 obstacle positions are not bitwise-identical"
    assert torch.equal(spawn0_a, spawn0_b), "seed=0 spawn positions are not bitwise-identical"
    assert not torch.equal(obs0_a, obs1), "seed=0 and seed=1 obstacle positions did not differ"
    assert not torch.equal(spawn0_a, spawn1), "seed=0 and seed=1 spawn positions did not differ"

    print("[PASS] train-style seed reproduces scene/spawn bitwise and different seeds diverge")


if __name__ == "__main__":
    main()
