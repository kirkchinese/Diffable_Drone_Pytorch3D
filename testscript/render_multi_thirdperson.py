#!/usr/bin/env python3
"""
Multi-Drone Third-Person Top-Down Visualization Script

Loads a multi-drone checkpoint, runs one full simulation episode, and renders
an MP4 video from a fixed third-person top-down camera that shows all 8 drones
and dynamic/static obstacles simultaneously.

Output:
  - ./viz_results/multi_thirdperson/multi_demo.mp4   (15 fps MP4 video)
  - ./viz_results/multi_thirdperson/trajectories.png  (top-down XY trajectory plot)

Usage:
  python testscript/render_multi_thirdperson.py

Requirements: CUDA-capable GPU (PyTorch3D does not support CPU rendering).
"""

import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- Ensure the project root is on the import path ----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pytorch3d.renderer import look_at_view_transform

from drone_env import DroneSimulator
from drone_renderer import (
    hfov_to_focal,
    build_cam_mount_R,
    transform_pos_ros2pt3d,
)
from model import Model_bigger
from scene_generator import SceneGenerator
from navigation_utils import DronePolicy

# ======================================================================
# Configuration  (aligned with multi_run_20260322 training checkpoint)
# ======================================================================

# Checkpoint candidates (tried in order)
_CHECKPOINT_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, 'checkpoints', 'multi_run_20260322', 'best_ar.pth'),
    os.path.join(_PROJECT_ROOT, 'checkpoints', 'multi_run_20260322', 'best_ar.pth'),
]

# Paths
DRONE_MESH_PATH = os.path.join(_PROJECT_ROOT, 'data', 'base_model', 'drone.obj')
BASE_MODEL_DIR = os.path.join(_PROJECT_ROOT, 'data', 'base_model')

# Multi-drone config
N_DRONES = 8
TIMESTEPS = 180
CONTROL_DT = 1.0 / 15.0
ARENA_RANGE = 10.0

# Model input (low-res, same as training)
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 64
MODEL_HFOV_DEG = 79.0
CAM_PITCH_DEG = 10.0

# Third-person camera
TP_ELEV_DEG = 60.0          # 0 = horizontal, 90 = straight down
TP_DISTANCE = 15.0           # metres from scene centre
TP_AZIM_DEG = 0.0            # 0 = view from +Z (ROS north) direction

# Video output
VIZ_HEIGHT = 480
VIZ_WIDTH = 640
VIZ_HFOV_DEG = 90.0
OUTPUT_FPS = 15
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'viz_results', 'multi_thirdperson')

# Policy
MAX_SPEED = 2.5
GPU_ID = 0
DEPTH_MIN = 0.3
DEPTH_MAX = 24.0


# ======================================================================
# Helpers
# ======================================================================


def _find_checkpoint(candidates):
    """Return the first existing checkpoint path from the candidate list."""
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'Checkpoint not found. Tried:\n' +
        '\n'.join(f'  {p}' for p in candidates)
    )


# ======================================================================
# Main
# ======================================================================


def main():
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type != 'cuda':
        print('[ERROR] PyTorch3D rendering requires a CUDA GPU.')
        sys.exit(1)

    checkpoint_path = _find_checkpoint(_CHECKPOINT_CANDIDATES)
    print(f'Checkpoint: {checkpoint_path}')

    # ------------------------------------------------------------------
    # 1. Scene generator  (matches training: random_scene, arena=10)
    # ------------------------------------------------------------------
    scene_gen = SceneGenerator(
        device=device,
        arena_range=ARENA_RANGE,
        num_obstacles_range=(40, 80),
        obstacle_scale_range=(0.3, 1.5),
        ground_ratio=0.6,
        cluster_ratio=0.3,
        cluster_spread=1.5,
    )

    # ------------------------------------------------------------------
    # 2. Focal length from horizontal FOV
    # ------------------------------------------------------------------
    focal_length = hfov_to_focal(MODEL_HFOV_DEG, IMAGE_WIDTH)
    print(f'Model camera: HFOV={MODEL_HFOV_DEG} deg  '
          f'focal={focal_length:.1f} px  {IMAGE_WIDTH}x{IMAGE_HEIGHT}')

    # ------------------------------------------------------------------
    # 3. DroneSimulator
    #    n_drones_per_group=8  means all 8 drones form a single group.
    #    This triggers the "single-group" code path where _update_render_scene
    #    places ALL drone meshes as dynamic meshes on the renderer, so the
    #    third-person camera sees every drone.
    # ------------------------------------------------------------------
    placeholder = os.path.join(BASE_MODEL_DIR, '球1_1.obj')

    env = DroneSimulator(
        batch_size=N_DRONES,
        dt=CONTROL_DT,
        device=device,
        mesh_path=placeholder,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        focal_length=focal_length,
        # Dynamics
        enable_airmode=True,
        noise_std=0.04,
        grad_decay=0.4,
        yaw_inertia=5.0,
        yaw_ctl_delay=12.0,
        pitch_ctl_delay=12.0,
        airmode_coef=0.5,
        init_p_range=ARENA_RANGE,
        init_margin_range=(0.3, 0.8),
        num_samples=50000,
        subdivide_times=0,
        z_clip_value=DEPTH_MIN,
        # Scene randomization
        enable_random_scene=True,
        scene_generator=scene_gen,
        safe_spawn_clearance=1.0,
        min_spawn_inter_distance=2.1,
        random_init_yaw=True,
        # Drone mesh / multi-drone interaction
        drone_mesh_path=DRONE_MESH_PATH,
        aero_margin=0.05,
        max_drone_faces=500,
        n_drones_per_group=N_DRONES,
        # Dynamic obstacles
        enable_dynamic_obstacles=True,
        num_dynamic_obstacles_range=(2, 5),
        dynamic_obstacle_speed_range=(-0.5, 0.5),
        dynamic_obstacle_scale_range=(0.2, 0.8),
        # Camera
        cam_mode='auto',
        cam_mount_rpy=(0.0, CAM_PITCH_DEG, 0.0),
    )
    print(f'Environment: {N_DRONES} drones, arena={ARENA_RANGE} m, '
          f'timesteps={TIMESTEPS}')

    # ------------------------------------------------------------------
    # 4. Third-person renderer variant  (high-res, wide FOV)
    #    Shares parent mesh/lights/obstacle_pcd via create_variant().
    # ------------------------------------------------------------------
    tp_renderer = env.renderer.create_variant(
        hfov_deg=VIZ_HFOV_DEG,
        image_size=(VIZ_HEIGHT, VIZ_WIDTH),
        z_clip_value=0.02,
    )
    print(f'Third-person renderer: HFOV={VIZ_HFOV_DEG} deg  '
          f'{VIZ_WIDTH}x{VIZ_HEIGHT}')

    # ------------------------------------------------------------------
    # 5. Load model checkpoint
    # ------------------------------------------------------------------
    model = Model_bigger(dim_obs=10, dim_action=6).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f'  version={ckpt.get("version", "?")}  '
              f'iteration={ckpt.get("iteration", "?")}  '
              f'best_ar={ckpt.get("best_ar", "?"):.4f}')
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [WARN] Missing keys: {missing}')
    if unexpected:
        print(f'  [WARN] Unexpected keys: {unexpected}')
    model.eval()
    print('Model: Model_bigger(dim_obs=10, dim_action=6)  loaded')

    # ------------------------------------------------------------------
    # 6. Policy wrapper
    # ------------------------------------------------------------------
    g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
    policy = DronePolicy(
        model=model,
        g_std=g_std,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        no_odom=False,
    )

    # ------------------------------------------------------------------
    # 7. Reset environment + randomise scene
    # ------------------------------------------------------------------
    env.reset()
    model.reset()

    if env.enable_random_scene:
        env.randomize_scene()
    if env.enable_dynamic_obstacles:
        env.randomize_dynamic_obstacles(arena_range=ARENA_RANGE)

    _, p_target = env.safe_reset_cross_map(
        arena_range=ARENA_RANGE,
        z_range=(1.0, 3.0),
    )
    p_start = env.p.clone()

    print('Start positions (ROS, first 3 drones):')
    print(p_start[:3].cpu().numpy())
    print('Target positions (ROS, first 3 drones):')
    print(p_target[:3].cpu().numpy())

    # ------------------------------------------------------------------
    # 8. Camera setup
    # ------------------------------------------------------------------

    # FPV camera mount rotation (per-drone body frame)
    cam_mount_R = build_cam_mount_R(
        roll_deg=0.0,
        pitch_deg=CAM_PITCH_DEG,
        yaw_deg=0.0,
        device=device,
        batch_size=N_DRONES,
    )

    # Third-person fixed-world camera
    #   PyTorch3D elev=60: camera sits 60 deg above the horizontal plane.
    #   Scene centre in PT3D coords ~ (0, 2.5, 0)  (mid-height of obstacles).
    scene_centre_pt3d = torch.tensor([[0.0, 2.5, 0.0]], device=device)
    up_pt3d = torch.tensor([[0.0, 1.0, 0.0]], device=device)

    R_tp, T_tp = look_at_view_transform(
        dist=TP_DISTANCE,
        elev=TP_ELEV_DEG,
        azim=TP_AZIM_DEG,
        at=scene_centre_pt3d,
        up=up_pt3d,
        device=device,
    )
    # R_tp: (1, 3, 3)    T_tp: (1, 3)
    print(f'Third-person camera: elev={TP_ELEV_DEG} deg  '
          f'dist={TP_DISTANCE} m  azim={TP_AZIM_DEG} deg')

    # ------------------------------------------------------------------
    # 9. Simulation loop
    # ------------------------------------------------------------------
    frames_rgb = []          # list of (H, W, 3) uint8 numpy arrays
    positions_all = []       # list of (B, 3) numpy arrays  (ROS coords)

    hx = None                # GRU hidden state
    act_lag = 1
    act_buffer = [env.act_curr.clone() for _ in range(act_lag + 1)]
    max_speed_t = torch.full((N_DRONES, 1), MAX_SPEED, device=device)
    thr_est_error = torch.ones(N_DRONES, 1, device=device)

    for t in tqdm(range(TIMESTEPS), desc='Simulating'):
        # --- FPV depth for policy input (low-res) ---
        #     env.render() internally calls _update_render_scene(), which
        #     sets all drone meshes + dynamic-obstacle meshes on the
        #     renderer.  The third-person variant shares these via
        #     self.parent.render_mesh, so it sees the same scene.
        _, depth_lo = env.render(
            cam_mount_R=cam_mount_R,
            return_tensor=True,
            return_rgb=False,
            return_depth=True,
        )

        # --- Record positions ---
        positions_all.append(env.p.cpu().detach().numpy().copy())

        # --- Third-person RGB render ---
        rgb_tp, _ = tp_renderer.render(
            R=R_tp,
            T=T_tp,
            return_tensor=True,
            return_rgb=True,
            return_depth=False,
        )
        # rgb_tp: (1, H, W, 3)  float in [0, 1]
        frame = (rgb_tp[0].clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        frames_rgb.append(frame)

        # --- Policy inference ---
        target_v_raw = p_target - env.p

        act_cmd, v_pred, target_v, hx = policy.infer(
            depth_lo,
            env.R,
            env.v,
            target_v_raw,
            env.margin,
            max_speed_t,
            thr_est_error,
            hx,
            depth_noise_std=0.0,
        )

        # --- Step the simulation ---
        env.step(
            act_cmd=act_buffer[t],
            target_pos_vector=target_v_raw,
        )
        act_buffer.append(act_cmd)

    positions_all = np.stack(positions_all, axis=0)  # (T, B, 3)

    # ------------------------------------------------------------------
    # 10. Save MP4 video
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DIR, 'multi_demo.mp4')

    import imageio
    writer = imageio.get_writer(
        video_path, fps=OUTPUT_FPS, codec='libx264', quality=8,
    )
    for frame in frames_rgb:
        writer.append_data(frame)
    writer.close()
    n_frames = len(frames_rgb)
    print(f'[Saved] Video: {video_path}  ({n_frames} frames, {OUTPUT_FPS} fps)')

    # ------------------------------------------------------------------
    # 11. Trajectory plot  (top-down XY plane)
    # ------------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, N_DRONES))
    p_start_np = p_start.cpu().numpy()
    p_target_np = p_target.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))

    for b in range(N_DRONES):
        # Full trajectory
        ax.plot(
            positions_all[:, b, 0],
            positions_all[:, b, 1],
            color=colors[b],
            linewidth=1.0,
            alpha=0.85,
            label=f'Drone {b}',
        )
        # Start (triangle)
        ax.scatter(
            p_start_np[b, 0], p_start_np[b, 1],
            color=colors[b], marker='^', s=120,
            edgecolors='k', linewidths=0.5, zorder=5,
        )
        # End (square)
        ax.scatter(
            positions_all[-1, b, 0], positions_all[-1, b, 1],
            color=colors[b], marker='s', s=80,
            edgecolors='k', linewidths=0.5, zorder=5,
        )
        # Target (star)
        ax.scatter(
            p_target_np[b, 0], p_target_np[b, 1],
            color=colors[b], marker='*', s=150,
            edgecolors='k', linewidths=0.3, zorder=4,
        )

    ax.set_xlabel('X (ROS, m)', fontsize=12)
    ax.set_ylabel('Y (ROS, m)', fontsize=12)
    ax.set_title(
        f'Multi-Drone Trajectories (Top-Down XY)\n'
        f'{N_DRONES} drones  |  {TIMESTEPS} steps  |  arena {ARENA_RANGE} m',
        fontsize=14,
    )
    ax.set_aspect('equal')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    margin = 1.5
    ax.set_xlim(-ARENA_RANGE - margin, ARENA_RANGE + margin)
    ax.set_ylim(-ARENA_RANGE - margin, ARENA_RANGE + margin)

    traj_path = os.path.join(OUTPUT_DIR, 'trajectories.png')
    fig.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] Trajectory plot: {traj_path}')

    # ------------------------------------------------------------------
    # 12. Per-drone navigation summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Per-Drone Navigation Summary')
    print('=' * 60)
    for b in range(N_DRONES):
        init_d = float(np.linalg.norm(p_start_np[b] - p_target_np[b]))
        dists = np.linalg.norm(
            positions_all[:, b] - p_target_np[b][None, :], axis=-1,
        )
        final_d = float(dists[-1])
        best_d = float(dists.min())
        reached = final_d <= 0.5
        print(
            f'  Drone {b}: init={init_d:.1f} m  '
            f'final={final_d:.2f} m  best={best_d:.2f} m  '
            f'reached={"YES" if reached else "no"}'
        )
    print('=' * 60)


if __name__ == '__main__':
    main()
