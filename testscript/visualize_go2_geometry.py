"""Generate auditable Go2 geometry, binding, camera, and terrain checks.

The deliberately asymmetric pose makes leg-order or joint-binding errors visible.
The report also compares the independent research mesh FK against the dynamics FK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.go2.aba import forward_kinematics  # noqa: E402
from diffsim.go2.dynamics import initial_state  # noqa: E402
from diffsim.go2.model import compile_go2_model  # noqa: E402
from diffsim.go2.render import Go2DepthRenderer  # noqa: E402
from diffsim.go2.scene import SHAPE_BOX, SHAPE_CAPSULE, SHAPE_SPHERE, random_scene  # noqa: E402
from diffsim.go2.types import Go2EnvConfig  # noqa: E402
from research.quadruped_migration.go2_3d.models.go2_render import Go2Twin  # noqa: E402


LEG_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def _save_mesh_binding(twin, model, output: Path, device: torch.device):
    nominal = model.default_joint_pos[None].to(device)
    asymmetric = nominal + nominal.new_tensor(
        [[0.20, -0.25, 0.15, -0.12, 0.18, -0.30, 0.08, 0.30, -0.12, -0.24, -0.10, 0.28]]
    )
    images = []
    for q in (nominal, asymmetric):
        rgb, _ = twin.render(q, image_size=480, dist=1.35, elev=16.0, azim=135.0, return_tensor=True)
        images.append(rgb[0].detach().cpu())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(
        axes,
        images,
        ("Nominal stance", "Asymmetric per-leg calibration pose"),
    ):
        axis.imshow(image.clamp(0, 1))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("URDF rigid mesh-to-link binding (hip / thigh / calf / foot colors)")
    fig.tight_layout()
    fig.savefig(output / "01_mesh_binding.png", dpi=170)
    plt.close(fig)
    return asymmetric


def _fk_and_collision_plot(twin, model, q, output: Path):
    state = initial_state(model, 1, base_height=0.0)
    state.joint_pos = q.to(state.joint_pos)
    dynamics_fk = forward_kinematics(model, state)
    reference_fk = twin.kin.forward(q.to(twin.device, twin.dtype))

    position_errors, rotation_errors = [], []
    for body, name in enumerate(model.body_names):
        reference = reference_fk[name][0]
        position_errors.append((dynamics_fk.body_position[0, body] - reference[:3, 3]).abs().max())
        rotation_errors.append((dynamics_fk.body_rotation[0, body] - reference[:3, :3]).abs().max())

    geometry = model.collisions
    owner = geometry.sample_owner
    samples = dynamics_fk.body_position[:, owner] + torch.einsum(
        "bsij,sj->bsi", dynamics_fk.body_rotation[:, owner], geometry.sample_pos
    )
    points = samples[0].detach().cpu().numpy()
    origins = dynamics_fk.body_position[0].detach().cpu().numpy()
    rotations = dynamics_fk.body_rotation[0].detach().cpu().numpy()

    fig = plt.figure(figsize=(11, 7))
    axis = fig.add_subplot(111, projection="3d")
    for body in range(model.n_bodies):
        color = "0.2" if body == 0 else LEG_COLORS[(body - 1) // 3]
        mask = owner.detach().cpu().numpy() == body
        axis.scatter(*points[mask].T, s=10, alpha=0.65, color=color)
        if body:
            parent = model.parent_indices[body - 1]
            axis.plot(*np.stack((origins[parent], origins[body])).T, color=color, linewidth=3)
            joint_axis_world = rotations[body] @ model.joint_axis[body - 1].detach().cpu().numpy()
            endpoint = origins[body] + 0.055 * joint_axis_world
            axis.quiver(*origins[body], *(endpoint - origins[body]), color="black", linewidth=1.2)
    axis.scatter(*origins.T, s=22, color="black", label="joint/body origins")
    axis.set_xlabel("+X forward [m]")
    axis.set_ylabel("+Y left [m]")
    axis.set_zlabel("+Z up [m]")
    axis.set_title("Dynamics skeleton, joint axes, and all 27 URDF collision groups\n"
                   "FL blue · FR orange · RL green · RR red")
    axis.set_box_aspect((0.9, 0.55, 0.6))
    axis.view_init(elev=20, azim=-130)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "02_skeleton_collision_binding.png", dpi=180)
    plt.close(fig)
    return max(x.item() for x in position_errors), max(x.item() for x in rotation_errors)


def _draw_box(axis, center, rotation, half_size, color):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = signs * half_size
    corners = corners @ rotation.T + center
    for i, a in enumerate(signs):
        for j, b in enumerate(signs):
            if j > i and np.count_nonzero(a != b) == 1:
                axis.plot(*np.stack((corners[i], corners[j])).T, color=color, linewidth=1)


def _camera_and_scene_plot(model, state, scene, renderer, output: Path):
    depth = renderer.render(state, scene)[0, 0].detach().cpu().numpy()
    terrain_vertices, _ = renderer._terrain_mesh(scene)
    vertices = terrain_vertices[0].detach().cpu().numpy()
    expected_height, _ = scene.terrain_height_and_normal(terrain_vertices[..., :2])
    terrain_error = (terrain_vertices[..., 2] - expected_height).abs().max().item()
    eye, at, up = renderer.camera_pose(state)
    eye_np, at_np, up_np = (item[0].detach().cpu().numpy() for item in (eye, at, up))

    fig = plt.figure(figsize=(13, 6))
    axis = fig.add_subplot(121, projection="3d")
    nx, ny = 28, 20
    x = vertices[:, 0].reshape(nx, ny)
    y = vertices[:, 1].reshape(nx, ny)
    z = vertices[:, 2].reshape(nx, ny)
    axis.plot_surface(x, y, z, cmap="terrain", alpha=0.68, linewidth=0)
    kinds = scene.obstacle_kind[0].detach().cpu().numpy()
    positions = scene.obstacle_position[0].detach().cpu().numpy()
    rotations = scene.obstacle_rotation[0].detach().cpu().numpy()
    sizes = scene.obstacle_size[0].detach().cpu().numpy()
    masks = scene.obstacle_mask[0].detach().cpu().numpy()
    for kind, position, rotation, size, mask in zip(kinds, positions, rotations, sizes, masks):
        if not mask:
            continue
        if kind == SHAPE_BOX:
            _draw_box(axis, position, rotation, size, "tab:red")
        else:
            extent = size[0] + (size[1] if kind == SHAPE_CAPSULE else 0.0)
            axis.scatter(*position, color="tab:purple" if kind == SHAPE_SPHERE else "tab:orange", s=120 * extent)
    forward = at_np - eye_np
    axis.scatter(*state.base_pos[0].detach().cpu().numpy(), s=50, color="black", label="base")
    axis.quiver(*eye_np, *forward, length=0.7, normalize=True, color="tab:blue", linewidth=3, label="camera +view")
    axis.quiver(*eye_np, *up_np, length=0.35, normalize=True, color="tab:green", linewidth=2, label="camera up")
    axis.set_xlim(-0.6, 4.5)
    axis.set_ylim(-1.1, 1.1)
    axis.set_zlim(-0.1, 0.9)
    axis.set_xlabel("+X forward")
    axis.set_ylabel("+Y left")
    axis.set_zlabel("+Z up")
    axis.set_title("Physics terrain/obstacles and chest-camera orientation")
    axis.view_init(elev=23, azim=-115)
    axis.legend(loc="upper right")

    depth_axis = fig.add_subplot(122)
    image = depth_axis.imshow(depth, cmap="magma", vmin=0.05, vmax=6.0)
    depth_axis.set_title("Actual chest depth input (48×64)\nnear=dark, far=yellow")
    depth_axis.set_xlabel("image +u")
    depth_axis.set_ylabel("image +v")
    fig.colorbar(image, ax=depth_axis, fraction=0.046, label="depth [m]")
    fig.tight_layout()
    fig.savefig(output / "03_camera_terrain_depth.png", dpi=180)
    plt.close(fig)
    return depth, eye_np, forward, terrain_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "viz_results" / "go2_geometry_validation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = Go2EnvConfig(batch_size=1)
    model = compile_go2_model(device=device)
    twin = Go2Twin(device=device, dtype=torch.float32)
    q = _save_mesh_binding(twin, model, args.output_dir, device)
    position_error, rotation_error = _fk_and_collision_plot(twin, model, q, args.output_dir)

    generator = torch.Generator(device=device).manual_seed(20260707)
    scene = random_scene(config, generator, device, torch.float32)
    state = initial_state(model, 1, base_height=0.34)
    renderer = Go2DepthRenderer(model, config, device)
    depth, eye, forward, terrain_error = _camera_and_scene_plot(
        model, state, scene, renderer, args.output_dir
    )
    forward = forward / np.linalg.norm(forward)
    report = {
        "joint_order": list(model.joint_names),
        "collision_groups": len(model.collisions.link_names),
        "collision_samples": int(model.collisions.sample_owner.numel()),
        "fk_max_position_error_m": position_error,
        "fk_max_rotation_abs_error": rotation_error,
        "terrain_render_physics_max_error_m": terrain_error,
        "camera_eye_world_m": eye.tolist(),
        "camera_forward_world": forward.tolist(),
        "camera_points_forward": bool(forward[0] > 0.95),
        "camera_points_down": bool(forward[2] < 0.0),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
    }
    assert position_error < 1e-5 and rotation_error < 1e-5
    assert terrain_error < 1e-6
    assert report["camera_points_forward"] and report["camera_points_down"]
    (args.output_dir / "geometry_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
