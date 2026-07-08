"""Zero-behavior-change constructor for the legacy drone environment."""

from __future__ import annotations

from drone_env import DroneSimulator

from ..factory import EnvBuildContext, register_env


def _ensure_spawn_spacing(args) -> None:
    """Preserve the legacy collision-safe spacing calculation and messages."""

    margin_max = getattr(args, "margin_max", 0.8)
    min_sid = getattr(args, "min_spawn_inter_distance", 0.0)
    safe_min_sid = 2.0 * margin_max + 0.5
    arena_r = getattr(args, "arena_range", 6.0)
    geo_max = (2.0 * arena_r**2 / args.batch_size) ** 0.5
    if safe_min_sid > geo_max:
        print(
            f"[Info] min_spawn_inter_distance: 碰撞安全值 {safe_min_sid:.2f}m "
            f"> 几何上限 {geo_max:.2f}m (N={args.batch_size}, R={arena_r}), "
            f"裁剪至 {geo_max:.2f}m"
        )
        safe_min_sid = geo_max
    if min_sid <= 0:
        min_sid = safe_min_sid
    elif min_sid < safe_min_sid:
        print(
            f"[Warn] min_spawn_inter_distance={min_sid:.1f}m < 安全值={safe_min_sid:.1f}m, "
            f"自动提升至 {safe_min_sid:.1f}m 以避免出生即碰撞"
        )
        min_sid = safe_min_sid
    args.min_spawn_inter_distance = min_sid


def build_drone_env(context: EnvBuildContext) -> DroneSimulator:
    args = context.args
    _ensure_spawn_spacing(args)
    enable_airmode = context.extras.get("enable_airmode", True)

    return DroneSimulator(
        batch_size=args.batch_size,
        dt=context.control_dt,
        mesh_path=args.mesh_path,
        image_size=(args.image_height, args.image_width),
        focal_length=context.focal_length,
        device=context.device,
        enable_airmode=enable_airmode,
        enable_induced_drag=False,
        noise_std=getattr(args, "noise_std", 0.04),
        grad_decay=args.grad_decay,
        yaw_inertia=getattr(args, "yaw_inertia", 5.0),
        yaw_ctl_delay=getattr(args, "yaw_ctl_delay", 12.0),
        pitch_ctl_delay=getattr(args, "pitch_ctl_delay", 12.0),
        airmode_coef=getattr(args, "airmode_coef", 0.5),
        init_p_range=getattr(args, "init_p_range", 2.0),
        init_margin_range=(
            getattr(args, "margin_min", 0.3),
            getattr(args, "margin_max", 0.8),
        ),
        num_samples=args.num_samples,
        subdivide_times=args.subdivide_times,
        z_clip_value=getattr(args, "depth_min", 0.3),
        enable_random_scene=getattr(args, "random_scene", False),
        scene_generator=context.scene_generator,
        safe_spawn_clearance=getattr(args, "safe_clearance", 1.0),
        min_spawn_inter_distance=args.min_spawn_inter_distance,
        random_init_yaw=getattr(args, "random_init_yaw", True),
        cam_mode=getattr(args, "cam_mode", "auto"),
        cam_extrinsic=getattr(args, "cam_extrinsic", None),
        cam_mount_rpy=(
            getattr(args, "cam_mount_roll", 0.0),
            getattr(args, "cam_angle", 10),
            getattr(args, "cam_mount_yaw", 0.0),
        ),
        drone_mesh_path=getattr(args, "drone_mesh_path", None),
        aero_margin=getattr(args, "aero_margin", 0.05),
        max_drone_faces=getattr(args, "max_drone_faces", 500),
        n_drones_per_group=(
            args.n_drones_per_group
            if args.n_drones_per_group is not None
            else args.batch_size
        ),
        enable_dynamic_obstacles=getattr(args, "enable_dynamic_obstacles", False),
        num_dynamic_obstacles_range=(
            getattr(args, "num_dynamic_obstacles_min", 2),
            getattr(args, "num_dynamic_obstacles_max", 5),
        ),
        dynamic_obstacle_speed_range=(
            getattr(args, "dynamic_obs_speed_min", -0.5),
            getattr(args, "dynamic_obs_speed_max", 0.5),
        ),
        dynamic_obstacle_scale_range=(
            getattr(args, "dynamic_obs_scale_min", 0.2),
            getattr(args, "dynamic_obs_scale_max", 0.8),
        ),
    )


register_env("drone", build_drone_env)
