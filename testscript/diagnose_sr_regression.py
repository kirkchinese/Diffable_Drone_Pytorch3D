"""
诊断脚本：分析 SR 回归根因
=========================
1. 在 checkpoint 上跑 eval (no grad) → 基线 SR/metrics
2. 分析 loss 各项加权贡献 + 梯度范数
3. 检验假设：安全损失梯度是否压制导航梯度
"""
import torch
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from scene_generator import SceneGenerator, sample_cross_map_spawn_target
from drone_env import DroneSimulator
from model import Model_adaptive
from navigation_utils import (
    DronePolicy, compute_navigation_metrics_torch, compute_local_frame, preprocess_depth_for_model
)
from loss import DroneLoss
from pytorch3d.ops import sample_points_from_meshes


def run_eval_episode(env, policy, losser, args, device, p_target):
    """跑一个 episode，返回 loss 分解 + SR/AR metrics"""
    B = args['batch_size']
    env.reset()
    policy.reset()

    p_history, v_history, target_v_history = [], [], []
    target_dist_history, vec_to_pt_history = [], []
    inter_drone_dist_list, v_preds = [], []

    h = None
    act_lag = 1
    initial_act = env.act_curr.clone()
    act_buffer = [initial_act.clone() for _ in range(act_lag + 1)]

    target_v_raw = p_target - env.p
    max_speed = torch.full((B, 1), 2.0, device=device)
    thr_est_error = torch.ones((B, 1), device=device)

    from drone_renderer import build_cam_mount_R
    cam_mount_R = build_cam_mount_R(
        roll_deg=torch.zeros(B, device=device),
        pitch_deg=torch.full((B,), 10.0, device=device),
        yaw_deg=torch.zeros(B, device=device),
        device=device, batch_size=B,
    )
    cam_offset_body = env.get_scaled_cam_offset()

    dt = args['ctl_dt']
    T = args['timesteps']

    for t in range(T):
        with torch.no_grad():
            if t % 1 == 0:
                _, depth = env.render(
                    cam_mount_R=cam_mount_R, cam_offset_body=cam_offset_body,
                    return_tensor=True, return_rgb=False, return_depth=True, dt=dt)

            p_history.append(env.p.clone())
            vec_to_pt_history.append(env.combined_vec_to_nearest(dt=dt))

            if env.n_drones_per_group > 1:
                drone_dist, _ = env.inter_drone_distances()
                inter_drone_dist_list.append(drone_dist)

            target_v_raw = p_target - env.p
            target_dist_history.append(torch.norm(target_v_raw, p=2, dim=-1))

            env.step(act_cmd=act_buffer[t], target_pos_vector=target_v_raw, dt=dt)

            act_cmd, v_pred, target_v, h = policy.infer(
                depth, env.R, env.v, target_v_raw,
                env.margin, max_speed, thr_est_error, h, depth_noise_std=0.0)

            v_preds.append(v_pred)
            act_buffer.append(act_cmd)
            v_history.append(env.v.clone())
            target_v_history.append(target_v)

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    target_dist_history = torch.stack(target_dist_history)
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    v_preds = torch.stack(v_preds)
    act_buffer_stacked = torch.stack(act_buffer)
    inter_drone_dist_history = torch.stack(inter_drone_dist_list) if inter_drone_dist_list else None

    # Loss breakdown
    loss, metrics = losser.forward(
        p_history=p_history, v_history=v_history,
        target_vel_history=target_v_history, act_history=act_buffer_stacked,
        vec_to_obj_history=vec_to_pt_history, v_preds=v_preds,
        env_margin=env.margin, inter_drone_dist_history=inter_drone_dist_history,
    )

    # Navigation metrics
    distance = torch.norm(vec_to_pt_history, 2, -1) - env.margin
    speed_history = v_history.norm(2, -1)
    collision_history = distance <= 0
    nav = compute_navigation_metrics_torch(
        target_dist_history=target_dist_history,
        collision_history=collision_history,
        speed_history=speed_history,
        reach_radius=0.5,
    )
    metrics.update(nav)

    # Also compute per-drone breakdown
    with torch.no_grad():
        reached = (target_dist_history <= 0.5).any(dim=0)
        if collision_history.dim() == 3:
            coll_free = ~collision_history.flatten(0, 1).any(dim=0)
        else:
            coll_free = ~collision_history.any(dim=0)
        avg_speed_per_drone = speed_history.mean(dim=0)
        final_dist_per_drone = target_dist_history[-1]

    return loss, metrics, {
        'reached_per_drone': reached,
        'coll_free_per_drone': coll_free,
        'avg_speed': avg_speed_per_drone,
        'final_dist': final_dist_per_drone,
    }


def main():
    device = torch.device('cuda:0')

    # Params matching user's training config
    B = 64
    T = 200
    arena_range = 10.0
    ctl_dt = 1/15

    # Loss coefficients from user's command
    coefs = {
        'v': 1.0, 'v_pred': 2.0, 'collide': 3.0, 'obj_avoidance': 2.0,
        'd_acc': 0.01, 'd_jerk': 0.001, 'ground_affinity': 0.001,
        'lateral': 0.5, 'drone_collide': 5.0, 'speed': 0.0, 'bias': 0.0,
        'd_snap': 0.0,
    }

    args = {
        'batch_size': B, 'timesteps': T, 'ctl_dt': ctl_dt,
    }

    # Scene generator
    gen = SceneGenerator(device=device, arena_range=arena_range,
                         num_obstacles_range=(10, 50),
                         obstacle_scale_range=(0.3, 1.5),
                         ground_ratio=0.6, cluster_ratio=0.3)
    scene_mesh, obstacle_info = gen.generate()
    obstacle_pcd = sample_points_from_meshes(scene_mesh, num_samples=50000)

    # Environment  (match train.py constructor)
    from drone_renderer import hfov_to_focal
    focal_length = hfov_to_focal(90.0, 64)
    env = DroneSimulator(
        batch_size=B, dt=ctl_dt, device=device,
        mesh_path='./data/sample/sample4.obj',
        image_size=(48, 64),
        focal_length=focal_length,
        num_samples=50000,
        init_p_range=9.0,
        enable_random_scene=True,
        init_margin_range=(0.3, 0.8),
        grad_decay=0.6,
        noise_std=0.04,
        n_drones_per_group=8,
        drone_mesh_path='./data/base_model/drone.obj',
        enable_dynamic_obstacles=True,
        z_clip_value=0.3,
    )
    env.reset()
    env.update_scene(scene_mesh, obstacle_pcd)
    env.randomize_dynamic_obstacles(arena_range=arena_range)

    # Spawn/target
    spawn, target = sample_cross_map_spawn_target(
        obstacle_pcd, num_points=B, arena_range=arena_range,
        z_range=(1.0, 3.0), min_clearance=1.0,
        min_inter_distance=1.77, device=device)
    env.p = spawn.clone()
    env.v = torch.zeros_like(spawn)

    # Model
    model = Model_adaptive(dim_obs=10, dim_action=6, hidden_dim=256)
    ckpt = torch.load(
        './checkpoints/odom_run0_2025_02_13/checkpoint_003000.pth',
        map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
    policy = DronePolicy(model, g_std, depth_min=0.3, depth_max=24.0)

    losser = DroneLoss(
        coef_v=coefs['v'], coef_v_pred=coefs['v_pred'],
        coef_collide=coefs['collide'], coef_obj_avoidance=coefs['obj_avoidance'],
        coef_d_acc=coefs['d_acc'], coef_d_jerk=coefs['d_jerk'],
        coef_ground_affinity=coefs['ground_affinity'],
        coef_lateral=coefs['lateral'], coef_drone_collide=coefs['drone_collide'],
        coef_speed=coefs['speed'], coef_bias=coefs['bias'], coef_d_snap=coefs['d_snap'],
        ctl_dt=ctl_dt, window_size=30,
        loss_v_mode='adaptive', adaptive_decay_rate=2.0,
        ga_z_ceiling=5.0,
    )

    print("=" * 70)
    print("1. EVAL 基线 (no_grad, 模型不更新)")
    print("=" * 70)
    with torch.no_grad():
        loss, metrics, detail = run_eval_episode(env, policy, losser, args, device, target)

    print(f"\n  SR = {float(metrics['success_rate']):.1%}")
    print(f"  collision_free = {float(metrics['collision_free_rate']):.1%}")
    print(f"  reach_rate = {float(metrics['reach_rate']):.1%}")
    print(f"  goal_progress = {float(metrics['goal_progress']):.3f}")
    print(f"  avg_speed = {float(metrics['avg_speed']):.3f} m/s")
    print(f"  goal_dist_final = {float(metrics['goal_distance_final']):.2f} m")

    print(f"\n  Loss BREAKDOWN (raw value × coefficient = contribution):")
    loss_items = [
        ('v', 'loss_v'), ('lateral', 'loss_lateral'), ('v_pred', 'loss_v_pred'),
        ('collide', 'loss_collide'), ('obj_avoidance', 'loss_obj_avoidance'),
        ('d_acc', 'loss_d_acc'), ('d_jerk', 'loss_d_jerk'),
        ('ground_affinity', 'loss_ground_affinity'),
        ('drone_collide', 'loss_drone_collide'),
    ]
    total_check = 0.0
    for coef_key, metric_key in loss_items:
        raw = float(metrics.get(metric_key, 0))
        c = coefs.get(coef_key, 0)
        contrib = raw * c
        total_check += contrib
        print(f"    {coef_key:20s}  raw={raw:.4f}  ×  coef={c}  = {contrib:.4f}")
    print(f"    {'TOTAL':20s}  {total_check:.4f}  (reported: {float(loss):.4f})")

    # Per-drone breakdown
    print(f"\n  Per-drone detail:")
    coll_free = detail['coll_free_per_drone'].float()
    reached = detail['reached_per_drone'].float()
    speeds = detail['avg_speed']
    final_d = detail['final_dist']
    print(f"    collision_free: {coll_free.sum().item():.0f}/{B} = {coll_free.mean():.1%}")
    print(f"    reached_target: {reached.sum().item():.0f}/{B} = {reached.mean():.1%}")
    print(f"    avg speed: {speeds.mean():.3f} ± {speeds.std():.3f} m/s")
    print(f"    final dist to target: {final_d.mean():.2f} ± {final_d.std():.2f} m")

    # 2. Gradient magnitude analysis
    print("\n" + "=" * 70)
    print("2. 梯度分析 (1步训练迭代)")
    print("=" * 70)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    # Need a new episode with gradients
    env.reset()
    env.update_scene(scene_mesh, obstacle_pcd)
    env.randomize_dynamic_obstacles(arena_range=arena_range)
    env.p = spawn.clone()
    env.v = torch.zeros_like(spawn)

    # Simplified forward pass with grad
    h = None
    act_lag = 1
    initial_act = env.act_curr.clone()
    act_buffer = [initial_act.clone() for _ in range(act_lag + 1)]
    p_hist, v_hist, tv_hist, td_hist, vtp_hist, vpreds = [], [], [], [], [], []
    inter_dd = []

    from drone_renderer import build_cam_mount_R
    cam_R = build_cam_mount_R(
        roll_deg=torch.zeros(B, device=device),
        pitch_deg=torch.full((B,), 10.0, device=device),
        yaw_deg=torch.zeros(B, device=device),
        device=device, batch_size=B)
    cam_off = env.get_scaled_cam_offset()
    max_sp = 0.75 + 3.0 * torch.rand((B, 1), device=device)
    thr_err = 1.0 + 0.01 * torch.randn((B, 1), device=device)

    for t in range(T):
        with torch.no_grad():
            _, depth = env.render(cam_mount_R=cam_R, cam_offset_body=cam_off,
                                  return_tensor=True, return_rgb=False, return_depth=True, dt=ctl_dt)
        p_hist.append(env.p)
        vtp_hist.append(env.combined_vec_to_nearest(dt=ctl_dt))
        if env.n_drones_per_group > 1:
            dd, _ = env.inter_drone_distances()
            inter_dd.append(dd)
        target_v_raw = target - env.p.detach()
        td_hist.append(torch.norm(target_v_raw, p=2, dim=-1))
        env.step(act_cmd=act_buffer[t], target_pos_vector=target_v_raw, dt=ctl_dt)
        act_cmd, v_pred, target_v, h = policy.infer(
            depth, env.R, env.v, target_v_raw,
            env.margin, max_sp, thr_err, h, depth_noise_std=0.02)
        if t > 0 and t % 30 == 0:
            h = h.detach()
        vpreds.append(v_pred)
        act_buffer.append(act_cmd)
        v_hist.append(env.v)
        tv_hist.append(target_v)

    loss_train, metrics_train = losser.forward(
        p_history=torch.stack(p_hist), v_history=torch.stack(v_hist),
        target_vel_history=torch.stack(tv_hist),
        act_history=torch.stack(act_buffer),
        vec_to_obj_history=torch.stack(vtp_hist),
        v_preds=torch.stack(vpreds),
        env_margin=env.margin,
        inter_drone_dist_history=torch.stack(inter_dd) if inter_dd else None,
    )

    print(f"\n  Total loss = {float(loss_train):.4f}")
    for coef_key, metric_key in loss_items:
        raw = float(metrics_train.get(metric_key, 0))
        c = coefs.get(coef_key, 0)
        print(f"    {coef_key:20s}  raw={raw:.4f}  weighted={raw*c:.4f}")

    # Compute per-loss gradients
    print(f"\n  Per-loss GRADIENT NORM analysis:")
    loss_components = {
        'v': coefs['v'] * metrics_train['loss_v'],
        'collide': coefs['collide'] * metrics_train['loss_collide'],
        'obj_avoidance': coefs['obj_avoidance'] * metrics_train['loss_obj_avoidance'],
        'drone_collide': coefs['drone_collide'] * metrics_train.get('loss_drone_collide', torch.tensor(0.0)),
        'v_pred': coefs['v_pred'] * metrics_train['loss_v_pred'],
        'lateral': coefs['lateral'] * metrics_train.get('loss_lateral', torch.tensor(0.0)),
    }

    for name, component in loss_components.items():
        optimizer.zero_grad(set_to_none=True)
        if isinstance(component, torch.Tensor) and component.requires_grad:
            component.backward(retain_graph=True)
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            print(f"    {name:20s}  weighted_loss={float(component):.4f}  grad_norm={grad_norm:.4f}")
        else:
            print(f"    {name:20s}  weighted_loss={float(component):.4f}  (no grad)")

    # Total gradient
    optimizer.zero_grad(set_to_none=True)
    loss_train.backward()
    total_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.data.norm(2).item() ** 2
    total_grad = total_grad ** 0.5
    print(f"    {'TOTAL':20s}  grad_norm={total_grad:.4f}")

    print("\n" + "=" * 70)
    print("3. 结论")
    print("=" * 70)
    safety_grad_sum = sum(
        float(loss_components[k].detach()) if isinstance(loss_components[k], torch.Tensor) else 0
        for k in ['collide', 'obj_avoidance', 'drone_collide']
    )
    nav_grad_sum = sum(
        float(loss_components[k].detach()) if isinstance(loss_components[k], torch.Tensor) else 0
        for k in ['v', 'lateral']
    )
    print(f"  安全损失总权重贡献: {safety_grad_sum:.4f}")
    print(f"  导航损失总权重贡献: {nav_grad_sum:.4f}")
    print(f"  安全/导航 比值: {safety_grad_sum / max(nav_grad_sum, 1e-8):.1f}:1")


if __name__ == '__main__':
    main()
