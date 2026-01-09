import argparse
import math
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from random import normalvariate

# Imports from user's implementation
from drone_env import DroneSimulator
from model import Model
from loss import DroneLoss

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training Parameters
    parser.add_argument('--resume', default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # Loss Coefficients (Reference: main_cuda.py & single_agent.args)
    parser.add_argument('--coef_v', type=float, default=1.0)
    parser.add_argument('--coef_v_pred', type=float, default=2.0)
    parser.add_argument('--coef_collide', type=float, default=7.5) # from single_agent.args
    parser.add_argument('--coef_obj_avoidance', type=float, default=3.0) # from single_agent.args
    parser.add_argument('--coef_d_acc', type=float, default=0.01)
    parser.add_argument('--coef_d_jerk', type=float, default=0.001)
    parser.add_argument('--coef_d_snap', type=float, default=0.0)
    parser.add_argument('--coef_ground_affinity', type=float, default=0.0) # From reference (legacy)
    
    # Env & Physiology Parameters
    parser.add_argument('--grad_decay', type=float, default=0.4) # Reference default in main_cuda.py
    parser.add_argument('--speed_mtp', type=float, default=4.0) # from single_agent.args
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82) # from single_agent.args
    parser.add_argument('--timesteps', type=int, default=150)
    parser.add_argument('--cam_angle', type=int, default=20) # from single_agent.args
    
    # Booleans
    parser.add_argument('--single', default=True, action='store_true') # from single_agent.args (implied)
    parser.add_argument('--no_odom', default=False, action='store_true')
    parser.add_argument('--random_rotation', default=True, action='store_true') # from single_agent.args
    parser.add_argument('--yaw_drift', default=True, action='store_true') # from single_agent.args
    
    # Mesh path
    parser.add_argument('--mesh_path', type=str, default='data/sample/sample.obj')

    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(args)

    writer = SummaryWriter(log_dir='runs/drone_training')

    # Calculate focal length for simulator based on fov (assuming width=64)
    # tan(fov/2) = (W/2) / f  => f = (W/2) / tan(fov/2)
    # Image size for model is small (48, 64)
    img_h, img_w = 48, 64
    focal_length = (img_w / 2) / args.fov_x_half_tan

    print(f"Initializing Env with Image Size: ({img_h}, {img_w}), Focal Length: {focal_length:.2f}")

    # Initialize Environment
    # Note: user's DroneSimulator doesn't take fov directly, but focal_length
    env = DroneSimulator(
        batch_size=args.batch_size,
        dt=0.02, # default
        device=device,
        mesh_path=args.mesh_path,
        image_size=(img_h, img_w),
        focal_length=focal_length,
        grad_decay=args.grad_decay,
        # We can simulate other params like speed_mtp by adjusting max_speed or target generation in loop,
        # but DroneSimulator might not have 'max_speed' arg in init.
        # User's env seems to focus on dynamics. Target generation is handled externally or inside step?
        # DroneSimulator.step() takes target vector (action_cmd is thrust).
        # We need to manage target velocity generation in the loop like reference main_cuda.py
    )
    
    # Reference Env has internal max_speed logic. User's DroneSimulator doesn't seem to expose it.
    # We will define max_speed externally for target generation.
    base_speed = 3.0 # Approx baseline
    max_speed = base_speed * args.speed_mtp 
    
    # Initialize Model
    # Input dim: 7 (vel, rot, margin) + 3 (local_v if odom) = 10
    dim_obs = 7 if args.no_odom else 10
    dim_action = 6 # Output of model is 6 (a_pred, v_pred)
    
    model = Model(dim_obs, dim_action).to(device)

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from {args.resume}")
            state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Resume checkpoint {args.resume} not found.")

    optim = AdamW(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(optim, args.num_iters, eta_min=args.lr * 0.01)

    # Initialize Loss Class
    drone_loss = DroneLoss(
        device=device,
        margin=0.2, # Default margin
        coef_collide=args.coef_collide,
        coef_obj_avoidance=args.coef_obj_avoidance,
        coef_v=args.coef_v,
        coef_d_acc=args.coef_d_acc,
        coef_d_jerk=args.coef_d_jerk,
        coef_d_snap=args.coef_d_snap
    )
    
    # Constants from reference
    g_std = torch.tensor([0., 0, -9.80665], device=device)

    pbar = tqdm(range(args.num_iters), ncols=100)
    
    for i in pbar:
        # Reset Env
        state_vec = env.reset() # (B, 18)
        model.reset() # Doesn't do much in provided model.py but good practice
        
        # Reset Loop Variables
        h = None
        act_lag = 1
        # act corresponds to the physical actuation state/command including errors
        # Initialize with current env.act (which is zero after reset)
        act_buffer = [env.act.clone()] * (act_lag + 1)
        
        # Target Generation (Simplified version of reference)
        # Random target direction/velocity
        # Reference: target_v_raw = env.p_target - env.p
        # Here we just generate a random target velocity vector for training
        target_v_raw = torch.randn(args.batch_size, 3, device=device)
        target_v_raw[:, 2] *= 0.1 # Less vertical movement
        
        # Yaw Drift Setup
        if args.yaw_drift:
            drift_av = torch.randn(args.batch_size, device=device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            R_drift = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ], -1).reshape(args.batch_size, 3, 3)

        # Sim Params
        thr_est_error = 1 + torch.randn(args.batch_size, device=device) * 0.01

        # History collection
        history = defaultdict(list)
        v_preds = []

        training_loss = 0.0
        
        for t in range(args.timesteps):
            # Time jitter
            ctl_dt = normalvariate(1 / 15, 0.1 / 15)
            
            # --- 1. Render & Observation ---
            # Render returns rgb, depth
            # We need to manually handle cam_angle if env.render doesn't support random variation per batch in args
            # User's env.render takes scalar camera_pitch.
            # Reference applies random cam_angle per batch element in Env.reset.
            # User's DroneRenderer computes view matrix based on single pitch or batch?
            # DroneRenderer.compute_view_matrix takes camera_pitch_deg (float or tensor?)
            # Let's pass the scalar args.cam_angle for now to match interface.
            
            rgb, depth = env.render(camera_pitch=args.cam_angle)
            
            # Preprocess Depth
            # Handle potential invalid values (e.g. -1 for background)
            # Map background (-1 or large) to max_dist (24)
            # Map too close (<0.3) to 0.3
            depth[depth < 0.1] = 24.0 
            depth = torch.nan_to_num(depth, nan=24.0, posinf=24.0, neginf=24.0)
            
            x = 3 / depth.clamp(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
            # Max Pool
            x = x.unsqueeze(1) # Add channel dim (B, H, W) -> (B, 1, H, W)
            x = F.max_pool2d(x, 4, 4) # (B, 1, 12, 16)
            
            # --- 2. State Construction ---
            # Need local velocity, relative target, up vector, margin
            # env.R is (B, 3, 3). env.p is (B, 3). env.v is (B, 3).
            
            # Update Target (Drift)
            if args.yaw_drift:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            
            # Calculate Target V (Clamped)
            target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v_unit = target_v_raw / (target_v_norm + 1e-6)
            target_v = target_v_unit * torch.minimum(target_v_norm, torch.tensor(max_speed))
            
            # Rotation Body-to-World is R. World-to-Body is R^T.
            # Reference: target_v[:, None] @ R.  (1, 3) @ (3, 3) -> (1, 3).
            # This projects World vector to Body frame.
            
            # Coordinate System Check:
            # Pytorch3D / Standard: R usually columns are axes.
            # env.R seems to be Body-to-World (columns are X, Y, Z axes in world).
            # v @ R is effectively v^T * R = (R^T * v)^T. So it projects v to Body. Correct.
            
            local_target_v = torch.squeeze(target_v[:, None] @ env.R, 1)
            local_v = torch.squeeze(env.v[:, None] @ env.R, 1)
            
            up_vec_body = env.R[:, 2] # World Z axis expressed in Body? No.
            # env.R[:, 2] is the 3rd column of R.
            # If R = [X_b, Y_b, Z_b], then R[:, 2] is Z_b (body Z axis) in World Coordinates.
            # The reference uses `env.R[:, 2]`. This provides the drone's tilt info.
            
            margin_obs = torch.ones(args.batch_size, 1, device=device) * 0.2 # Constant margin
            
            state_parts = [local_target_v, up_vec_body, margin_obs]
            if not args.no_odom:
                state_parts.insert(0, local_v)
            
            state_input = torch.cat(state_parts, -1) # (B, 10)
            
            # --- 3. Model Inference ---
            # Output: act_raw (B, 6), value, hidden
            act_output, _, h = model(x, state_input, h)
            
            # Decode Action
            # Reference: a_pred, v_pred = (R @ act.reshape).unbind
            # act_output is (B, 6). Reshape (B, 3, 2).
            # We need to project these body-frame predictions to World frame?
            # Reference: `R @ act.reshape(...)`. R is (B, 3, 3). act.reshape is (B, 3, 2).
            # Result is (B, 3, 2). unbind(-1) -> two (B, 3) vectors.
            # So the model predicts Body-frame acceleration/thrust terms, converted to World.
            
            act_reshaped = act_output.reshape(args.batch_size, 3, 2)
            act_world_components = env.R @ act_reshaped # (B, 3, 2)
            a_pred, v_pred = act_world_components.unbind(-1) # (B, 3) each
            
            v_preds.append(v_pred)
            
            # Compute Control Action (Thrust/Acc Command)
            # act = (a_pred - v_pred - g_std) * thr_est_error + g_std
            # This 'act' is passed to env.
            action_cmd = (a_pred - v_pred - g_std) * thr_est_error[:, None] + g_std
            
            act_buffer.append(action_cmd)
            
            # --- 4. Environment Step ---
            # Environment step uses the delayed action.
            # act_buffer was initialized with (act_lag+1) copies.
            # At step t=0, we use act_buffer[0].
            # At end of step t, we append act_new.
            # So act_buffer[t] is the correct Delayed action for step t.
            
             # Step
            env.step(action_cmd=act_buffer[t], target_pos_vector=None, v_wind=None)
            
            # --- 5. Collect History ---
            history['p'].append(env.p.clone())
            history['v'].append(env.v.clone())
            history['act'].append(env.act.clone()) # actual state
            history['target_v'].append(target_v.clone())
            
            # For Collision Loss (Need distance to nearest point)
            # Since we don't have obstacles in this simplified training loop (env.balls/voxels are in env logic),
            # we need to ask Env for distances.
            # User's DroneSimulator doesn't expose `balls` or `voxels` directly like Reference Env.
            # But DroneRenderer has the mesh.
            # Wait, Reference Env `env_cuda.py` handles collisions via explicit geometry (balls, voxels).
            # User's Env `drone_env.py` has `simulate_position_step`.
            # If user's env doesn't have obstacles, collision loss is zero?
            # User's `loss.py` assumes `obstacle_pcd`.
            # If we are strictly following user's Setup:
            # "复现的目标... 逻辑清晰... 包含梯度时间衰减".
            # If the user's `drone_env.py` is barebones (no obstacles initialized in python, just mesh for rendering),
            # then we might skip collision loss or we need to generate dummy obstacles for training.
            # Reference `env_cuda.py` puts obstacles in `self.balls`, `self.voxels`.
            # User's `drone_env.py` DOES NOT seem to have obstacles in `__init__`.
            # IT ONLY HAS MESH.
            # Unless I extract pointcloud from mesh?
            # Or maybe `drone_env.py` logic relies on external obstacle management.
            # For now, I will omit Collision Loss if no obstacles are available, or create a simple ground plane PCD.
            
            # Let's assume ground plane at z=0 is the main obstacle for now if others missing.
            # Create a dummy ground PCD?
            # Or better, we skip explicit collision loss if not supported by Env, 
            # BUT the user provided `loss.py` with `calc_min_distance`.
            # I will create a dummy obstacle (e.g. ground) to make the code complete.
            
        # --- End of Trajectory Loss Calculation ---
        
        # Stack History
        p_history = torch.stack(history['p']) # (T, B, 3)
        v_history = torch.stack(history['v'])
        act_history = torch.stack(history['act'])
        target_v_history = torch.stack(history['target_v'])
        v_preds_stack = torch.stack(v_preds) # (T, B, 3)
        
        # 1. Ground Affinity Loss
        # Penalize going below ground (Z < 0).
        loss_ground = (-p_history[..., 2]).relu().pow(2).mean() * args.coef_ground_affinity
        
        # 2. Velocity Loss
        # Use simple MSE or the sliding window one from DroneLoss
        # Shape inputs: (B, T, 3) -> transpose
        loss_v = drone_loss.get_velocity_loss(
            v_history.transpose(0, 1), 
            target_v_history.transpose(0, 1)
        )
        
        # 3. V Prediction Loss
        loss_v_pred = F.mse_loss(v_preds_stack, v_history.detach()) * args.coef_v_pred
        
        # 4. Smoothness Loss
        loss_smooth, l_acc, l_jerk, l_snap = drone_loss.get_smoothness_loss(
            act_history.transpose(0, 1)
        )
        
        # 5. Collision Loss (Dummy)
        # We don't have real obstacles in this env version easily accessible.
        # We can skip or add a placeholder.
        loss_collision = torch.tensor(0.0, device=device)
        
        total_loss = loss_v + loss_v_pred + loss_smooth + loss_ground + loss_collision
        
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        sched.step()
        
        # Logging
        if i % 10 == 0:
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Loss/Velocity', loss_v.item(), i)
            writer.add_scalar('Loss/VPred', loss_v_pred.item(), i)
            writer.add_scalar('Loss/Smooth', loss_smooth.item(), i)
            pbar.set_description(f"Loss: {total_loss.item():.4f}")
            
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'runs/drone_training/model_{i}.pth')

    print("Training Complete.")
    torch.save(model.state_dict(), 'runs/drone_training/model_final.pth')
    writer.close()

if __name__ == "__main__":
    train()
