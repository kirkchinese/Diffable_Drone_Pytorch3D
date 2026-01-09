
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
import math

# 项目导入
from drone_env import DroneSimulator
from drone_dynamics import simulate_position_step
from drone_renderer import DroneRenderer
from loss import calc_min_distance, calc_interpolated_distances

# 直接从参考项目导入模型
import sys
sys.path.append("参考项目/DiffPhysDrone")
from model import Model

def barrier(dists, visible_dist=4.0):
    """
    障碍函数损失 (Barrier Loss)
    当距离小于 margin 时，损失急剧上升。
    DiffPhysDrone 参考实现：(v_to_pt * (1 - x).relu().pow(2)).mean()
    这里简化实现。
    """
    # 距离越小，损失越大
    # dists: (B, steps)
    # 归一化距离: x = dist / visible_dist
    # loss = (1 - x)^2 if x < 1 else 0
    x = dists / visible_dist
    return F.relu(1.0 - x).pow(2).mean()

def train():
    # ---------------- 设置 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数
    # 参考项目默认 Batch 64
    BATCH_SIZE = 128  
    # 渲染图像分辨率 (参考项目 64x48, 这里保持较小但稍微增大以测试)
    RES_H, RES_W = 48, 64 
    # 原始焦距为 640 宽度下的 500。按新宽度缩放。
    # f_new = f_old * (w_new / w_old) = 500 * (64 / 640) = 50.0
    FOCAL_LENGTH = 50.0  # 新焦距
    
    DT = 0.02  # 环境时间步长
    # 参考项目 time steps ~150
    TRAIN_HORIZON = 100 
    NUM_ITERS = 20000 # 训练总迭代次数
    LR = 1e-3
    GRAD_DECAY = 0.8   # 梯度时间衰减因子
    
    # 损失权重 (参考 DiffPhysDrone 配置)
    COEF_VEL = 1.0           # 速度跟踪
    COEF_OBJ_AVOID = 2.0     # 避障 (Barrier)
    COEF_COLLISION = 5.0     # 碰撞惩罚 (Softplus/Relu)
    COEF_REG = 0.001        # 动作正则化
    COEF_SMOOTH = 0.001     # 平滑度 (Jerk)

    # 初始化环境
    env = DroneSimulator(
        batch_size=BATCH_SIZE,
        dt=DT,
        device=device,
        mesh_path="data/sample/sample.obj",
        image_size=(RES_H, RES_W),
        focal_length=FOCAL_LENGTH,
        grad_decay=GRAD_DECAY,
        # 动力学参数 (参考项目默认开启)
        enable_induced_drag=True
    )
    
    # 初始化模型
    # 参考项目输入：
    # 1. 深度图 (处理后)
    # 2. 状态向量 (10维): [Target_V_Body(3), Body_Z_World(3), Margin(1), Local_V_Body(3)]
    # 这里我们简化去掉 Margin (假设固定), 保留 9 维
    dim_obs = 9 
    dim_action = 3 # 加速度命令 (x, y, z)
    model = Model(dim_obs=dim_obs, dim_action=dim_action).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_ITERS)
    writer = SummaryWriter("runs/drone_training")
    
    # ---------------- 训练循环 ----------------
    pbar = tqdm.tqdm(range(1, NUM_ITERS + 1))
    
    for i in pbar:
        # 1. 重置环境与状态
        state_vec = env.reset()
        # state_vec: (B, 18) [位置, 速度, 旋转, 动作]
        
        # 重置模型隐藏状态
        hx = None
        
        # 生成随机目标 (B, 3)
        # 目标距离稍微拉远，鼓励高速飞行
        target_pos = torch.randn((BATCH_SIZE, 3), device=device) * 4.0
        target_pos[:, 2] = torch.rand(BATCH_SIZE, device=device) * 2.0 + 1.0 
        
        # 碰撞障碍点云
        obstacle_pcd = env.renderer.mesh.verts_packed().unsqueeze(0) # (1, N, 3)
        
        loss_total_acc = 0.0
        
        # 记录统计数据
        dist_history = []
        speed_history = []
        
        # 动作平滑记录
        current_action_detached = torch.zeros((BATCH_SIZE, 3), device=device)

        for t in range(TRAIN_HORIZON):
            # 2. 感知（渲染深度图）
            _, depth = env.render(return_rgb=False, return_depth=True)
            
            # --- [Critical] 深度图预处理 ---
            # 参考项目使用倒数深度: 3.0 / depth - 0.6
            # 深度范围通常 [0.3, infinity]
            depth_safe = torch.clamp(depth, 0.1, 20.0)
            inverse_depth = 3.0 / depth_safe - 0.6
            # (B, H, W) -> (B, 1, H, W)
            depth_input = inverse_depth.unsqueeze(1)
            
            # [Fix] 下采样以匹配模型输入 (48x64 -> 12x16)，这也是参考项目的做法
            depth_input = F.max_pool2d(depth_input, kernel_size=4, stride=4)
            
            # 3. 构建观测向量 (转为 Body Frame 以提高泛化性)
            # 当前状态
            pos = env.p
            vel = env.v            # World Frame
            rot_mat = env.R        # Body to World [B, 3, 3]
            act_current = env.act  # Current Actuation (net acc)
            
            # A. 目标速度向量 (World -> Body)
            rel_pos = target_pos - pos
            dist_to_target = torch.norm(rel_pos, dim=1, keepdim=True)
            target_dir = rel_pos / (dist_to_target + 1e-6)
            
            # 设定期望速度大小 (P控制器)
            speed_limit = 4.0
            desired_speed = torch.clamp(dist_to_target, max=speed_limit)
            target_vel_world = target_dir * desired_speed
            
            # World Vector @ R = Body Vector (Project onto columns of R)
            # (B, 1, 3) @ (B, 3, 3) -> (B, 1, 3)
            # 注意：如果R是Body->World，那么Body = World * R (行向量乘法? No)
            # R 的列是 Body 的基向量. V_world = x*b_x + y*b_y + z*b_z
            # V_world . b_x = x. (Dot product projects onto basis because basis is orthonormal)
            # 所以 V_body = V_world @ R (Tensor积在最后一维) 是正确的投影
            target_vel_body = (target_vel_world.unsqueeze(1) @ rot_mat).squeeze(1)
            
            # B. 当前速度向量 (World -> Body)
            local_vel_body = (vel.unsqueeze(1) @ rot_mat).squeeze(1)
            
            # C. 机体 Z 轴 (World Frame, 描述姿态)
            # R 的第三列是 Body Z 在 World 中的表示
            body_z_world = rot_mat[:, :, 2] 
            
            obs_vec = torch.cat([target_vel_body, local_vel_body, body_z_world], dim=1) # (B, 9)
            
            # 4. 模型推断
            action, _, hx = model(depth_input, obs_vec, hx)
            
            # 保存上一步动作用于平滑 Loss
            if t > 0:
                prev_action = current_action_detached
            else:
                prev_action = action
            current_action_detached = action.detach() # 用于下一帧 Loss
            
            # 5. 环境步进
            # action 是期望推力加速度
            # target_pos_vector=rel_pos 用于辅助偏航控制 (看向目标)
            next_state_vec = env.step(action_cmd=action, target_pos_vector=rel_pos)
            
            # 6. 计算损失 (Losses)
            
            # A. 速度跟踪 (主要目标)
            # 比较 Body Frame 下的速度更好，或者 World Frame 也可以
            loss_vel = F.smooth_l1_loss(vel, target_vel_world)
            
            # B. 碰撞规避 (Barrier Loss)
            # 计算到障碍物的距离
            # 使用简单的插值检测
            dists = calc_interpolated_distances(pos, env.p, obstacle_pcd, steps=5)
            # dists: (B, Steps)
            loss_avoid = barrier(dists, visible_dist=3.0)
            
            # C. 硬碰撞惩罚
            safety_margin = 0.3
            loss_col = F.relu(safety_margin - dists.min(dim=1)[0]).mean()
            
            # D. 正则化
            loss_reg = torch.mean(action**2) # 最小化能量/输入
            loss_smooth = torch.mean((action - prev_action)**2) # 动作平滑
            
            # 总损失
            loss_step = COEF_VEL * loss_vel + \
                        COEF_OBJ_AVOID * loss_avoid + \
                        COEF_COLLISION * loss_col + \
                        COEF_REG * loss_reg + \
                        COEF_SMOOTH * loss_smooth
            
            loss_total_acc += loss_step
            
            # 记录
            dist_history.append(dist_to_target.mean().item())
            speed_history.append(torch.norm(vel, dim=1).mean().item())

        # 7. 优化
        optimizer.zero_grad()
        final_loss = loss_total_acc / TRAIN_HORIZON
        
        # 检查 NaN
        if torch.isnan(final_loss):
            print(f"Loss NaN at iter {i}, resetting...")
            continue
            
        final_loss.backward()
        
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 8. 日志记录
        if i % 10 == 0:
            avg_dist = sum(dist_history)/len(dist_history)
            avg_speed = sum(speed_history)/len(speed_history)
            
            pbar.set_description(f"Iter {i} | Loss: {final_loss.item():.4f} | Dist: {avg_dist:.2f} | Spd: {avg_speed:.2f}")
            writer.add_scalar("Loss/Total", final_loss.item(), i)
            writer.add_scalar("Metric/AvgDistToTarget", avg_dist, i)
            writer.add_scalar("Metric/AvgSpeed", avg_speed, i)
            
        if i % 1000 == 0:
            # 保存检查点
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(model.state_dict(), f"checkpoints/model_{i}.pth")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    train()
