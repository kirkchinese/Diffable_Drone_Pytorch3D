import torch.nn.functional as F
import torch
from pytorch3d.ops import knn_points



def calc_min_distance(drone_pos, obstacle_pcd):
    """
    计算无人机中心点与障碍物点云之间的最短距离 (Single Step)
    """
    p1 = drone_pos.unsqueeze(1) 
    B = p1.shape[0]
    
    if obstacle_pcd.shape[0] != B:
        obstacle_pcd_expanded = obstacle_pcd.expand(B, -1, -1)
    else:
        obstacle_pcd_expanded = obstacle_pcd
        
    result = knn_points(p1, obstacle_pcd_expanded, K=1)
    sq_dists = result.dists.squeeze(-1) # (B, 1) -> (B,)
    dists = torch.sqrt(sq_dists + 1e-6).squeeze(-1) 
    return dists

def calc_interpolated_distances(p_start, p_end, obstacle_pcd, steps=10):
    """
    计算从 p_start 到 p_end 路径上的插值点到障碍物的距离 (Sub-stepping)
    解决隧穿效应 (Tunneling Effect) 以及提供更平滑的碰撞梯度。
    
    Args:
        p_start, p_end: (B, 3) 起点和终点
        obstacle_pcd: (1, N, 3) or (B, N, 3) 障碍物点云
        steps: 插值步数 (参考项目使用了 10)
        
    Returns:
        dists: (B, steps) 每个插值点的欧氏距离
    """
    B = p_start.shape[0]
    device = p_start.device
    
    # 1. 线性插值生成路径点
    # alphas: (1, steps, 1) 范围 [0, 1]
    alphas = torch.linspace(0, 1, steps, device=device).view(1, steps, 1)
    
    # 插值公式: p(alpha) = p1 + (p2 - p1) * alpha
    # (B, 1, 3) + (B, 1, 3) * (1, steps, 1) -> (B, steps, 3)
    traj_points = p_start.unsqueeze(1) + (p_end - p_start).unsqueeze(1) * alphas
    
    # 2. 准备 KNN 输入
    # obstacle_pcd 扩展: (B, N, 3)
    if obstacle_pcd.shape[0] != B:
        obstacle_input = obstacle_pcd.expand(B, -1, -1)
    else:
        obstacle_input = obstacle_pcd
        
    # 3. 计算距离 
    # knn_points 支持输入: P1=(B, Steps, 3), P2=(B, N_obs, 3)
    # output.dists: (B, Steps, K) - 平方距离
    knn_res = knn_points(traj_points, obstacle_input, K=1)
    
    # 开根号得到欧氏距离
    dists = torch.sqrt(knn_res.dists.squeeze(-1) + 1e-6) # (B, steps)
    return dists


class DroneLoss:
    """
    损失函数计算类，参考 DiffPhysDrone 项目实现。
    
    Code Review Notes:
    1. Collision Loss: 采用了线性插值 (Linear Interpolation) 替代参考项目的速度外推 (Velocity Extrapolation)，
       这在已知 next_state 的情况下更准确。
    2. Snap Loss: 参考项目计算的是推力方向 (Angular) 的 Snap，本实现计算的是线加速度 (Linear) 的 Snap。
       鉴于参考项目中该项权重默认为 0.0，此差异可忽略。
    """
    def __init__(self, device, margin=0.2, dt=0.02, coef_collide=2.0, coef_obj_avoidance=1.5,coef_v=1.0,
                 coef_d_acc=0.01, coef_d_jerk=0.001, coef_d_snap=0.0,V_TO_PT_SCALE=135.0,COLLISION_SCALE=-32.0):
        self.device = torch.device(device)
        self.margin = margin # 安全边际 (radius + safety buffer) (参考项目 env.margin)
        self.dt = dt
        
        # 权重系数 (DiffPhysDrone/main_cuda.py args)
        self.coef_collide = coef_collide
        self.coef_obj_avoidance = coef_obj_avoidance
        self.coef_v = coef_v
        self.coef_d_acc = coef_d_acc
        self.coef_d_jerk = coef_d_jerk
        self.coef_d_snap = coef_d_snap # Reference default is 0.0
        

        self.V_TO_PT_SCALE = V_TO_PT_SCALE
        self.COLLISION_SCALE = COLLISION_SCALE
        
    def barrier(self, x, v_to_pt):
        """
        障碍物避障 Barrier Loss
        x: 实际距离 - margin
        v_to_pt: 接近速度权重 (由 diff(distance) 计算)
        Formula: mean( v * ReLU(1 - x)^2 )
        Reference: main_cuda.py barrier()
        """
        # 注意: x 是已经减去 margin 后的距离，如果 x < 0 说明小于 margin (潜在碰撞)
        # reference: (1-x).relu() 当 x 很小时 (比如 -0.1)，value > 1，惩罚大
        # 当 x (distance) 很大时，(1-x) 为负，relu 为 0，无损失。
        # 这里的 1.0 实际上也是一个隐含的 threshold 距离 (margin + 1.0)
        return (v_to_pt * (1 - x).relu().pow(2)).mean()

    def get_collision_loss(self, p_start, p_end, obstacle_pcd, sub_steps=10):
        """
        计算碰撞和避障损失
        Args:
            p_start, p_end: (B, 3) 无人机当前步和下一步的位置
            obstacle_pcd: (1, N, 3) 障碍物点云
        """
        # 计算插值路径上的距离 (Calls global calc_interpolated_distances)
        # Review: 参考项目使用 p + v * sub_div (基于速度的外推)。
        # 这里使用 p_start 到 p_end 的线性插值。如果 p_end 是物理更新后的位置，这种“弦”插值也是合理的。
        dists = calc_interpolated_distances(p_start, p_end, obstacle_pcd, steps=sub_steps)
        
        # 减去安全边际
        eff_dists = dists - self.margin
        
        # 计算“接近速度”
        diff = torch.diff(eff_dists, dim=1)
        v_to_pt = (-diff * self.V_TO_PT_SCALE).clamp_min(1.0)
        
        # 对应 diff 后的距离数组 (减少了一个长度)
        curr_dist = eff_dists[:, 1:]
        
        # Collide Loss: 使用 Softplus 实现平滑的指数惩罚
        loss_collide = F.softplus(curr_dist * self.COLLISION_SCALE).mul(v_to_pt).mean()

        # Avoidance Loss: 使用 Barrier 函数推离障碍物
        loss_obj = self.barrier(curr_dist, v_to_pt)
        
        total = self.coef_collide * loss_collide + self.coef_obj_avoidance * loss_obj
        return total, loss_collide, loss_obj

    def get_smoothness_loss(self, acc_seq):
        """
        计算动作平滑度损失 (Acc, Jerk, Snap)
        Args:
            acc_seq: (B, T, 3) 加速度序列 (或 Force/Thrust)
        """
        # 确保输入有时间维度
        if acc_seq.dim() == 2:
             acc_seq = acc_seq.unsqueeze(1) # (B, 1, 3)
             
        T = acc_seq.shape[1]
        
        loss_acc = acc_seq.pow(2).sum(-1).mean()
        
        loss_jerk = torch.tensor(0.0, device=self.device)
        loss_snap = torch.tensor(0.0, device=self.device)
        
        if T >= 2:
            scale_fps = 1.0 / self.dt
            # Jerk: accelerations derivative
            jerk = torch.diff(acc_seq, dim=1) * scale_fps
            loss_jerk = jerk.pow(2).sum(-1).mean()
            
            if T >= 3:
                # Snap: jerk derivative
                # NOTE: Reference implementation calculates Snap on "Normalized Thrust Direction"
                # (removing gravity), penalizing angular angular acceleration.
                # Current implementation penalizes linear snap.
                snap = torch.diff(jerk, dim=1) * scale_fps
                loss_snap = snap.pow(2).sum(-1).mean()
        
        total = self.coef_d_acc * loss_acc + \
                self.coef_d_jerk * loss_jerk + \
                self.coef_d_snap * loss_snap
                
        return total, loss_acc, loss_jerk, loss_snap

    def get_velocity_loss(self, current_v, target_v, use_sliding_window=True, window_size=30):
        """
        计算速度追踪损失。
        Args:
            current_v: (B, T, 3) 历史速度
            target_v: (B, T, 3) 历史期望速度
        """
        if current_v.dim() == 2:
             current_v = current_v.unsqueeze(1)
             target_v = target_v.unsqueeze(1)
             
        T = current_v.shape[1]

        # Review: 严格匹配 main_cuda.py 中的 sliding window 逻辑
        if use_sliding_window and T > window_size:
            v_cum = current_v.cumsum(dim=1)
            # 计算滑动窗口平均速度
            v_avg = (v_cum[:, window_size:] - v_cum[:, :-window_size]) / window_size
            
            # 对齐 Target: reference slices target [1 : -29] (length T-30)
            # v_avg length is T-30.
            # 这里的切片逻辑需确保长度对齐:
            tgt_slice = target_v[:, 1:-window_size+1] 
            
            min_len = min(v_avg.shape[1], tgt_slice.shape[1])
            v_avg = v_avg[:, :min_len]
            tgt_slice = tgt_slice[:, :min_len]
            
            delta_v = torch.norm(v_avg - tgt_slice, p=2, dim=-1)
            loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
        else:
            loss_v = F.smooth_l1_loss(current_v, target_v)
            
        return self.coef_v * loss_v