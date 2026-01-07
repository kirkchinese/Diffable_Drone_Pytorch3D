import torch
import math

class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, param):
        ctx.param = param
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.param, None

def g_decay(x, param):
    return GDecay.apply(x, param)

def solve_attitude_from_thrust_and_goal_vec(
    thrust_vector: torch.Tensor, 
    velocity: torch.Tensor, 
    R_old: torch.Tensor, 
    yaw_inertia: float = 5.0,
    dt: float = 0.02,
    yaw_ctl_delay: float = 12.0 # 建议默认值 12.0 (参考的DiffPhysDrone default)
) -> torch.Tensor:
    """
    根据期望推力向量和目标向量，反解机体姿态 (旋转矩阵)。
    支持单个输入或批量输入。
    
    Warning:
        参考项目(DiffPhysDrone)中的 `a_thr` 是包含重力抵消的合加速度 (act, Net Acceleration)。
        但在计算 Z 轴时，参考项目代码执行了 `az = a_thr[:, 2] + 9.8`，从而还原了纯推力向量。
        本函数的 `thrust_vector` 参数期望的是 **纯推力向量 (Thrust Vector)**，即已经包含了重力补偿。
        如果你使用控制器的输出 `act` (Net Acc)，请务必传入 `act + g` (e.g., act + [0,0,9.8])。
    
    Args:
        thrust_vector (torch.Tensor): 期望推力向量 (世界坐标系)，对应机体 Z 轴方向。
                                      注意：即 Thrust Force / Mass。应包含抵消重力的分量。
                                      Shape: (3,) or (B, 3)
        velocity (torch.Tensor): 指向目标的向量 (世界坐标系)，用于确定机头朝向。Shape: (3,) or (B, 3)
        R_old (torch.Tensor): 上一时刻的旋转矩阵 (Body to World)。Shape: (3, 3) or (B, 3, 3)
        yaw_inertia (float): 偏航惯性系数 (混合权重)。
        dt (float): 时间步长，用于计算平滑系数。
        yaw_ctl_delay (float): 偏航响应系数 (Yaw Response Rate)。注意：不要被命名误导。
                               公式为 alpha = exp(-delay * dt)。
                               值越大 -> alpha越小 -> 对新输入的响应越快 (延迟越小)。
                               建议默认值 12.0。
    """
    # 维度处理 (统一转为 Batch 模式)
    is_batch = True
    if thrust_vector.dim() == 1:
        is_batch = False
        thrust_vector = thrust_vector.unsqueeze(0) # [1, 3]
        velocity = velocity.unsqueeze(0)           # [1, 3]
        R_old = R_old.unsqueeze(0)                 # [1, 3, 3]
        
    # 确定机体 Z 轴 (Body Up)
    # Z 轴方向即为推力方向
    # normalize dim=1 (across x,y,z)
    z_new = torch.nn.functional.normalize(thrust_vector, dim=1)
    
    # 确定机体 X 轴 (Body Forward)
    # 逻辑：混合“旧的机头朝向”与“当前速度方向”，并强制与 Z 轴垂直
    
    # 获取旧的 X 轴 (R 的第一列)
    # R_old: [B, 3, 3] -> R_old[:, :, 0]: [B, 3]
    x_old = R_old[:, :, 0]
    
    # [一级平滑/混合] 混合向量 (基于 yaw_inertia)
    # 参考项目 Update: forward_vec = forward_vec * yaw_inertia + v_pred
    # 物理含义: 风向标效应。速度越大，对机头朝向的修正力越大。
    # 不进行归一化，自然处理低速/悬停情况
    f_mixed_raw = x_old * yaw_inertia + velocity
    
    # 计算目标朝向 (Heading)
    target_heading = torch.nn.functional.normalize(f_mixed_raw, dim=1)
    
    # [二级平滑] Exponential Smoothing (参考项目逻辑)
    # alpha = exp(-delay * dt). delay 越大 (响应越快), alpha 越小, 结果越接近 target.
    # f_temp = target * (1 - alpha) + old * alpha
    alpha = math.exp(-yaw_ctl_delay * dt)
    f_temp = target_heading * (1 - alpha) + x_old * alpha
    
    # 正交化 (Orthogonalization)
    # 约束: x_new . z_new = 0  =>  fx*ux + fy*uy + fz*uz = 0
    # 解法: 保留 f_temp 的 x, y 分量，计算 fz
    ux, uy, uz = z_new[:, 0], z_new[:, 1], z_new[:, 2]
    fx, fy = f_temp[:, 0], f_temp[:, 1]
    
    # 准备结果容器
    fz_new = torch.zeros_like(fx)
    
    # 避免除以零 (当机体 Z 轴完全水平时，uz=0)
    mask_stable = torch.abs(uz) > 1e-6
    
    # 正常情况计算
    if mask_stable.any():
        fz_new[mask_stable] = -(fx[mask_stable] * ux[mask_stable] + fy[mask_stable] * uy[mask_stable]) / uz[mask_stable]
        
    x_new_unnormalized = torch.stack([fx, fy, fz_new], dim=1)
    
    # 备选方案：如果 Z 轴水平 (uz 接近 0)，使用旧的 Y 轴与 Z 轴叉乘得到 X 轴
    if not mask_stable.all():
        y_old = R_old[:, :, 1]
        # cross product
        x_fallback = torch.cross(y_old, z_new, dim=1)
        x_new_unnormalized[~mask_stable] = x_fallback[~mask_stable]
        
    # 归一化得到新的 X 轴
    x_new = torch.nn.functional.normalize(x_new_unnormalized, dim=1)
    
    # 确定机体 Y 轴 (Body Left)
    # 右手定则: y = z cross x
    y_new = torch.cross(z_new, x_new, dim=1)
    
    # 组合旋转矩阵
    # 列向量分别为 X, Y, Z -> stack dim=2
    R_new = torch.stack([x_new, y_new, z_new], dim=2)
    
    # 如果输入不是 batch，还原维度
    if not is_batch:
        return R_new.squeeze(0)
        
    return R_new

def simulate_position_step(
    p: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    act: torch.Tensor,
    act_pred: torch.Tensor,
    R: torch.Tensor,
    dt: float,
    # --- 环境与模型参数 ---
    dg: torch.Tensor = None,
    v_wind: torch.Tensor = None,
    drag_coef_lin: float = 0.375,
    drag_coef_quad: float = 0.0,
    z_drag_coef: float = 1.0,
    pitch_ctl_delay: float = 12.0,
    rotor_drag_coef: float = 0.07,
    airmode_coef: float = 0.5,
    enable_airmode: bool = True,
    enable_induced_drag: bool = False,
    grad_decay: float = 1.0
):
    """
    模拟单步位置和速度更新 (基于 DiffPhysDrone 动力学模型)。
    包含：执行器延迟、气动阻力(线性+二次)、螺旋桨诱导阻力(H-force)、Airmode效应。
    
    [输入参数辨析]:
    - act (Current Action State): 
        上一时刻结束时，电机/螺旋桨实际达到的物理状态（推力加速度）。
        这是一个状态变量，需要你在外部循环中维护（即把本函数返回的 act_next 传给下一时刻的 act）。
        它模拟了电机的物理惯性，不能瞬间改变。
    - act_pred (Target Action Input):
        当前时刻控制器（神经网络）输出的期望目标值。
        
    [坐标系]:
    - 位置、速度、加速度均为 ROS World Frame (ENU)。
    - R 为 Body -> World 的旋转矩阵。
    
    Args:
        p (torch.Tensor):        当前位置 [B, 3]。Unit: m
        v (torch.Tensor):        当前速度 [B, 3]。Unit: m/s
        a (torch.Tensor):        上一时刻的物理加速度 [B, 3] (用于Verlet积分)。Unit: m/s^2
        act (torch.Tensor):      [状态变量] 上一时刻的实际“净”推力加速度。
        act_pred (torch.Tensor): [控制输入] 当前时刻期望的“净”推力加速度。
                                 [0,0,0]代表完全悬停(意味着实际推力=9.8N/kg)；
                                 [0,0,-9.8]代表自由落体(意味着实际推力=0)。
                                 Shape: [B, 3]。Unit: m/s^2
        R (torch.Tensor):        当前机体姿态旋转矩阵 (Body -> World)。
                                 Shape: [B, 3, 3]。
        dt (float):              仿真时间步长。Unit: s
        
        # --- 环境与模型参数 (通常保持固定或作为 Learnable Params) ---
        dg (torch.Tensor, optional):      外部扰动加速度 (如阵风)。Shape: [B, 3]。Default: 0
        v_wind (torch.Tensor, optional):  环境风速 (World Frame)。Shape: [B, 3]。Default: 0
        drag_coef_lin (float):            线性阻力系数 (kv1)。F_drag ~ v
        drag_coef_quad (float):           二次阻力系数 (kv2)。F_drag ~ v^2
        z_drag_coef (float):              Z轴阻力各向异性倍率。通常旋翼平面垂直方向阻力更大/更小。
        pitch_ctl_delay (float):          姿态/推力响应系数 (Response Rate)。
                                          公式: alpha = exp(-delay * dt)。
                                          值越大 -> alpha越小 -> (1-alpha)越大 -> 趋向目标越快。
                                          注意：虽然叫 "delay"，但数值越大响应越快 (Bandwidth越高)。
        rotor_drag_coef (float):          螺旋桨诱导阻力系数 (H-force)。
        airmode_coef (float):             Airmode 角速度诱导加速度系数。
        enable_airmode (bool):            是否开启 Airmode 模拟。
        enable_induced_drag (bool):       是否开启诱导阻力模拟。
        grad_decay (float):               梯度衰减系数 (训练用技巧)。
        
    Returns:
        tuple: (p_next, v_next, a_next, act_next)
            - p_next: 下一时刻位置 [B, 3]
            - v_next: 下一时刻速度 [B, 3]
            - a_next: 当前时刻计算出的总物理加速度 (含重力、阻力等) [B, 3]
            - act_next: 更新后的执行器推力加速度 (经过低通滤波) [B, 3]
    """
    # 1. 执行器动力学 (低通滤波模拟延迟)
    # alpha = exp(-delay * dt)
    # 兼容 Tensor 输入 (如果 pitch_ctl_delay 是 Tensor)
    if isinstance(pitch_ctl_delay, torch.Tensor):
        alpha = torch.exp(-pitch_ctl_delay * dt)
    else:
        alpha = math.exp(-pitch_ctl_delay * dt)
        
    act_next = act_pred * (1 - alpha) + act * alpha
    
    # 2. 阻力计算 (Drag)
    # 相对速度
    if v_wind is None:
        v_rel = v
    else:
        v_rel = v - v_wind
        
    # 将相对速度投影到机体坐标系 (Body Frame)
    # v_body = v_rel @ R  (注意 R 是 Body->World, 所以 v_rel @ R 相当于 v_rel . col_i)
    # R 的列向量分别是 Body 的 X, Y, Z 轴在 World 下的表示
    # v_fwd (X) = v_rel . X_axis
    # v_left (Y) = v_rel . Y_axis
    # v_up (Z) = v_rel . Z_axis
    
    # 批量矩阵乘法: v_rel [B, 3], R [B, 3, 3]
    # v_rel.unsqueeze(1) [B, 1, 3] @ R [B, 3, 3] -> [B, 1, 3]
    v_body = torch.bmm(v_rel.unsqueeze(1), R).squeeze(1)
    
    v_fwd = v_body[:, 0]
    v_left = v_body[:, 1]
    v_up = v_body[:, 2]
    
    # 计算带符号的平方项 (用于二次阻力)
    v_fwd_2 = v_fwd * torch.abs(v_fwd)
    v_left_2 = v_left * torch.abs(v_left)
    v_up_2 = v_up * torch.abs(v_up)
    
    # 计算阻力加速度 (在机体坐标系下计算分量，然后合成到世界坐标系)
    # 线性阻力分量
    a_drag_lin_body = torch.stack([
        v_fwd, 
        v_left, 
        v_up * z_drag_coef
    ], dim=1)
    
    # 二次阻力分量
    a_drag_quad_body = torch.stack([
        v_fwd_2, 
        v_left_2, 
        v_up_2 * z_drag_coef
    ], dim=1)
        
    # 转换回世界坐标系
    # a_drag_world = R @ a_drag_body.T -> [B, 3, 3] @ [B, 3, 1]
    a_drag_lin_world = torch.bmm(R, a_drag_lin_body.unsqueeze(2)).squeeze(2)
    a_drag_quad_world = torch.bmm(R, a_drag_quad_body.unsqueeze(2)).squeeze(2)
    
    # 常规气动阻力
    a_drag = a_drag_lin_world * drag_coef_lin + a_drag_quad_world * drag_coef_quad
    
    # [New] 3. 螺旋桨诱导阻力 (Propeller Induced Drag / H-force)
    # 仅当 enable_induced_drag=True 时计算
    a_drag_induced = torch.zeros_like(act_next)
    
    if enable_induced_drag:
        # 计算推力向量 (去除重力, 假设 Z 轴向下为负, g_std=[0,0,-9.8])
        # act_next = Thrust + Disturbance + Gravity. 
        # 这里近似认为 act_next 代表了当前的合力状态，从中提取 Thrust magnitude
        # 假设标准重力
        g_vec = torch.tensor([0.0, 0.0, -9.80665], device=act.device)
        thrust_acc_vec = act_next - g_vec # [B, 3]
        thrust_mag = torch.norm(thrust_acc_vec, dim=1) # [B]
        
        # 电机速度估算 (sqrt of thrust)
        motor_vel = torch.sqrt(torch.clamp(thrust_mag, min=1e-6))
        
        # 计算垂直于推力方向的速度 (v_prep)
        # v_perp_body = [v_fwd, v_left, 0]
        # 诱导阻力主要作用在水平面 (机体 XY 平面)
        a_drag_induced_body = torch.stack([
            v_fwd * motor_vel * z_drag_coef, # Apply z_drag_coef as scaling
            v_left * motor_vel * z_drag_coef,
            torch.zeros_like(v_up)
        ], dim=1)
        
        # 变换回世界坐标系
        a_drag_induced_world = torch.bmm(R, a_drag_induced_body.unsqueeze(2)).squeeze(2)
        
        # 应用诱导阻力系数
        a_drag_induced = a_drag_induced_world * rotor_drag_coef
    
    # [New] 4. Airmode (Angular Velocity Induced Acceleration)
    # 仅当 enable_airmode=True 时计算
    a_airmode = torch.zeros_like(act_next)
    
    if enable_airmode:
        # 计算 act 与 act_next 之间的夹角 (用于估算角速度 av)
        # 参考项目逻辑：scalar_t dot = act... * act_next...;
        # scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;
        
        # 补全重力，计算推力向量的变化 (Thrust Vector Change) 
        # 参考项目逻辑：dot product 和 norm 都是基于 (act + g) 计算的
        g_correction = torch.tensor([0.0, 0.0, 9.80665], device=act.device)
        thrust_vec_curr = act + g_correction
        thrust_vec_next = act_next + g_correction

        # 范数计算 (避免除零)
        n1 = torch.norm(thrust_vec_curr, dim=1, keepdim=True)
        n2 = torch.norm(thrust_vec_next, dim=1, keepdim=True)
        n1 = torch.clamp(n1, min=1e-8)
        n2 = torch.clamp(n2, min=1e-8)
        
        # 点积
        dot = torch.sum(thrust_vec_curr * thrust_vec_next, dim=1, keepdim=True)
        
        # 夹角
        cos_theta = torch.clamp(dot / (n1 * n2), min=-1.0, max=1.0)
        angle = torch.acos(cos_theta) # [B, 1]
        
        # 角速度近似
        av = angle / dt 
        
        # 计算额外加速度
        # 参考项目: airmode_a axes = ax/thrust * av * airmode_av2a
        # 方向：沿纯推力方向 (Thrust)，需从 act (含重力) 中还原
        # src/dynamics_kernel.cu 中: az = act[i][2] + 9.80665;
        g_correction = torch.tensor([0.0, 0.0, 9.80665], device=act.device)
        thrust_vec = act + g_correction
        thrust_norm = torch.norm(thrust_vec, dim=1, keepdim=True).clamp(min=1e-6)
        
        dir_vec = thrust_vec / thrust_norm
        
        a_airmode = dir_vec * av * airmode_coef

    # 5. 合加速度计算
    # a_next = Thrust_acc + Gravity + Disturbance - Drag - Induced_Drag + Airmode
    if dg is None:
        dg = torch.zeros_like(act_next)
        
    a_next = act_next + dg - a_drag - a_drag_induced + a_airmode
    
    # [New] 计算梯度衰减参数 (decay per step)
    # 必须 **dt 以保证时间一致性：即无论 dt 大小如何，单位时间内的总衰减量恒定。
    # 对应参考项目 CUDA 源码 dynamics_kernel.cu 中的 `pow(grad_decay, ctl_dt)`
    decay_factor = grad_decay ** dt

    # 使用 g_decay 包裹上一时刻的状态变量 p 和 v
    # 这样在反向传播时，流经 p 和 v 的梯度会被衰减，防止长序列梯度爆炸
    p_decayed = g_decay(p, decay_factor)
    v_decayed = g_decay(v, decay_factor)

    # p_next = p + v * dt + 0.5 * a * dt^2
    p_next = p_decayed + v * dt + 0.5 * a * dt**2
    
    # v_next = v + 0.5 * (a + a_next) * dt
    v_next = v_decayed + 0.5 * (a + a_next) * dt
    
    return p_next, v_next, a_next, act_next

def update_dg(dg_curr: torch.Tensor, dt: float, noise_std: float = 0.2) -> torch.Tensor:
    """
    更新额外加速度 (Disturbance Gravity)。
    使用一阶马尔可夫过程 (Ornstein-Uhlenbeck Process 近似) 模拟随时间变化的扰动。
    
    Args:
        dg_curr (torch.Tensor): 当前的扰动向量 [B, 3]
        dt (float): 时间步长
        noise_std (float): 噪声标准差，控制扰动的幅度
        
    Returns:
        dg_new (torch.Tensor): 更新后的扰动向量
    """
    # 衰减系数，确保扰动不会无限增长，且保持时间相关性
    # 参考项目公式: dg = dg * sqrt(1 - dt/4) + noise * std * sqrt(dt/4)
    # 这是一种保持方差恒定的随机游走更新
    
    decay_factor = math.sqrt(1 - dt / 4.0)
    noise_factor = noise_std * math.sqrt(dt / 4.0)
    
    # 生成随机噪声
    noise = torch.randn_like(dg_curr)
    
    dg_new = dg_curr * decay_factor + noise * noise_factor
    return dg_new

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ==========================================
    # 一个批量处理示范过程


    # 1. 初始化
    B = 10
    steps = 100
    dt = 0.02

    # 状态变量
    p = torch.zeros((B, 3), device=device)       # 位置
    v = torch.zeros((B, 3), device=device)       # 速度
    a = torch.zeros((B, 3), device=device)       # 加速度
    R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1) # 姿态
    act = torch.zeros((B, 3), device=device)     # 实际推力(加速度形式)
    dg = torch.randn((B, 3), device=device) * 0.2 # 初始化扰动

    # 期望控制量 (假设我们要悬停，只有重力补偿)

    act_pred = torch.zeros((B, 3), device=device) # 期望合加速度为0 (悬停)

    # 记录轨迹
    traj_p = []
    traj_dg = []

    for i in range(steps):
        # 模拟一步动力学
        # 可以在这里控制 enable_airmode 和 enable_induced_drag
        p, v, a, act = simulate_position_step(
            p, v, a, act, act_pred, R, dt, 
            dg=dg, # 传入当前的 dg
            enable_airmode=True,       # 开启 Airmode
            enable_induced_drag=False,  # 关闭 H-force (默认配置)
            grad_decay=0.99            # 梯度衰减系数，防止长序列梯度爆炸
        )

        # 更新扰动
        dg = update_dg(dg, dt)
        
        # 更新姿态 (这里简单假设姿态完美跟踪或者是根据某种策略更新)
        # 在这个简单的测试中，我们保持姿态不变(水平)，或者你可以加入姿态控制逻辑
        # 比如我们希望它稍微抵抗一点风，但这里我们先通过 update_dg 观察位置漂移
        
        traj_p.append(p.clone())
        traj_dg.append(dg.clone())

    # 转换为 Tensor 方便绘图
    traj_p = torch.stack(traj_p).squeeze(1).cpu()
    traj_dg = torch.stack(traj_dg).squeeze(1).cpu()

    print(f"仿真结束。最后位置: {p}")
    print(f"位置漂移 (由于 dg): {p.norm()}")

    # 简单的绘图 (如果有 matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(traj_p[:, 0], label='X')
        plt.plot(traj_p[:, 1], label='Y')
        plt.plot(traj_p[:, 2], label='Z')
        plt.title("Position Drift (Hovering with Disturbance)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(traj_dg[:, 0], label='dg_X')
        plt.plot(traj_dg[:, 1], label='dg_Y')
        plt.plot(traj_dg[:, 2], label='dg_Z')
        plt.title("Disturbance Acceleration (dg) over Time")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not found, skipping plot.")