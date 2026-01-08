import torch
import os
import sys

# 尝试导入同目录下的模块
# 如果在 notebooks 中运行，确保父目录在 sys.path 中
try:
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import DroneRenderer
except ImportError:
    # 简单的 Fallback，防止直接运行此文件时找不到模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import DroneRenderer

class DroneSimulator:
    def __init__(self, 
                 batch_size=4, 
                 dt=0.02, 
                 device=None,
                 mesh_path="data/sample/sample.obj", # 默认假设从根目录运行，可在外部覆盖
                 image_size=(480, 640),
                 focal_length=500.0,
                 # 动力学参数
                 enable_airmode=True,
                 enable_induced_drag=False,     # New
                 noise_std=0.04,
                 grad_decay=0.8,
                 yaw_inertia=5.0,
                 yaw_ctl_delay=4.0,
                 pitch_ctl_delay=12.0,          # New, default in dynamics is 12.0
                 drag_coef_lin=0.375,           # New
                 drag_coef_quad=0.0,            # New
                 z_drag_coef=1.0,               # New
                 rotor_drag_coef=0.07,          # New
                 airmode_coef=0.5               # New
                 ):
        
        self.B = batch_size
        self.dt = dt
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 动力学配置
        self.enable_airmode = enable_airmode
        self.enable_induced_drag = enable_induced_drag
        self.noise_std = noise_std
        self.grad_decay = grad_decay
        self.yaw_inertia = yaw_inertia
        self.yaw_ctl_delay = yaw_ctl_delay
        self.pitch_ctl_delay = pitch_ctl_delay
        self.drag_coef_lin = drag_coef_lin
        self.drag_coef_quad = drag_coef_quad
        self.z_drag_coef = z_drag_coef
        self.rotor_drag_coef = rotor_drag_coef
        self.airmode_coef = airmode_coef
        
        # 渲染器初始化
        # 如果路径以 ".." 开头，说明是相对路径，这里保留原样交给 Renderer 处理
        # 或者转为绝对路径以增加鲁棒性
        if not os.path.exists(mesh_path) and not mesh_path.startswith("/"):
             # 尝试寻找相对于当前文件的路径（应对从不同目录调用的情况）
             potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mesh_path)
             if os.path.exists(potential_path):
                 mesh_path = potential_path

        print(f"Loading mesh from: {mesh_path}")
        self.renderer = DroneRenderer(
            mesh_path=mesh_path,
            device=self.device,
            image_size=image_size,
            focal_length=focal_length
        )
        
        # 物理/控制参数
        self.gravity_vec = torch.tensor([0, 0, -9.80665], device=self.device).unsqueeze(0).repeat(self.B, 1)
        
        # 内部状态初始化
        self.reset()

    def reset(self):
        """重置无人机状态"""
        # 运动学状态
        self.p = torch.rand(self.B, 3, device=self.device) * 2.0  # 位置
        self.v = torch.zeros(self.B, 3, device=self.device)       # 速度
        self.a = torch.zeros(self.B, 3, device=self.device)       # 加速度
        self.act = torch.zeros(self.B, 3, device=self.device)     # 实际推力状态 (Internal actuation state)
        self.R = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.B, 1, 1) # 姿态
        
        # 环境扰动
        self.dg = torch.randn((self.B, 3), device=self.device) * 0.2
        
        # 模拟控制延迟的队列
        self.act_queue = [torch.zeros(self.B, 3, device=self.device) for _ in range(2)]
        
        return self._get_state()

    def step(self, action_cmd, target_pos_vector=None, v_wind=None):
        """
        执行一步模拟并更新状态
        
        Args:
            action_cmd (Tensor): 期望推力指令 (B, 3)。
            target_pos_vector (Tensor, optional): 目标方向向量 (B, 3)，用于姿态解算中的机头朝向 (Velocity direction)。
                                                  如果在闭环控制中，这通常是 (target_pos - current_pos)。
                                                  如果为 None，默认使用当前速度 v。
            v_wind (Tensor, optional): 风速向量 (B, 3)。如果不传，则随机生成。
            
        Returns:
            state (Tensor): 将状态扁平化拼接的 Tensor, Shape: (B, 18)。
                            [px, py, pz, vx, vy, vz, r00, r01, r02, r10, r11, r12, r20, r21, r22, ax, ay, az]
                            包含了 pos(3), vel(3), rot(9), act(3) (实际推力状态)
        """
        # 0. 备份旧位置（用于姿态解算中的位移方向计算，如果外部没传 target_pos_vector）
        p_old = self.p.clone()
        
        # 1. 更新扰动
        self.dg = update_dg(dg_curr=self.dg, dt=self.dt, noise_std=self.noise_std)

        # 2. 处理控制延迟 (队列)
        self.act_queue.append(action_cmd)
        current_act_cmd = self.act_queue.pop(0)
        
        # 3. 动力学模拟 (Simulate Physics)
        # 随机风 (也可以提取为类参数)
        if v_wind is None:
            v_wind = torch.randn((self.B, 3), device=self.device) * 0.1
        
        self.p, self.v, self.a, self.act = simulate_position_step(
            p=self.p, v=self.v, a=self.a, R=self.R, act=self.act,
            act_pred=current_act_cmd,
            dt=self.dt,
            enable_airmode=self.enable_airmode,
            enable_induced_drag=self.enable_induced_drag, # New
            dg=self.dg,
            v_wind=v_wind,
            grad_decay=self.grad_decay,
            # 完整暴露参数
            pitch_ctl_delay=self.pitch_ctl_delay,
            drag_coef_lin=self.drag_coef_lin,
            drag_coef_quad=self.drag_coef_quad,
            z_drag_coef=self.z_drag_coef,
            rotor_drag_coef=self.rotor_drag_coef,
            airmode_coef=self.airmode_coef
        )
        
        # 4. 姿态控制/解算 (Attitude Solver)
        # 计算纯推力向量（去除重力分量，指向机体Z轴）
        thrust_net = self.act - self.gravity_vec 
        
        # 确定期望的机头/速度朝向
        if target_pos_vector is None:
            # 如未指定，使用实际位移方向
            velocity_vector = self.p - p_old
        else:
            velocity_vector = target_pos_vector
            
        self.R = solve_attitude_from_thrust_and_goal_vec(
            thrust_vector=thrust_net,
            velocity=velocity_vector,
            R_old=self.R,
            yaw_inertia=self.yaw_inertia,
            dt=self.dt,
            yaw_ctl_delay=self.yaw_ctl_delay
        )
        
        return self._get_state()

    def render(self, camera_pitch=10.0, return_tensor=True, return_rgb=True, return_depth=True):
        """
        渲染当前帧
        
        Args:
            camera_pitch (float): 相机俯仰角
            return_tensor (bool): 是否返回 Tensor
            return_rgb (bool): 是否返回 RGB
            return_depth (bool): 是否返回 Depth
        """
        R_camera, T_camera = self.renderer.compute_view_matrix(
            p_ros=self.p, 
            R_ros=self.R, 
            camera_pitch_deg=camera_pitch
        )
        rgb, depth = self.renderer.render(
            R=R_camera, 
            T=T_camera, 
            return_tensor=return_tensor, 
            return_rgb=return_rgb, 
            return_depth=return_depth
        )
        return rgb, depth
    def update_mesh(self, mesh_path):
        """
        更换无人机的可视化网格模型
        
        Args:
            mesh_path (str): 新的 .obj 文件路径
        """
        # 路径预处理 (复用 __init__ 中的逻辑)
        if not os.path.exists(mesh_path) and not mesh_path.startswith("/"):
             potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mesh_path)
             if os.path.exists(potential_path):
                 mesh_path = potential_path

        print(f"Updating mesh to: {mesh_path}")
        self.renderer.update_mesh(mesh_path)

    @staticmethod
    def clean_depth_map(depth, min_dist=0.2, max_dist=10.0):
        """
        清洗深度图 (Static Helper)
        """
        # 背景通常为 -1
        if torch.is_tensor(depth):
            depth = depth.clone()
            # 兼容 Float Tensor
            invalid_mask = (depth == -1) | (depth > max_dist) | (depth < min_dist)
            depth[invalid_mask] = float('nan')
        else:
            # Numpy
            depth = depth.copy()
            invalid_mask = (depth == -1) | (depth > max_dist) | (depth < min_dist)
            depth[invalid_mask] = float('nan')
        return depth

    def _get_state(self):
        """
        Helper to return current state as a flat tensor (Gym-like observation).
        Returns: (B, 3+3+9+3) = (B, 18)
        [pos, vel, rot_flattened, act]
        """
        B = self.p.shape[0]
        # Flatten R from (B, 3, 3) to (B, 9)
        R_flat = self.R.view(B, -1)
        
        # Concatenate: pos(3) + vel(3) + rot(9) + act(3)
        state = torch.cat([self.p, self.v, R_flat, self.act], dim=1)
        return state
