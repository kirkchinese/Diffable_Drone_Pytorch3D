import torch
import os
import sys
from pytorch3d.ops import knn_points

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
                 principal_point=None,           # New
                 lights_location=[[0.0, 0.0, -3.0]], # New
                 num_samples=20000,
                 # 动力学参数
                 enable_airmode=True,
                 enable_induced_drag=False,
                 noise_std=0.04,
                 grad_decay=0.8,
                 yaw_inertia=5.0,
                 yaw_ctl_delay=12.0,            # 更新默认值
                 pitch_ctl_delay=12.0,
                 drag_coef_lin=0.375,
                 drag_coef_quad=0.0,
                 z_drag_coef=1.0,
                 rotor_drag_coef=0.07,
                 airmode_coef=0.5,
                 gravity=9.80665,
                 # 初始化范围参数
                 init_p_range=2.0,
                 init_v_range=0.0,
                 init_dg_range=0.2,
                 init_margin_range=(0.1, 0.3),
                 # 运行参数
                 wind_std=0.1,
                 act_queue_len=2,
                 # 相机参数
                 cam_offset_body=[0.1, 0.0, 0.0]
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
        self.gravity = gravity
        self.gravity_vec = torch.tensor([0, 0, -gravity], device=self.device).unsqueeze(0).repeat(self.B, 1)

        # 初始化/运行配置
        self.init_p_range = init_p_range
        self.init_v_range = init_v_range
        self.init_dg_range = init_dg_range
        self.init_margin_range = init_margin_range
        self.wind_std = wind_std
        self.act_queue_len = act_queue_len
        self.cam_offset_body = cam_offset_body
        self.num_samples = num_samples
        
        # 渲染器初始化
        if not os.path.exists(mesh_path) and not mesh_path.startswith("/"):
             potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mesh_path)
             if os.path.exists(potential_path):
                 mesh_path = potential_path

        print(f"Loading mesh from: {mesh_path}")
        self.renderer = DroneRenderer(
            mesh_path=mesh_path,
            device=self.device,
            image_size=image_size,
            focal_length=focal_length,
            principal_point=principal_point,
            lights_location=lights_location,
            num_samples=num_samples
        )
        
        # 内部状态初始化
        self.reset()

    def reset(self):
        """重置无人机状态"""
        # 运动学状态
        self.p = torch.rand(self.B, 3, device=self.device) * self.init_p_range  # 位置
        self.v = torch.randn(self.B, 3, device=self.device) * self.init_v_range  # 速度
        self.a = torch.zeros(self.B, 3, device=self.device)       # 加速度
        self.act_curr = torch.zeros(self.B, 3, device=self.device)     # 实际推力状态 (Internal actuation state)
        self.R = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.B, 1, 1) # 姿态
        
        # 环境扰动
        self.dg = torch.randn((self.B, 3), device=self.device) * self.init_dg_range
        
        # 模拟控制延迟的队列
        self.act_queue = [torch.zeros(self.B, 3, device=self.device) for _ in range(self.act_queue_len)]
        
        # 安全边距
        low, high = self.init_margin_range
        self.margin = torch.rand((self.B,), device=self.device) * (high - low) + low

        return self._get_state()

    def step(self, act_cmd, target_pos_vector=None, v_wind=None, dt=None):
        """
        执行一步模拟并更新状态
        
        Args:
            act_cmd (Tensor): 期望推力指令 (B, 3),
            target_pos_vector (Tensor, optional): 目标方向向量 (B, 3)，用于姿态解算中的机头朝向 (Velocity direction)。
                                                  如果在闭环控制中，这通常是 (target_pos - current_pos)。
                                                  如果为 None，默认使用当前速度 v。
            v_wind (Tensor, optional): 风速向量 (B, 3)。如果不传，则根据 wind_std 随机生成。
            dt (float, optional): 当前步的仿真时间步长。如果为 None，使用初始化时的 self.dt。
            
        Returns:
            state (Tensor): 将状态扁平化拼接的 Tensor, Shape: (B, 18)。
                            [px, py, pz, vx, vy, vz, r00, r01, r02, r10, r11, r12, r20, r21, r22, act_curr_x, act_curr_y, act_curr_z]
                            包含了 pos(3), vel(3), rot(9), act_curr(3) (实际推力状态)
        """
        current_dt = dt if dt is not None else self.dt
        p_old = self.p.clone()
        
        # 更新环境扰动
        self.dg = update_dg(dg_curr=self.dg, dt=current_dt, noise_std=self.noise_std)

        # 模拟控制延迟：进队，出队
        self.act_queue.append(act_cmd)
        current_act_cmd = self.act_queue.pop(0)
        
        # 随机风
        if v_wind is None:
            v_wind = torch.randn((self.B, 3), device=self.device) * self.wind_std
            
        # 模拟动力学，自动更新内部状态 p, v, a, act_curr
        self.p, self.v, self.a, self.act_curr = simulate_position_step(
            p=self.p,
            v=self.v,
            a=self.a,
            R=self.R,
            act_curr=self.act_curr,
            act_cmd=current_act_cmd,
            dt=current_dt,
            enable_airmode=self.enable_airmode,
            enable_induced_drag=self.enable_induced_drag,
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
        
        # 计算纯推力向量（去除重力分量，指向机体Z轴）
        thrust_net = self.act_curr - self.gravity_vec 
        
        # 确定期望的机头/速度朝向
        if target_pos_vector is None:
            # 如未指定，使用实际位移方向
            velocity_vector = self.p - p_old
        else:
            velocity_vector = target_pos_vector
            
        # 更新姿态旋转矩阵
        self.R = solve_attitude_from_thrust_and_goal_vec(
            thrust_vector=thrust_net,
            velocity=velocity_vector,
            R_old=self.R,
            yaw_inertia=self.yaw_inertia,
            dt=current_dt,
            yaw_ctl_delay=self.yaw_ctl_delay
        )
        
        return self._get_state()

    def render(self, camera_pitch=10.0, return_tensor=True, return_rgb=True, return_depth=True, dt=None):
        """
        渲染当前帧
        
        Args:
            camera_pitch (float): 相机俯仰角
            return_tensor (bool): 是否返回 Tensor
            return_rgb (bool): 是否返回 RGB
            return_depth (bool): 是否返回 Depth
            dt (float, optional): 仿真时间步长，用于特定渲染效果（如运动模糊、流计算），可选。
        """
        R_camera, T_camera = self.renderer.compute_view_matrix(
            p_ros=self.p, 
            R_ros=self.R, 
            camera_pitch_deg=camera_pitch,
            cam_offset_body=self.cam_offset_body
        )
        rgb, depth = self.renderer.render(
            R=R_camera, 
            T=T_camera, 
            return_tensor=return_tensor, 
            return_rgb=return_rgb, 
            return_depth=return_depth,
            dt=dt
        )
        return rgb, depth

    def calc_min_distance(self, p=None):
        """
        计算无人机到障碍物的最短距离。
        
        Args:
            p (Tensor, optional): 无人机位置 (B, 3)。如果为 None，使用当前状态 self.p。
            
        Returns:
            dist (B,)
        """
        pos = p if p is not None else self.p
        p1 = pos.unsqueeze(1) 
        B_size = p1.shape[0]
        
        obstacle_pcd = self.renderer.obstacle_pcd
        if obstacle_pcd.shape[0] != B_size:
            obstacle_pcd_expanded = obstacle_pcd.expand(B_size, -1, -1)
        else:
            obstacle_pcd_expanded = obstacle_pcd
            
        result = knn_points(p1, obstacle_pcd_expanded, K=1)
        sq_dists = result.dists.squeeze(-1) # (B, 1) -> (B,)
        dists = torch.sqrt(sq_dists + 1e-6).squeeze(-1) 
        return dists

    def distance_to_obj(self, p=None):
        """
        计算当前无人机位置到障碍物的最短距离 (Alias)
        """
        return self.calc_min_distance(p)

    def vec_to_obj(self, p=None):
        """
        计算无人机位置到最近障碍物点的向量。
        
        Args:
            p (Tensor, optional): 无人机位置 (B, 3)。如果为 None，使用当前状态 self.p。
            
        Returns:
            vec (B, 3)
        """
        pos = p if p is not None else self.p
        p1 = pos.unsqueeze(1) 
        B_size = p1.shape[0]
        
        obstacle_pcd = self.renderer.obstacle_pcd
        if obstacle_pcd.shape[0] != B_size:
            obstacle_pcd_expanded = obstacle_pcd.expand(B_size, -1, -1)
        else:
            obstacle_pcd_expanded = obstacle_pcd
            
        result = knn_points(p1, obstacle_pcd_expanded, K=1)
        idx = result.idx.squeeze(-1).squeeze(-1)  # (B,)
        
        # 获取最近的点
        nearest_points = obstacle_pcd_expanded[torch.arange(B_size), idx]  # (B, 3)
        
        # 向量：最近点 - 无人机位置
        vecs = nearest_points - pos
        return vecs


    def update_mesh(self, mesh_path, num_samples=None):
        """
        更换无人机的可视化网格模型
        
        Args:
            mesh_path (str): 新的 .obj 文件路径
            num_samples (int, optional): 重新采样点云的点数。如果为 None，使用 initialization 时的值。
        """
        # 路径预处理 (复用 __init__ 中的逻辑)
        if not os.path.exists(mesh_path) and not mesh_path.startswith("/"):
             potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mesh_path)
             if os.path.exists(potential_path):
                 mesh_path = potential_path

        print(f"Updating mesh to: {mesh_path}")
        samples = num_samples if num_samples is not None else self.num_samples
        self.renderer.update_mesh(mesh_path, num_samples=samples)

    @staticmethod
    def clean_depth_map(depth, min_dist=0.2, max_dist=10.0):
        """
        清洗深度图 (调用 DroneRenderer 的实现)
        """
        return DroneRenderer.clean_depth_map(depth, min_dist=min_dist, max_dist=max_dist)

    def _get_state(self):
        """
        返回一维化的状态张量
        Returns: (B, 3+3+9+3) = (B, 18)
        [pos, vel, rot_flattened, act_curr]
        """
        B = self.p.shape[0] # Batch size
        R_flat = self.R.view(B, -1) # 9 
        
        # Concatenate: pos(3) + vel(3) + rot(9) + act_curr(3)
        state = torch.cat([self.p, self.v, R_flat, self.act_curr], dim=1) # (B, 18)
        return state
