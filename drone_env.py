import torch
import os
import sys
from pytorch3d.ops import knn_points, sample_points_from_meshes

# 尝试导入同目录下的模块
# 如果在 notebooks 中运行，确保父目录在 sys.path 中
try:
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import DroneRenderer
    from scene_generator import (SceneGenerator, sample_safe_points, sample_safe_targets,
                                  sample_cross_map_spawn_target, obj_to_ros, ros_to_obj)
except ImportError:
    # 简单的 Fallback，防止直接运行此文件时找不到模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import DroneRenderer
    from scene_generator import (SceneGenerator, sample_safe_points, sample_safe_targets,
                                  sample_cross_map_spawn_target, obj_to_ros, ros_to_obj)

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
                 subdivide_times=0,              # 网格细分次数 (0=不细分, 降低可大幅提升批量渲染性能)
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
                 cam_offset_body=[0.1, 0.0, 0.0],
                 # 渲染参数
                 z_clip_value=0.3,
                 # 场景随机化参数
                 enable_random_scene=False,
                 scene_generator=None,
                 safe_spawn_clearance=1.0,
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
        
        # 场景随机化配置
        self.enable_random_scene = enable_random_scene
        self.scene_generator = scene_generator
        self.safe_spawn_clearance = safe_spawn_clearance
        
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
            num_samples=num_samples,
            subdivide_times=subdivide_times,
            z_clip_value=z_clip_value,
        )
        
        # 内部状态初始化
        self.reset()

    def reset(self):
        """重置无人机状态"""
        # 运动学状态
        # 位置随机化：X/Y 使用 randn 分布在 0 附近，Z 使用 rand 分布在 [0.5, 2.5] 之间 (假设 init_p_range=2.0)
        # 原逻辑: torch.rand * range -> [0, range] (X,Y,Z 都是正的，且 Z 可能为 0)
        # 新逻辑: 
        #   X, Y: uniform(-range, range)
        #   Z: uniform(0.5, range + 0.5)
        self.p = (torch.rand(self.B, 3, device=self.device) - 0.5) * 2 * self.init_p_range
        self.p[:, 2] = torch.rand(self.B, device=self.device) * self.init_p_range + 0.5
        
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

    @torch.no_grad()
    def randomize_scene(self, num_obstacles=None):
        """
        随机生成新场景并更新渲染器和点云。
        
        使用 SceneGenerator 生成随机场景网格，替换当前渲染器中的网格。
        只影响渲染和碰撞检测用的点云，不影响无人机状态。
        
        Args:
            num_obstacles (int, optional): 障碍物数量，None 则由 SceneGenerator 随机决定。
            
        Returns:
            obstacle_info (list): 障碍物信息列表
        """
        if self.scene_generator is None:
            raise RuntimeError("randomize_scene() requires a SceneGenerator. "
                               "Pass scene_generator to DroneSimulator or set enable_random_scene=True.")
        
        scene_mesh, obstacle_info = self.scene_generator.generate(num_obstacles=num_obstacles)
        
        # 直接替换渲染器的 mesh 和点云，而非通过 update_mesh (它需要文件路径)
        self.renderer.mesh = scene_mesh
        self.renderer.obstacle_pcd = sample_points_from_meshes(
            scene_mesh, num_samples=self.num_samples
        ).to(self.device)
        # 使 extend 缓存失效
        self.renderer._extended_mesh_cache = None
        self.renderer._extended_mesh_bs = 0
        
        return obstacle_info

    @torch.no_grad()
    def safe_reset(self, arena_range=None, z_range=(1.0, 6.0)):
        """
        碰撞安全的环境重置：先重置动力学状态，然后用拒绝采样确保出生点不在障碍物内部。
        
        注意：z_range 语义为"高度范围"，内部通过 OBJ↔ROS 坐标转换正确映射。
        
        Args:
            arena_range (float, optional): 出生点水平范围，默认使用 self.init_p_range
            z_range (tuple): 出生点高度范围 (min, max)
            
        Returns:
            state (Tensor): 重置后的状态
        """
        # 先执行常规 reset（初始化所有动力学状态）
        self.reset()
        
        if arena_range is None:
            arena_range = self.init_p_range
        
        # 在 OBJ 坐标系中采样安全出生点（obstacle_pcd 在 OBJ 空间）
        safe_p_obj = sample_safe_points(
            obstacle_pcd=self.renderer.obstacle_pcd,
            num_points=self.B,
            arena_range=arena_range,
            z_range=z_range,
            min_clearance=self.safe_spawn_clearance,
            device=self.device,
        )
        
        # 从 OBJ 坐标系转换到 ROS 坐标系
        self.p = obj_to_ros(safe_p_obj)
        
        return self._get_state()

    @torch.no_grad()
    def sample_safe_target(self, arena_range=None, z_range=(1.5, 6.0),
                           min_distance=3.0, max_distance=8.0):
        """
        为当前出生点采样碰撞安全的目标点。
        
        注意：z_range 语义为"高度范围"，内部通过 OBJ↔ROS 坐标转换正确映射。
        
        Args:
            arena_range: 水平范围
            z_range: 高度范围
            min_distance: 到出生点最小距离
            max_distance: 到出生点最大距离
            
        Returns:
            targets (Tensor): (B, 3) 安全目标点（ROS 坐标系）
        """
        if arena_range is None:
            arena_range = self.init_p_range
        
        # 将出生点从 ROS 转换到 OBJ 坐标系（与 obstacle_pcd 一致）
        spawn_obj = ros_to_obj(self.p)
        
        targets_obj = sample_safe_targets(
            obstacle_pcd=self.renderer.obstacle_pcd,
            spawn_points=spawn_obj,
            arena_range=arena_range,
            z_range=z_range,
            min_clearance=self.safe_spawn_clearance,
            min_distance=min_distance,
            max_distance=max_distance,
            device=self.device,
        )
        
        # 从 OBJ 转换回 ROS
        return obj_to_ros(targets_obj)

    @torch.no_grad()
    def safe_reset_cross_map(self, arena_range=None, z_range=(1.0, 3.0)):
        """
        跨地图碰撞安全重置：出生点和目标点在场景的对向两侧。

        确保无人机必须穿越场景中央障碍物区域，不能绕边缘飞行。
        同时返回出生后的状态和目标位置。

        注意：z_range 语义为"高度范围"，内部通过 OBJ↔ROS 坐标转换正确映射。

        Args:
            arena_range (float, optional): 水平范围，默认使用 self.init_p_range
            z_range (tuple): 高度范围 (min, max)

        Returns:
            state (Tensor): 重置后的状态
            targets_ros (Tensor): (B, 3) 目标点（ROS 坐标系）
        """
        # 先执行常规 reset（初始化所有动力学状态）
        self.reset()

        if arena_range is None:
            arena_range = self.init_p_range

        # 在 OBJ 坐标系中采样跨地图出生/目标点对
        spawn_obj, target_obj = sample_cross_map_spawn_target(
            obstacle_pcd=self.renderer.obstacle_pcd,
            num_points=self.B,
            arena_range=arena_range,
            z_range=z_range,
            min_clearance=self.safe_spawn_clearance,
            device=self.device,
        )

        # 从 OBJ 转换到 ROS 坐标系
        self.p = obj_to_ros(spawn_obj)

        return self._get_state(), obj_to_ros(target_obj)

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
        
        # 计算纯推力向量（用于姿态解算）
        # 参考项目 env_cuda.py update_state_vec: a_thr = a_thr - g_std，其中 g_std = [0,0,-9.80665]
        # act_curr 是净加速度（包含重力），纯推力 = act_curr - gravity = act_curr - [0,0,-g] = act_curr + [0,0,g]
        # 关键修复：gravity_vec 是 [0,0,-9.8]，所以 thrust_net = act_curr - gravity_vec 
        # 等价于 act_curr - [0,0,-9.8] = act_curr + [0,0,9.8]
        thrust_net = self.act_curr - self.gravity_vec  # 这里 gravity_vec = [0,0,-9.8]
        
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

    def _knn_query(self, p=None):
        """
        共享的 KNN 查询，返回到最近障碍物的距离和向量。
        避免 calc_min_distance 和 vec_to_obj 各自独立调用 knn_points。

        注意: obstacle_pcd 在 OBJ 坐标系中，无人机位置在 ROS 坐标系中，
              需要先将无人机位置转换到 OBJ 坐标系，查询后再将向量转回 ROS。

        Args:
            p (Tensor, optional): 无人机位置 (B, 3)，ROS 坐标系。如果为 None，使用当前状态 self.p。

        Returns:
            dists (B,): 到最近障碍物的欧氏距离（标量距离在两个坐标系中相同）
            vecs (B, 3): 从无人机到最近障碍物点的向量（ROS 坐标系）
        """
        pos_ros = p if p is not None else self.p
        # 将无人机位置从 ROS 转换到 OBJ 坐标系
        pos_obj = ros_to_obj(pos_ros)
        p1 = pos_obj.unsqueeze(1)
        B_size = p1.shape[0]

        obstacle_pcd = self.renderer.obstacle_pcd
        if obstacle_pcd.shape[0] != B_size:
            obstacle_pcd_expanded = obstacle_pcd.expand(B_size, -1, -1)
        else:
            obstacle_pcd_expanded = obstacle_pcd

        # return_nn=True 直接返回最近邻坐标，避免手动索引
        result = knn_points(p1, obstacle_pcd_expanded, K=1, return_nn=True)
        sq_dists = result.dists.squeeze(-1).squeeze(-1)  # (B,)
        dists = torch.sqrt(sq_dists + 1e-6)

        nearest_points_obj = result.knn.squeeze(1).squeeze(1)  # (B, 3)
        vecs_obj = nearest_points_obj - pos_obj

        # 将向量从 OBJ 转换回 ROS 坐标系
        # 注意: 对于方向向量，只需要轴变换不需要位移，obj_to_ros 作为线性变换适用
        vecs_ros = obj_to_ros(vecs_obj)

        return dists, vecs_ros

    def calc_min_distance(self, p=None):
        """
        计算无人机到障碍物的最短距离。
        
        Args:
            p (Tensor, optional): 无人机位置 (B, 3)。如果为 None，使用当前状态 self.p。
            
        Returns:
            dist (B,)
        """
        dists, _ = self._knn_query(p)
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
        _, vecs = self._knn_query(p)
        return vecs

    def vec_to_obj_subdivided(self, n_subdiv=10, dt=None):
        """
        在子步插值位置上计算到最近障碍物的向量（参考项目 find_vec_to_nearest_pt 的实现）。

        沿当前步轨迹 p + v * t 均匀采样 n_subdiv 个点，逐点查询最近障碍物。
        与单点 vec_to_obj 相比，能捕获步内的碰撞风险。

        Args:
            n_subdiv (int): 子步数量，默认 10（与参考项目一致）。
            dt (float, optional): 当前步时间步长。默认使用 self.dt。

        Returns:
            vecs (n_subdiv, B, 3): 各子步位置到最近障碍物的向量。
        """
        current_dt = dt if dt is not None else self.dt
        sub_div = torch.linspace(0, current_dt, n_subdiv, device=self.device)

        vecs_list = []
        for t_frac in sub_div:
            p_interp = self.p + self.v * t_frac
            _, vec = self._knn_query(p_interp)
            vecs_list.append(vec)
        return torch.stack(vecs_list)  # (n_subdiv, B, 3)

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
