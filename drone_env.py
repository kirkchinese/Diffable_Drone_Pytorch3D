import torch
import os
import sys
from pytorch3d.ops import knn_points, sample_points_from_meshes

# 尝试导入同目录下的模块
try:
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import (DroneRenderer, compute_drone_safety_radius,
                                load_drone_mesh, transform_pos_ros2pt3d, transform_rot_ros2pt3d,
                                build_cam_mount_R)
    from scene_generator import (SceneGenerator, sample_safe_points, sample_safe_targets,
                                  sample_cross_map_spawn_target, obj_to_ros, ros_to_obj)
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from drone_dynamics import simulate_position_step, solve_attitude_from_thrust_and_goal_vec, update_dg
    from drone_renderer import (DroneRenderer, compute_drone_safety_radius,
                                load_drone_mesh, transform_pos_ros2pt3d, transform_rot_ros2pt3d,
                                build_cam_mount_R)
    from scene_generator import (SceneGenerator, sample_safe_points, sample_safe_targets,
                                  sample_cross_map_spawn_target, obj_to_ros, ros_to_obj)

try:
    from drone_renderer_dynamic import DynamicObstacle, MOTION_MODES
except ImportError:
    DynamicObstacle = None
    MOTION_MODES = ('linear',)

# 基础几何体目录和可用形状
_BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'base_model')
_OBSTACLE_SHAPES = ('圆柱体2_2_2', '圆环5_5_1', '方块', '椎体2_2_2', '球1_1')

class DroneSimulator:
    def __init__(self, 
                 batch_size=4, 
                 dt=0.02, 
                 device=None,
                 mesh_path="data/sample/sample.obj",
                 image_size=(480, 640),
                 focal_length=500.0,
                 principal_point=None,
                 lights_location=[[0.0, 0.0, -3.0]],
                 num_samples=20000,
                 subdivide_times=0,
                 # 动力学参数
                 enable_airmode=True,
                 enable_induced_drag=False,
                 noise_std=0.04,
                 grad_decay=0.8,
                 yaw_inertia=5.0,
                 yaw_ctl_delay=12.0,
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
                 init_margin_range=(0.3, 0.8),
                 # 运行参数
                 wind_std=0.1,
                 act_queue_len=2,
                 # 相机参数
                 cam_mode='auto',
                 cam_extrinsic=None,
                 cam_offset_body=None,
                 cam_mount_rpy=(0.0, 10.0, 0.0),
                 # 渲染参数
                 z_clip_value=0.3,
                 # 场景随机化参数
                 enable_random_scene=False,
                 scene_generator=None,
                 safe_spawn_clearance=1.0,
                 min_spawn_inter_distance=0.0,
                 random_init_yaw=True,
                 # 无人机网格参数
                 drone_mesh_path=None,
                 aero_margin=0.05,
                 # 多无人机交互参数
                 n_drones_per_group=1,
                 # 动态障碍物参数
                 enable_dynamic_obstacles=False,
                 num_dynamic_obstacles_range=(2, 5),
                 dynamic_obstacle_speed_range=(-0.5, 0.5),
                 dynamic_obstacle_scale_range=(0.2, 0.8),
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
        self.cam_mode = cam_mode
        self.cam_offset_body = cam_offset_body
        self.cam_mount_rpy = cam_mount_rpy

        # 手动模式：解析 3×4 外参矩阵 [R(3×3) | t(3×1)]
        self._cam_manual_R = None   # (3, 3)
        self._cam_manual_t = None   # (3,)
        if cam_extrinsic is not None:
            ext = torch.as_tensor(cam_extrinsic, device=self.device, dtype=torch.float32)
            if ext.numel() == 12:
                ext = ext.reshape(3, 4)
            self._cam_manual_R = ext[:, :3]   # 安装旋转
            self._cam_manual_t = ext[:, 3]    # 机体偏移
        self.num_samples = num_samples
        
        # 场景随机化配置
        self.enable_random_scene = enable_random_scene
        self.scene_generator = scene_generator
        self.safe_spawn_clearance = safe_spawn_clearance
        self.min_spawn_inter_distance = min_spawn_inter_distance
        self.random_init_yaw = random_init_yaw
        
        # 多无人机交互配置
        self.n_drones_per_group = n_drones_per_group
        self._inter_drone_eye_mask = None  # 延迟初始化，依赖 n_drones_per_group
        self._per_group_drone_meshes = None  # per-group 渲染用，每帧更新

        # 动态障碍物配置
        self.enable_dynamic_obstacles = enable_dynamic_obstacles
        self.num_dynamic_obstacles_range = num_dynamic_obstacles_range
        self.dynamic_obstacle_speed_range = dynamic_obstacle_speed_range
        self.dynamic_obstacle_scale_range = dynamic_obstacle_scale_range
        self._dynamic_obstacles = []
        self._base_mesh_cache = {}  # 缓存已加载的基础几何体网格
        
        # 无人机网格加载
        self.drone_mesh = None
        self.drone_bounding_radius = 0.15  # 默认值
        self.aero_margin = aero_margin
        if drone_mesh_path is not None:
            drone_mesh_path = self._resolve_path(drone_mesh_path)
            if os.path.exists(drone_mesh_path):
                self.drone_mesh, self._drone_centroid, self.drone_bounding_radius = \
                    load_drone_mesh(drone_mesh_path, device=self.device)
                centered = self.drone_mesh.verts_packed() - self._drone_centroid
                # OBJ Y-up → ROS Z-up 旋转 (等效于绕 X 轴旋转 -90°)：
                #   body_X = obj_X, body_Y = -obj_Z, body_Z = obj_Y
                # 列交换 [0,2,1] 的行列式为 -1（反射），翻转面法线导致渲染异常；
                # 乘以 [1,-1,1] 后行列式 = +1，是正确的右手旋转。
                self._drone_verts_centered = centered[:, [0, 2, 1]] * torch.tensor([1.0, -1.0, 1.0], device=self.device)

                # 从网格几何计算相机安装基准偏移（未缩放，仿照真机前下方安装位置）
                x_front = self._drone_verts_centered[:, 0].max().item()
                z_bottom = self._drone_verts_centered[:, 2].min().item()
                self._mesh_cam_offset_base = torch.tensor(
                    [x_front * 1.15, 0.0, z_bottom * 0.3],
                    device=self.device,
                )
                print(f"[DroneSimulator] 无人机网格已加载: {drone_mesh_path}, "
                      f"包围球半径={self.drone_bounding_radius:.3f}m, "
                      f"安全半径={self.drone_bounding_radius + aero_margin:.3f}m, "
                      f"相机基准偏移(未缩放)={self._mesh_cam_offset_base.tolist()}")
            else:
                print(f"[DroneSimulator] 警告: 无人机网格文件不存在: {drone_mesh_path}")

        # 渲染器初始化
        mesh_path = self._resolve_path(mesh_path)
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

    @staticmethod
    def _resolve_path(path):
        """将相对路径解析为绝对路径。"""
        if not os.path.exists(path) and not path.startswith("/"):
            potential = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if os.path.exists(potential):
                return potential
        return path

    def reset(self):
        """重置无人机状态"""
        self.p = (torch.rand(self.B, 3, device=self.device) - 0.5) * 2 * self.init_p_range
        self.p[:, 2] = torch.rand(self.B, device=self.device) * self.init_p_range + 0.5

        # gravity_vec 已在 __init__ 中创建，batch size 不变无需重建
        
        self.v = torch.randn(self.B, 3, device=self.device) * self.init_v_range
        self.a = torch.zeros(self.B, 3, device=self.device)
        self.act_curr = torch.zeros(self.B, 3, device=self.device)
        self.R = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.B, 1, 1)
        if self.random_init_yaw:
            yaw = (torch.rand(self.B, device=self.device) * 2.0 - 1.0) * torch.pi
            c = torch.cos(yaw)
            s = torch.sin(yaw)
            o = torch.zeros_like(yaw)
            l = torch.ones_like(yaw)
            self.R = torch.stack([
                c, -s, o,
                s, c, o,
                o, o, l,
            ], -1).reshape(self.B, 3, 3)
        
        # 环境扰动
        self.dg = torch.randn((self.B, 3), device=self.device) * self.init_dg_range
        
        # 模拟控制延迟的队列
        self.act_queue = [torch.zeros(self.B, 3, device=self.device) for _ in range(self.act_queue_len)]
        
        # 安全边距：始终从 init_margin_range 采样，保证训练/评估一致
        # 无人机网格仅作为视觉表示，按 margin 缩放渲染大小
        low, high = self.init_margin_range
        self.margin = torch.rand((self.B,), device=self.device) * (high - low) + low

        return self._get_state()

    def get_scaled_cam_offset(self):
        """按当前 margin 缩放的网格比例相机偏移 (B, 3)。

        相机安装位置由网格几何决定，并随 margin 缩放保持与机体的相对关系。
        若未加载无人机网格，返回零偏移。
        """
        if not hasattr(self, '_mesh_cam_offset_base'):
            return torch.zeros(self.B, 3, device=self.device)
        base_safety = self.drone_bounding_radius + self.aero_margin
        scales = (self.margin / base_safety).unsqueeze(1)          # (B, 1)
        return self._mesh_cam_offset_base.unsqueeze(0) * scales    # (B, 3)

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
            min_inter_distance=self.min_spawn_inter_distance,
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
            min_inter_distance=self.min_spawn_inter_distance,
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
            min_inter_distance=self.min_spawn_inter_distance,
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

        # 更新动态障碍物位置
        if self._dynamic_obstacles:
            self._step_dynamic_obstacles(current_dt)

        return self._get_state()

    # ================================================================
    # 动态场景合成（无人机机体渲染 + 动态障碍物）
    # ================================================================

    def _update_render_scene(self):
        """
        预计算当前帧的动态网格数据，供 per-group 渲染使用。

        每组相机只能看到同组无人机（自身在相机后方不会遮挡），不跨组可见。
        动态障碍物对所有相机可见。
        """
        # --- 无人机机体网格（按组分） ---
        if self.drone_mesh is not None and self.n_drones_per_group > 1:
            all_drone_meshes = self._compose_drone_meshes()
            G = self.n_drones_per_group
            n_groups = self.B // G
            self._per_group_drone_meshes = [
                all_drone_meshes[g * G : g * G + G] for g in range(n_groups)
            ]
        else:
            self._per_group_drone_meshes = None

        # --- 动态障碍物网格 + 点云（所有组共享） ---
        self._frame_obs_meshes = []
        self._frame_obs_pcds = []
        for obs in self._dynamic_obstacles:
            self._frame_obs_meshes.append(obs.get_transformed_mesh())
            self._frame_obs_pcds.append(obs.get_transformed_pcd())

        # 对于无多机分组的情况，沿用旧路径直接设置渲染器
        if self._per_group_drone_meshes is None:
            if self._frame_obs_meshes:
                self.renderer.set_dynamic_meshes(
                    self._frame_obs_meshes,
                    self._frame_obs_pcds if self._frame_obs_pcds else None)
            elif self.renderer._dynamic_meshes:
                # 仅当已有动态网格时才清除，避免无谓地使 extend 缓存失效
                self.renderer.clear_dynamic_meshes()

    def _render_per_group(self, renderer, R_camera, T_camera, **render_kw):
        """
        按组渲染：每组相机仅看到同组无人机 + 动态障碍物。

        Args:
            renderer: DroneRenderer 或 DroneRendererVariant。
            R_camera: (B, 3, 3) 相机旋转矩阵。
            T_camera: (B, 3) 相机平移向量。
            **render_kw: 传递给 renderer.render() 的额外参数。

        Returns:
            (rgb, depth): 拼接后的 (B, H, W, 3) 和 (B, H, W)。
        """
        parent = renderer.parent if hasattr(renderer, 'parent') else renderer
        G = self.n_drones_per_group
        n_groups = len(self._per_group_drone_meshes)
        obs_meshes = self._frame_obs_meshes
        obs_pcds = self._frame_obs_pcds if self._frame_obs_pcds else None

        rgb_parts = []
        depth_parts = []

        for g in range(n_groups):
            sl = slice(g * G, g * G + G)
            group_meshes = self._per_group_drone_meshes[g] + obs_meshes
            parent.set_dynamic_meshes(group_meshes, obs_pcds)
            rgb_g, depth_g = renderer.render(
                R=R_camera[sl], T=T_camera[sl], **render_kw)
            if rgb_g is not None:
                rgb_parts.append(rgb_g)
            if depth_g is not None:
                depth_parts.append(depth_g)

        # 不整除的尾部（无对应组，只渲染障碍物）
        remainder = self.B - n_groups * G
        if remainder > 0:
            sl = slice(n_groups * G, self.B)
            parent.set_dynamic_meshes(obs_meshes or [], obs_pcds)
            rgb_g, depth_g = renderer.render(
                R=R_camera[sl], T=T_camera[sl], **render_kw)
            if rgb_g is not None:
                rgb_parts.append(rgb_g)
            if depth_g is not None:
                depth_parts.append(depth_g)

        # 最终恢复：设置障碍物 PCD 供碰撞检测 (full_obstacle_pcd) 使用
        parent.set_dynamic_meshes(obs_meshes or [], obs_pcds)

        rgb = torch.cat(rgb_parts, dim=0) if rgb_parts else None
        depth = torch.cat(depth_parts, dim=0) if depth_parts else None
        return rgb, depth

    def render_with_renderer(self, renderer, R_camera, T_camera, **render_kw):
        """
        使用指定渲染器进行 per-group 渲染（公共接口，供外部调用如 visualize_eval）。

        调用前须已执行 _update_render_scene()（通常由 render() 触发）。

        Returns:
            (rgb, depth): 拼接后的结果。
        """
        if self._per_group_drone_meshes is not None:
            return self._render_per_group(renderer, R_camera, T_camera, **render_kw)
        return renderer.render(R=R_camera, T=T_camera, **render_kw)

    @torch.no_grad()
    def _compose_drone_meshes(self):
        """
        在所有无人机当前位置生成变换后的无人机网格（PyTorch3D 坐标系）。

        Returns:
            list[Meshes]: B 个变换后的无人机网格。
        """
        from pytorch3d.structures import Meshes

        base_verts = self._drone_verts_centered  # (V, 3)
        base_faces = self.drone_mesh.faces_packed()
        base_tex = self.drone_mesh.textures
        p_pt3d = transform_pos_ros2pt3d(self.p)  # (B, 3)
        base_safety = self.drone_bounding_radius + self.aero_margin

        # 批量计算所有无人机的世界坐标顶点
        R_pt3d = transform_rot_ros2pt3d(self.R)  # (B, 3, 3)
        scales = (self.margin / base_safety).view(self.B, 1, 1)  # (B, 1, 1)
        verts_all = torch.bmm(
            base_verts.unsqueeze(0).expand(self.B, -1, -1) * scales,
            R_pt3d.transpose(1, 2)
        ) + p_pt3d.unsqueeze(1)  # (B, V, 3)

        return [Meshes(verts=[verts_all[b]], faces=[base_faces], textures=base_tex)
                for b in range(self.B)]

    def _step_dynamic_obstacles(self, dt):
        """推进所有动态障碍物一步。"""
        for obs in self._dynamic_obstacles:
            obs.step(dt)

    @torch.no_grad()
    def _load_base_mesh(self, name):
        """加载并缓存基础几何体网格，居中并归一化到单位包围球半径，避免每 episode 重复 IO。"""
        if name in self._base_mesh_cache:
            return self._base_mesh_cache[name]

        from pytorch3d.io import load_obj
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        obj_path = os.path.join(_BASE_MODEL_DIR, f'{name}.obj')
        verts, faces_data, _ = load_obj(obj_path, load_textures=False)
        verts = verts.to(self.device)
        faces = faces_data.verts_idx.to(self.device)

        # 居中 + 归一化到单位包围球（scale 参数直接控制实际大小）
        centroid = verts.mean(dim=0)
        verts = verts - centroid
        radius = verts.norm(dim=1).max().item()
        if radius > 1e-6:
            verts = verts / radius

        tex = TexturesVertex(verts_features=torch.ones(1, verts.shape[0], 3, device=self.device))
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        self._base_mesh_cache[name] = mesh
        return mesh

    @torch.no_grad()
    def randomize_dynamic_obstacles(self, arena_range=None):
        """
        随机生成动态障碍物（多种几何体 + 多种运动模式），每 episode 开始时调用。

        从 data/base_model/ 加载真实 OBJ 几何体（方块、球、圆柱、锥体、圆环），
        并为每个障碍物随机分配运动模式（linear/sinusoidal/circular/figure8/pendulum/static）。
        """
        if DynamicObstacle is None:
            print("[DroneSimulator] 警告: 无法导入 DynamicObstacle，跳过动态障碍物")
            return
        self.clear_dynamic_obstacles()

        if arena_range is None:
            arena_range = self.init_p_range

        from pytorch3d.renderer import TexturesVertex
        from pytorch3d.structures import Meshes

        lo, hi = self.num_dynamic_obstacles_range
        num = torch.randint(lo, hi + 1, (1,)).item()
        speed_lo, speed_hi = self.dynamic_obstacle_speed_range
        scale_lo, scale_hi = self.dynamic_obstacle_scale_range
        n_shapes = len(_OBSTACLE_SHAPES)
        n_modes = len(MOTION_MODES)

        for _ in range(num):
            # 随机几何体（从 data/base_model/ 缓存加载）
            shape_name = _OBSTACLE_SHAPES[torch.randint(n_shapes, (1,)).item()]
            base_mesh = self._load_base_mesh(shape_name)
            base_verts = base_mesh.verts_list()[0]

            # 随机颜色纹理
            color = torch.rand(3, device=self.device)
            mesh = Meshes(
                verts=[base_verts], faces=[base_mesh.faces_packed()],
                textures=TexturesVertex(verts_features=color.expand(base_verts.shape[0], 3)[None]),
            )

            # OBJ 坐标系随机位置
            pos = torch.rand(3, device=self.device)
            pos[0] = pos[0] * 2 * arena_range - arena_range
            pos[1] = pos[1] * arena_range * 0.8 + 0.3
            pos[2] = pos[2] * 2 * arena_range - arena_range

            # 随机速度
            vel = torch.rand(3, device=self.device) * (speed_hi - speed_lo) + speed_lo
            scale = (torch.rand(1, device=self.device) * (scale_hi - scale_lo) + scale_lo).item()

            # 随机角速度
            angular_vel = torch.randn(3, device=self.device) * 0.5

            # 随机运动模式 + 参数
            motion_mode = MOTION_MODES[torch.randint(n_modes, (1,)).item()]
            motion_params = self._random_motion_params(motion_mode, arena_range)

            obs = DynamicObstacle(
                mesh=mesh, position=pos, velocity=vel,
                angular_velocity=angular_vel, scale=scale,
                num_pcd_samples=500, device=self.device,
                motion_mode=motion_mode, motion_params=motion_params,
            )
            self._dynamic_obstacles.append(obs)

        if num > 0:
            modes = [o.motion_mode for o in self._dynamic_obstacles]
            print(f"[DynamicObs] 已生成 {num} 个动态障碍物, 运动模式: {modes}")

    def _random_motion_params(self, mode, arena_range):
        """为给定运动模式生成随机参数（纯 torch，无 numpy）。"""
        params = {}
        if mode in ('sinusoidal', 'pendulum'):
            amp_hi = min(2.0, arena_range * 0.5)
            params['amplitude'] = float(torch.rand(1).item() * (amp_hi - 0.5) + 0.5)
            params['frequency'] = float(torch.rand(1).item() * 0.6 + 0.2)
            params['phase'] = float(torch.rand(1).item() * 6.283185307179586)
        elif mode in ('circular', 'figure8'):
            r_hi = min(2.0, arena_range * 0.4)
            r = float(torch.rand(1).item() * (r_hi - 0.5) + 0.5)
            params['frequency'] = float(torch.rand(1).item() * 0.4 + 0.1)
            # 随机轨道平面 (Gram-Schmidt 正交化)
            u = torch.randn(3, device=self.device)
            u = u / u.norm()
            ref = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            if torch.dot(u, ref).abs().item() > 0.9:
                ref = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            v = ref - torch.dot(ref, u) * u
            v = v / v.norm()
            params['plane_u'] = u
            params['plane_v'] = v
            if mode == 'circular':
                params['radius'] = r
            else:
                params['amplitude_u'] = r
                params['amplitude_v'] = r * float(torch.rand(1).item() * 0.4 + 0.3)
        return params

    def clear_dynamic_obstacles(self):
        """清除所有动态障碍物并重置渲染器合成。"""
        self._dynamic_obstacles = []
        self.renderer.clear_dynamic_meshes()

    def render(self, camera_pitch=None, cam_mount_R=None, cam_offset_body=None,
               return_tensor=True, return_rgb=True, return_depth=True, dt=None):
        """
        渲染当前帧
        
        Args:
            camera_pitch (float|Tensor, optional): 相机俯仰角 (度)。
                当 cam_mount_R 未提供时使用，默认取 self.cam_mount_rpy[1]。
            cam_mount_R (Tensor, optional): 相机安装旋转矩阵 (B,3,3)；覆盖 camera_pitch。
            cam_offset_body (Tensor|list, optional): 相机机体坐标系偏移 (3,) 或 (B,3)。
                默认按网格几何自动计算（前下方，随 margin 缩放）；
                若无网格则使用 self.cam_offset_body 兜底。
            return_tensor (bool): 是否返回 Tensor
            return_rgb (bool): 是否返回 RGB
            return_depth (bool): 是否返回 Depth
            dt (float, optional): 仿真时间步长。
        """
        self._update_render_scene()

        # 确定相机安装旋转和位置偏移
        if cam_mount_R is None and cam_offset_body is None:
            if self.cam_mode == 'manual' and self._cam_manual_R is not None:
                # 手动模式：使用用户提供的外参矩阵
                cam_mount_R = self._cam_manual_R
                cam_offset_body = self._cam_manual_t
            else:
                # 自动模式：网格比例偏移 + RPY 旋转
                if hasattr(self, '_mesh_cam_offset_base'):
                    cam_offset_body = self.get_scaled_cam_offset()
                elif self.cam_offset_body is not None:
                    cam_offset_body = self.cam_offset_body
        elif cam_offset_body is None:
            # 仅 cam_mount_R 被显式传入，自动计算偏移
            if self.cam_mode == 'manual' and self._cam_manual_t is not None:
                cam_offset_body = self._cam_manual_t
            elif hasattr(self, '_mesh_cam_offset_base'):
                cam_offset_body = self.get_scaled_cam_offset()
            elif self.cam_offset_body is not None:
                cam_offset_body = self.cam_offset_body

        if cam_mount_R is None and camera_pitch is None:
            camera_pitch = self.cam_mount_rpy[1]

        R_camera, T_camera = self.renderer.compute_view_matrix(
            p_ros=self.p, 
            R_ros=self.R, 
            camera_pitch_deg=camera_pitch,
            cam_offset_body=cam_offset_body,
            cam_mount_R=cam_mount_R,
        )

        # 多机分组：按组渲染，每组相机只看到同组无人机
        if self._per_group_drone_meshes is not None:
            rgb, depth = self._render_per_group(
                self.renderer, R_camera, T_camera,
                return_tensor=return_tensor,
                return_rgb=return_rgb,
                return_depth=return_depth,
                dt=dt,
            )
        else:
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

        obstacle_pcd = self.renderer.full_obstacle_pcd
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

        沿当前步轨迹 p + v * t 均匀采样 n_subdiv 个点，**一次性批量 KNN 查询**
        所有子步的最近障碍物。与单点 vec_to_obj 相比，能捕获步内的碰撞风险。

        优化：使用单次 CUDA 调用处理所有子步。本实现将 S×B 个查询点
        合并为一次 knn_points 调用，消除 S 次独立 CUDA kernel launch 的开销。

        Args:
            n_subdiv (int): 子步数量，默认 10（与参考项目一致）。
            dt (float, optional): 当前步时间步长。默认使用 self.dt。

        Returns:
            vecs (n_subdiv, B, 3): 各子步位置到最近障碍物的向量（ROS 坐标系）。
        """
        current_dt = dt if dt is not None else self.dt
        sub_div = torch.linspace(0, current_dt, n_subdiv, device=self.device)

        # 批量生成所有子步的插值位置: (S, B, 3) in ROS
        p_all_ros = self.p.unsqueeze(0) + self.v.unsqueeze(0) * sub_div[:, None, None]
        S, B, _ = p_all_ros.shape

        # 转换到 OBJ 坐标系并展平为 (S*B, 3)
        p_flat_obj = ros_to_obj(p_all_ros.reshape(S * B, 3))

        # 单次批量 KNN 查询: (1, S*B, 3) vs (1, N, 3)
        p_query = p_flat_obj.unsqueeze(0)  # (1, S*B, 3)
        obstacle_pcd = self.renderer.full_obstacle_pcd  # (1, N, 3)

        result = knn_points(p_query, obstacle_pcd, K=1, return_nn=True)
        nearest_obj = result.knn.squeeze(0).squeeze(1)  # (S*B, 3)

        # 计算向量 (OBJ 空间) 并转换回 ROS
        vecs_obj = nearest_obj - p_flat_obj  # (S*B, 3)
        vecs_ros = obj_to_ros(vecs_obj)      # (S*B, 3)

        return vecs_ros.reshape(S, B, 3)  # (S, B, 3)

    def update_mesh(self, mesh_path, num_samples=None):
        """更换障碍物网格模型"""
        mesh_path = self._resolve_path(mesh_path)
        print(f"Updating mesh to: {mesh_path}")
        samples = num_samples if num_samples is not None else self.num_samples
        self.renderer.update_mesh(mesh_path, num_samples=samples)

    @staticmethod
    def clean_depth_map(depth, min_dist=0.2, max_dist=10.0):
        return DroneRenderer.clean_depth_map(depth, min_dist=min_dist, max_dist=max_dist)

    # ================================================================
    # 无人机间交互
    # ================================================================

    def inter_drone_distances(self, p=None):
        """
        计算同组无人机之间的距离（使用椭球模型，参考项目 Z 轴 2x 缩放）。

        参考项目在 CUDA kernel 中使用 4*(oz-cz)^2 (即 Z 轴距离权重 4x)，
        等效于 Z 轴方向缩放 2x 的椭球碰撞模型。

        Args:
            p: (B, 3) 位置，默认使用 self.p。

        Returns:
            min_dist: (B,) 每架无人机到同组最近其他无人机的椭球距离。
            min_vec: (B, 3) 到最近同组无人机的向量 (ROS 坐标系)。
        """
        if self.n_drones_per_group <= 1:
            # 单机模式：无交互
            return (torch.full((self.B,), 1e6, device=self.device),
                    torch.zeros(self.B, 3, device=self.device))

        pos = p if p is not None else self.p
        B = pos.shape[0]
        G = self.n_drones_per_group

        # 将批量按组重排 (n_groups, G, 3)
        n_groups = B // G
        p_grouped = pos[:n_groups * G].view(n_groups, G, 3)

        # 计算组内所有两两距离（椭球距离：Z 轴权重 2x）
        # diff: (n_groups, G, G, 3)
        diff = p_grouped.unsqueeze(2) - p_grouped.unsqueeze(1)
        # 椭球距离：sqrt(dx^2 + dy^2 + 4*dz^2)
        ellipsoid_dist_sq = diff[..., 0]**2 + diff[..., 1]**2 + 4 * diff[..., 2]**2
        ellipsoid_dist = torch.sqrt(ellipsoid_dist_sq + 1e-8)

        # 自身距离置为极大值（缓存 eye_mask 避免每步重建）
        if self._inter_drone_eye_mask is None or self._inter_drone_eye_mask.shape[-1] != G:
            self._inter_drone_eye_mask = torch.eye(G, device=self.device, dtype=torch.bool).unsqueeze(0)
        ellipsoid_dist = ellipsoid_dist.masked_fill(self._inter_drone_eye_mask, 1e6)

        # 每架无人机的最近同组无人机
        min_dist_grouped, min_idx = ellipsoid_dist.min(dim=2)  # (n_groups, G)

        # 减去双方 margin 之和，使返回距离语义与障碍物一致（负值 = 碰撞）
        margin_grouped = self.margin[:n_groups * G].view(n_groups, G)
        margin_other = margin_grouped.gather(1, min_idx)
        min_dist_grouped = min_dist_grouped - margin_grouped - margin_other

        # 对应向量
        min_idx_exp = min_idx.unsqueeze(-1).expand(-1, -1, 3)
        min_vec_grouped = diff.gather(2, min_idx_exp.unsqueeze(2)).squeeze(2)

        # 展平回 (B,)
        min_dist = min_dist_grouped.view(-1)
        min_vec = min_vec_grouped.view(-1, 3)

        # 处理 B 不能整除 G 的剩余部分
        if B > n_groups * G:
            pad_n = B - n_groups * G
            min_dist = torch.cat([min_dist,
                                  torch.full((pad_n,), 1e6, device=self.device)])
            min_vec = torch.cat([min_vec,
                                 torch.zeros(pad_n, 3, device=self.device)])

        return min_dist, min_vec

    def inter_drone_vec_subdivided(self, n_subdiv=10, dt=None):
        """
        子步细分版本的无人机间距离计算（全向量化，无 Python 循环）。

        沿轨迹 p + v * t 均匀采样 n_subdiv 个点，
        计算每个子步位置处到同组最近无人机的向量。

        Returns:
            vecs: (S, B, 3) 各子步位置到最近同组无人机的向量。
        """
        if self.n_drones_per_group <= 1:
            current_dt = dt if dt is not None else self.dt
            S = n_subdiv
            return torch.zeros(S, self.B, 3, device=self.device)

        current_dt = dt if dt is not None else self.dt
        sub_div = torch.linspace(0, current_dt, n_subdiv, device=self.device)
        p_all = self.p.unsqueeze(0) + self.v.unsqueeze(0) * sub_div[:, None, None]
        S, B, _ = p_all.shape
        G = self.n_drones_per_group
        n_groups = B // G

        # 将 S 个子步与 n_groups 个组合并为一个批维度: (S*n_groups, G, 3)
        p_grouped = p_all[:, :n_groups * G].view(S * n_groups, G, 3)
        diff = p_grouped.unsqueeze(2) - p_grouped.unsqueeze(1)  # (S*n_groups, G, G, 3)

        ellipsoid_dist_sq = diff[..., 0]**2 + diff[..., 1]**2 + 4 * diff[..., 2]**2
        ellipsoid_dist = torch.sqrt(ellipsoid_dist_sq + 1e-8)

        if self._inter_drone_eye_mask is None or self._inter_drone_eye_mask.shape[-1] != G:
            self._inter_drone_eye_mask = torch.eye(G, device=self.device, dtype=torch.bool).unsqueeze(0)
        ellipsoid_dist = ellipsoid_dist.masked_fill(self._inter_drone_eye_mask, 1e6)

        _, min_idx = ellipsoid_dist.min(dim=2)  # (S*n_groups, G)
        min_idx_exp = min_idx.unsqueeze(-1).expand(-1, -1, 3)
        min_vec_grouped = diff.gather(2, min_idx_exp.unsqueeze(2)).squeeze(2)  # (S*n_groups, G, 3)

        # 还原: (S*n_groups, G, 3) -> (S, n_groups*G, 3)
        vecs = min_vec_grouped.view(S, n_groups * G, 3)

        if B > n_groups * G:
            pad_n = B - n_groups * G
            vecs = torch.cat([vecs, torch.zeros(S, pad_n, 3, device=self.device)], dim=1)

        return vecs

    def combined_vec_to_nearest(self, n_subdiv=10, dt=None):
        """
        计算到最近障碍物（包括其他无人机）的向量 (子步细分)。

        融合静态障碍物和动态无人机的最近点查询：
        对每个子步取两者中距离更近者。

        Returns:
            vecs: (S, B, 3) 到最近物体（障碍物或无人机）的向量。
        """
        # 静态障碍物的子步查询
        vecs_obs = self.vec_to_obj_subdivided(n_subdiv=n_subdiv, dt=dt)  # (S, B, 3)

        if self.n_drones_per_group <= 1:
            return vecs_obs

        # 其他无人机的子步查询
        vecs_drone = self.inter_drone_vec_subdivided(n_subdiv=n_subdiv, dt=dt)

        # 取距离更近者（比较平方范数，省去 sqrt）
        dist_sq_obs = (vecs_obs * vecs_obs).sum(dim=-1)
        dist_sq_drone = (vecs_drone * vecs_drone).sum(dim=-1)

        use_drone = dist_sq_drone < dist_sq_obs
        use_drone_exp = use_drone.unsqueeze(-1)  # (S, B, 1)

        return torch.where(use_drone_exp, vecs_drone, vecs_obs)

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
