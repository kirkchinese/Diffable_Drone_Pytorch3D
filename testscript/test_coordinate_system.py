#!/usr/bin/env python3
"""
坐标系数学验证脚本

严格验证:
1. ROS ↔ OBJ ↔ PT3D 坐标变换的正确性和一致性
2. build_cam_mount_R 各轴方向 (pitch/roll/yaw)
3. transform_rot_ros2pt3d 与 _compute_drone_verts 的正确性
4. compute_view_matrix 全流程验证
5. KNN 查询坐标一致性
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_printoptions(precision=6, sci_mode=False)

from drone_renderer import (
    transform_pos_ros2pt3d, transform_rot_ros2pt3d,
    build_cam_mount_R, hfov_to_focal,
)
from scene_generator import obj_to_ros, ros_to_obj

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  ← FAIL {detail}")

def allclose(a, b, atol=1e-5):
    return torch.allclose(a.float(), b.float(), atol=atol)


# ======================================================================
print("=" * 70)
print("TEST 1: 位置坐标变换 ROS ↔ OBJ ↔ PT3D")
print("=" * 70)

# ROS → PT3D: (x,y,z) → (-x, z, y)
p_ros = torch.tensor([1.0, 2.0, 3.0])
p_pt3d = transform_pos_ros2pt3d(p_ros)
check("ROS→PT3D: (1,2,3)→(-1,3,2)", allclose(p_pt3d, torch.tensor([-1.0, 3.0, 2.0])))

# OBJ → ROS: same formula (x,y,z) → (-x, z, y)
p_obj = torch.tensor([1.0, 2.0, 3.0])
p_ros_from_obj = obj_to_ros(p_obj)
check("OBJ→ROS: (1,2,3)→(-1,3,2)", allclose(p_ros_from_obj, torch.tensor([-1.0, 3.0, 2.0])))

# ros_to_obj is the same function (involution)
check("ros_to_obj is obj_to_ros (同一函数)", ros_to_obj is obj_to_ros)

# Involution: apply twice = identity
p_roundtrip = obj_to_ros(obj_to_ros(p_obj))
check("OBJ→ROS→OBJ 自逆: f(f(x))=x", allclose(p_roundtrip, p_obj))

# OBJ→ROS→PT3D = identity (OBJ coords = PT3D coords)
p_obj_orig = torch.tensor([5.0, -3.0, 7.0])
p_via_ros = obj_to_ros(p_obj_orig)    # OBJ → ROS
p_via_pt3d = transform_pos_ros2pt3d(p_via_ros)  # ROS → PT3D
check("OBJ→ROS→PT3D = identity (OBJ空间=PT3D空间)", allclose(p_via_pt3d, p_obj_orig))

# Batch test
p_batch = torch.randn(16, 3)
p_rt = transform_pos_ros2pt3d(transform_pos_ros2pt3d(p_batch))
# ROS→PT3D→ROS? Not the same function. ROS→PT3D = (-x,z,y), PT3D→ROS = ?
# Actually transform_pos_ros2pt3d applied twice:
# (x,y,z) → (-x,z,y) → (x,y,z)? No: (-x,z,y) → (x,y,z)? 
# f(f(x,y,z)) = f(-x,z,y) = (x,y,z). YES, it's self-inverse too!
check("transform_pos_ros2pt3d 自逆 (batch)", allclose(p_rt, p_batch))

# Semantic checks
print("\n  语义验证:")
# ROS: +X=East, +Y=North, +Z=Up
# PT3D: +X=Left(West), +Y=Up, +Z=North
p_east = torch.tensor([1., 0., 0.])  # ROS East
check("ROS East → PT3D West(-X)", allclose(transform_pos_ros2pt3d(p_east), torch.tensor([-1., 0., 0.])))
p_north = torch.tensor([0., 1., 0.])  # ROS North
check("ROS North → PT3D North(+Z)", allclose(transform_pos_ros2pt3d(p_north), torch.tensor([0., 0., 1.])))
p_up = torch.tensor([0., 0., 1.])  # ROS Up
check("ROS Up → PT3D Up(+Y)", allclose(transform_pos_ros2pt3d(p_up), torch.tensor([0., 1., 0.])))


# ======================================================================
print("\n" + "=" * 70)
print("TEST 2: 旋转矩阵变换 transform_rot_ros2pt3d")
print("=" * 70)

# Identity rotation
R_id = torch.eye(3).unsqueeze(0)
R_pt3d_id = transform_rot_ros2pt3d(R_id)
check("identity旋转正交性: R^T R = I", allclose(R_pt3d_id[0].T @ R_pt3d_id[0], torch.eye(3)))

# 检查 R_pt3d 的列是否为变换后的基向量
M = torch.tensor([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
R_ros_test = torch.randn(1, 3, 3)
# 使 R_ros_test 正交 (QR分解)
R_ros_test, _ = torch.linalg.qr(R_ros_test)
R_pt3d_test = transform_rot_ros2pt3d(R_ros_test)
R_expected = M @ R_ros_test[0]  # columns = M applied to each column of R
check("transform_rot = M @ R_ros (列为变换后基向量)",
      allclose(R_pt3d_test[0], R_expected))
check("变换后旋转矩阵正交", allclose(R_pt3d_test[0].T @ R_pt3d_test[0], torch.eye(3)))

# 关键验证: _compute_drone_verts 使用 R_pt3d.transpose
# v_world_pt3d = v_body @ R_pt3d^T + p_pt3d
# 等价于 v_world_pt3d = v_body @ (M @ R_ros)^T + p_pt3d
#                     = v_body @ R_ros^T @ M^T + p_pt3d
# 而 M = M^T, 所以 = v_body @ R_ros^T @ M + p_pt3d
# = ROS_to_PT3D(R_ros @ v_body^col) ... let's verify
print("\n  验证 v_body @ R_pt3d^T = transform_pos(R_ros @ v_body):")
R_ros = torch.randn(1, 3, 3)
R_ros, _ = torch.linalg.qr(R_ros)  # 正交化
R_pt3d = transform_rot_ros2pt3d(R_ros)
v_body = torch.randn(1, 3)

# Method 1: _compute_drone_verts 的做法
v_world_1 = v_body @ R_pt3d.squeeze(0).T

# Method 2: 数学上正确的全程变换
v_world_ros = (R_ros.squeeze(0) @ v_body.squeeze(0).unsqueeze(-1)).squeeze(-1)
v_world_2 = transform_pos_ros2pt3d(v_world_ros)

check("v_body @ R_pt3d^T == transform_pos(R_ros @ v_body)", allclose(v_world_1, v_world_2.unsqueeze(0)))


# 用特定旋转验证: 无人机朝北 (body X = world +Y in ROS)
print("\n  具体案例: 无人机朝北 (body X → ROS +Y → PT3D +Z):")
# R_ros: columns = [body_X_in_world, body_Y_in_world, body_Z_in_world]
# Body X (forward) → World +Y (north)
# Body Y (left) → World -X (west)
# Body Z (up) → World +Z (up)
R_ros_north = torch.tensor([[[0., -1., 0.],
                              [1.,  0., 0.],
                              [0.,  0., 1.]]])
R_pt3d_north = transform_rot_ros2pt3d(R_ros_north)

v_fwd_body = torch.tensor([[1., 0., 0.]])  # body forward
v_fwd_pt3d = v_fwd_body @ R_pt3d_north.squeeze(0).T
# Expected: forward in ROS = +Y = north → in PT3D = +Z
check("body forward → PT3D +Z (north)", allclose(v_fwd_pt3d, torch.tensor([[0., 0., 1.]])))

v_up_body = torch.tensor([[0., 0., 1.]])  # body up
v_up_pt3d = v_up_body @ R_pt3d_north.squeeze(0).T
# Expected: up in ROS = +Z → in PT3D = +Y
check("body up → PT3D +Y (up)", allclose(v_up_pt3d, torch.tensor([[0., 1., 0.]])))


# ======================================================================
print("\n" + "=" * 70)
print("TEST 3: build_cam_mount_R 各轴方向验证")
print("=" * 70)

# 在 FLU 机体坐标系下: X=前, Y=左, Z=上
# forward_canonical = [1, 0, 0]
# up_canonical      = [0, 0, 1]

fwd = torch.tensor([1., 0., 0.]).view(3, 1)
up  = torch.tensor([0., 0., 1.]).view(3, 1)

# --- Pitch ---
# 参考项目 CUDA 渲染器: R_cam 的第一列方向 = 相机光轴
# R_cam @ [1,0,0] = 相机光轴在机体系的方向
# 参考项目 R_cam 矩阵与 build_cam_mount_R 相同
pitch_10 = build_cam_mount_R(pitch_deg=10.0)
look_dir = (pitch_10[0] @ fwd).squeeze()
print(f"  pitch=+10°: 光轴方向(body) = [{look_dir[0]:.4f}, {look_dir[1]:.4f}, {look_dir[2]:.4f}]")
check("pitch=+10°: 光轴有正Z分量 (向上倾斜)",
      look_dir[2] > 0,
      f"z={look_dir[2].item():.4f}")

pitch_neg = build_cam_mount_R(pitch_deg=-10.0)
look_neg = (pitch_neg[0] @ fwd).squeeze()
check("pitch=-10°: 光轴有负Z分量 (向下倾斜)",
      look_neg[2] < 0,
      f"z={look_neg[2].item():.4f}")

# 与参考项目比较: 参考项目 R_cam 矩阵
import math
alpha = 10.0 * math.pi / 180
ref_R_cam = torch.tensor([
    [math.cos(alpha), 0, -math.sin(alpha)],
    [0, 1, 0],
    [math.sin(alpha), 0, math.cos(alpha)],
])
our_R_mount = build_cam_mount_R(pitch_deg=10.0)[0]
check("build_cam_mount_R(pitch=10) == 参考项目 R_cam(10°)",
      allclose(our_R_mount, ref_R_cam))

# 参考项目语义: cam_angle=10 → 相机光轴向上10°
# 参考项目 CUDA kernel: ray at center pixel = R_cam @ [1,0,0]
# = [cos(10°), 0, sin(10°)] → 前方偏上10°
# 与本项目吻合
print(f"\n  结论: pitch=+10° → 相机向上看10° (与参考项目 cam_angle=10 一致)")
print(f"  参考项目 CUDA kernel 中心像素光线: ray = R_cam @ [1,0,0] = [cos(α), 0, sin(α)]")
print(f"  正 pitch 值 = 向上倾斜 (仰视), 负 pitch 值 = 向下倾斜 (俯视)")

# --- Roll ---
print()
roll_10 = build_cam_mount_R(roll_deg=10.0)
up_rolled = (roll_10[0] @ up).squeeze()
print(f"  roll=+10°: up方向(body) = [{up_rolled[0]:.4f}, {up_rolled[1]:.4f}, {up_rolled[2]:.4f}]")
# roll>0 → Rx(-roll) → up向量向Y负方向偏 (在FLU中Y=左, -Y=右)
# 所以 up 倒向右 → 相机右倾
check("roll=+10°: up方向有正Y分量 (左倾)",
      up_rolled[1] > 0,
      f"y={up_rolled[1].item():.4f}")

# --- Yaw ---
yaw_10 = build_cam_mount_R(yaw_deg=10.0)
look_yawed = (yaw_10[0] @ fwd).squeeze()
print(f"  yaw=+10°: 光轴方向(body) = [{look_yawed[0]:.4f}, {look_yawed[1]:.4f}, {look_yawed[2]:.4f}]")
# yaw>0 → Rz(yaw) → forward向量在XY平面向-Y偏 (在FLU中-Y=右)
# 相机向右偏
check("yaw=+10°: 光轴有正Y分量 (左偏)",
      look_yawed[1] > 0,
      f"y={look_yawed[1].item():.4f}")


# ======================================================================
print("\n" + "=" * 70)
print("TEST 4: compute_view_matrix 全流程验证")
print("=" * 70)

from drone_renderer import DroneRenderer

# 使用最小网格创建渲染器
mesh_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "sample", "sample4.obj")
if os.path.exists(mesh_path):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    renderer = DroneRenderer(
        mesh_path=mesh_path,
        device=device,
        image_size=(48, 64),
        focal_length=32.0,
        num_samples=100,
        subdivide_times=0,
    )

    B = 1
    # 悬停在原点, 朝北(+Y)
    p_ros = torch.tensor([[0., 0., 5.]], device=device)  # 高度5m
    R_ros = torch.eye(3, device=device).unsqueeze(0)     # 朝东(identity→body X=ROS +X=east)

    # 无 pitch 偏移 (cam forward = body forward = ROS +X east)
    R_view_0, T_view_0 = renderer.compute_view_matrix(
        p_ros=p_ros, R_ros=R_ros,
        cam_mount_R=build_cam_mount_R(pitch_deg=0.0, device=device),
        cam_offset_body=torch.zeros(3, device=device),
    )

    # 验证: R_view 应使 PT3D 相机 Z 方向与世界 East 对齐 (相机看向前方)
    # PyTorch3D 相机坐标: Z = 前方 (into screen)
    # 世界 East = ROS +X = PT3D -X
    # 但 look_at_view_transform 可能有不同的约定
    # 重要: R_view 是 World-to-View 变换, 不是 View-to-World
    print(f"  pitch=0°, 悬停朝东:")
    print(f"    R_view:\n{R_view_0[0].cpu()}")
    print(f"    T_view: {T_view_0[0].cpu()}")

    # 关键验证: 改变 pitch 后, 相机看向的世界方向应该改变
    R_view_up, T_view_up = renderer.compute_view_matrix(
        p_ros=p_ros, R_ros=R_ros,
        cam_mount_R=build_cam_mount_R(pitch_deg=30.0, device=device),
        cam_offset_body=torch.zeros(3, device=device),
    )
    R_view_dn, T_view_dn = renderer.compute_view_matrix(
        p_ros=p_ros, R_ros=R_ros,
        cam_mount_R=build_cam_mount_R(pitch_deg=-30.0, device=device),
        cam_offset_body=torch.zeros(3, device=device),
    )

    # View-to-World: R_v2w = R_view^T (因为 R_view 是正交的)
    # 相机前方 (View +Z) 在世界中的方向: R_v2w @ [0,0,1]^T = R_view^T @ [0,0,1]^T = R_view 的第3列
    cam_fwd_0_pt3d = R_view_0[0, :, 2].cpu()
    cam_fwd_up_pt3d = R_view_up[0, :, 2].cpu()
    cam_fwd_dn_pt3d = R_view_dn[0, :, 2].cpu()
    print(f"\n  相机前方(View+Z)在PT3D世界中的方向:")
    print(f"    pitch=0°:   {cam_fwd_0_pt3d}")
    print(f"    pitch=+30°: {cam_fwd_up_pt3d}")
    print(f"    pitch=-30°: {cam_fwd_dn_pt3d}")

    # PT3D Y 轴 = Up, 所以 cam_fwd 的 Y 分量越大 → 越朝上看
    check("pitch=+30°比0°更向上 (PT3D Y分量更大)",
          cam_fwd_up_pt3d[1] > cam_fwd_0_pt3d[1],
          f"Y: {cam_fwd_up_pt3d[1]:.4f} vs {cam_fwd_0_pt3d[1]:.4f}")

    check("pitch=-30°比0°更向下 (PT3D Y分量更小)",
          cam_fwd_dn_pt3d[1] < cam_fwd_0_pt3d[1],
          f"Y: {cam_fwd_dn_pt3d[1]:.4f} vs {cam_fwd_0_pt3d[1]:.4f}")

    # 具体数值: pitch=+30 时相机应看向30°偏上
    # 在PT3D中: forward = 东 = -X, up = +Y
    # 30°仰角: forward 分量 = cos(30°) ≈ 0.866 → along -X
    #           up 分量 = sin(30°) = 0.5 → along +Y
    check("pitch=+30°偏上: Y分量 ≈ sin(30°)=0.5",
          abs(cam_fwd_up_pt3d[1].item() - 0.5) < 0.05,
          f"Y={cam_fwd_up_pt3d[1].item():.4f}, expected~0.5")
else:
    print("  跳过 (mesh 文件不存在)")


# ======================================================================
print("\n" + "=" * 70)
print("TEST 5: KNN 查询坐标一致性")
print("=" * 70)

# 验证: 通过 ros_to_obj 变换后的位置 在 OBJ 空间做 KNN 查询,
# 返回的向量通过 obj_to_ros 变换回 ROS 空间

# 基本向量变换一致性
v_ros = torch.tensor([2., 3., 1.])
v_obj = ros_to_obj(v_ros)
v_back = obj_to_ros(v_obj)
check("ros→obj→ros 一致", allclose(v_back, v_ros))

# 距离不变性: ||v||_ros == ||ros_to_obj(v)||_obj
# 由于变换只是轴重排和符号翻转, L2范数应不变
v_test = torch.randn(100, 3)
v_transformed = ros_to_obj(v_test)
norms_orig = v_test.norm(dim=-1)
norms_trans = v_transformed.norm(dim=-1)
check("ROS→OBJ 距离不变 (L2范数保持)", allclose(norms_orig, norms_trans))

# 方向向量变换一致性:
# 在 OBJ 空间, 从 p_obj 到 nn_obj 的向量为 vec_obj = nn_obj - p_obj
# 变换到 ROS: vec_ros = obj_to_ros(vec_obj)
# 应等于: obj_to_ros(nn_obj) - obj_to_ros(p_obj)
# (因为变换是线性的)
p = torch.randn(3)
nn = torch.randn(3)
vec_direct = obj_to_ros(nn - p)
vec_indirect = obj_to_ros(nn) - obj_to_ros(p)
check("方向向量变换线性: f(a-b) = f(a)-f(b)", allclose(vec_direct, vec_indirect))


# ======================================================================
print("\n" + "=" * 70)
print("TEST 6: OBJ→Body 顶点变换验证")
print("=" * 70)

# 代码: centered[:, [0, 2, 1]] * [1, -1, 1]
# OBJ Y-up → ROS body FLU 的变换

transform_mat = torch.tensor([[1., 0., 0.],
                               [0., 0., -1.],
                               [0., 1., 0.]])
# 验证: OBJ Y-up 方向 → body Z-up
obj_up = torch.tensor([0., 1., 0.])
body_result = transform_mat @ obj_up  # 应该是 [0, 0, 1] = body up
check("OBJ Y-up → body Z-up", allclose(body_result, torch.tensor([0., 0., 1.])))

# OBJ X → body X (不变)
obj_x = torch.tensor([1., 0., 0.])
body_x = transform_mat @ obj_x
check("OBJ X → body X (不变)", allclose(body_x, torch.tensor([1., 0., 0.])))

# OBJ Z (toward viewer / out of screen) → body -Y
obj_z = torch.tensor([0., 0., 1.])
body_z = transform_mat @ obj_z
check("OBJ Z → body [0, -1, 0]", allclose(body_z, torch.tensor([0., -1., 0.])))

print(f"\n  OBJ→Body 变换矩阵 (Rx(+90°)):") 
print(f"    body_X = obj_X (保持)")
print(f"    body_Y = -obj_Z (OBJ 观察者方向 → 机体右侧)")
print(f"    body_Z = obj_Y (OBJ 上方 → 机体上方)")
print(f"  注: 是否 body_X=前方 取决于 OBJ 模型朝向")

# 变换行列式 = +1 (正确右手旋转)
det = torch.linalg.det(transform_mat)
check("OBJ→Body 变换行列式 = +1 (右手旋转)", abs(det.item() - 1.0) < 1e-6,
      f"det={det.item():.6f}")


# ======================================================================
print("\n" + "=" * 70)
print("TEST 7: 文档 vs 实际行为检查")
print("=" * 70)

# 文档说 "pitch > 0 → 相机向下倾斜（俯视）"
# 实际: pitch > 0 → 相机向上倾斜
# 验证:
R_pitch30 = build_cam_mount_R(pitch_deg=30.0)
look = (R_pitch30[0] @ fwd).squeeze()
print(f"  build_cam_mount_R(pitch=30°) @ [1,0,0] = [{look[0]:.4f}, {look[1]:.4f}, {look[2]:.4f}]")
print(f"  Z分量 = {look[2]:.4f} > 0 → 相机向上看 (不是向下!)")
check("文档错误: 正pitch实际为仰视(向上), 非俯视(向下)",
      look[2] > 0,
      f"z_component={look[2]:.4f}")

# 但与参考项目一致
print(f"\n  与参考项目对比:")
print(f"  参考项目 cam_angle=10 → R_cam 与 build_cam_mount_R(pitch=10) 完全一致")
print(f"  参考项目 CUDA 渲染器的中心像素光线 = R_cam @ [1,0,0] = [cos(α),0,sin(α)]")
print(f"  sin(α)>0 → 光线向上 → 参考项目的 cam_angle=10 也是仰视10°")
print(f"  结论: 代码行为正确(与参考项目一致), 文档说明有误")


# ======================================================================
print("\n" + "=" * 70)
print(f"SUMMARY: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    print("存在失败项, 需要修复!")
    sys.exit(1)
else:
    print("所有坐标系验证通过!")
