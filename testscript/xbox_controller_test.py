#!/usr/bin/env python3
"""
Xbox 手柄手动控制无人机测试脚本

操控映射:
  左摇杆 X/Y    → 水平推力 (前后/左右)
  右摇杆 Y      → 垂直推力 (上升/下降)
  右摇杆 X      → 偏航 (暂留)
  LT / RT       → 减速 / 加速 (推力增益)
  A 按钮        → 重置位置
  B 按钮        → 随机化场景
  X 按钮        → 切换第三人称/第一人称视角
  START          → 退出

运行:
  python testscript/xbox_controller_test.py [--device cuda:0] [--image_size 480 640]
"""

import sys
import os
import argparse
import time

import numpy as np
import torch
import cv2

# 确保项目根目录在 path 中
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from drone_env import DroneSimulator

# ========== pygame 手柄初始化 ==========
try:
    import pygame
except ImportError:
    print("需要安装 pygame：pip install pygame")
    sys.exit(1)


def init_joystick():
    """初始化第一个连接的手柄，返回 pygame.joystick.Joystick 或 None。"""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[警告] 未检测到手柄，使用键盘 fallback (WASD + QE + Space/LShift)")
        return None
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[手柄] 已连接: {js.get_name()}")
    return js


def deadzone(val, threshold=0.15):
    """应用死区过滤。"""
    return val if abs(val) > threshold else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Xbox 手柄手动控制无人机")
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--image_size', type=int, nargs=2, default=[480, 640])
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--thrust_scale', type=float, default=3.0,
                   help='摇杆→推力换算系数')
    p.add_argument('--mesh_path', type=str, default='data/sample/sample.obj')
    p.add_argument('--focal_length', type=float, default=500.0)
    p.add_argument('--drone_mesh', type=str, default=None,
                   help='无人机网格路径 (可选)')
    return p.parse_args()


# ========== 主循环 ==========
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 初始化手柄
    joystick = init_joystick()

    # 创建仿真环境 (batch_size=1 用于单机手动控制)
    print("[初始化] 创建仿真环境...")
    env = DroneSimulator(
        batch_size=1,
        dt=args.dt,
        device=device,
        mesh_path=args.mesh_path,
        image_size=tuple(args.image_size),
        focal_length=args.focal_length,
        enable_random_scene=True,
        drone_mesh_path=args.drone_mesh,
        noise_std=0.0,     # 手动控制关闭噪声
        grad_decay=1.0,    # 不衰减
    )
    env.safe_reset()
    print(f"[初始化] 完成 | 位置: {env.p[0].cpu().tolist()}")

    # 渲染一次确认
    rgb, depth = env.render(return_tensor=False)
    print(f"[渲染] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

    # 显示窗口
    win_name = "Drone Manual Control (Xbox / Keyboard)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, args.image_size[1], args.image_size[0])

    # 控制状态
    third_person = False
    running = True
    frame_count = 0
    fps_timer = time.time()
    fps_display = 0.0

    # 键盘 fallback 状态
    key_state = {'w': False, 's': False, 'a': False, 'd': False,
                 'q': False, 'e': False, 'space': False, 'lshift': False}

    print("\n=== 控制说明 ===")
    if joystick:
        print("左摇杆: 前后/左右  |  右摇杆Y: 上升/下降")
        print("LT: 减速  RT: 加速  |  A: 重置  B: 换场景  X: 切视角  START: 退出")
    print("键盘: WASD=水平  Space/LShift=升降  Q/E=偏航  R=重置  N=换场景  ESC=退出")
    print("================\n")

    while running:
        # ---- 1. 读取输入 ----
        ax_forward = 0.0   # 前后 (正=前)
        ax_lateral = 0.0   # 左右 (正=右)
        ax_vertical = 0.0  # 上下 (正=上)
        thrust_gain = 1.0
        reset_flag = False
        scene_flag = False

        # pygame 事件泵
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:  # A
                    reset_flag = True
                elif event.button == 1:  # B
                    scene_flag = True
                elif event.button == 2:  # X
                    third_person = not third_person
                    print(f"[视角] {'第三人称' if third_person else '第一人称'}")
                elif event.button == 7:  # START
                    running = False

        # OpenCV 键盘 (1ms 等待)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            running = False
        elif key == ord('r'):
            reset_flag = True
        elif key == ord('n'):
            scene_flag = True
        elif key == ord('x'):
            third_person = not third_person
            print(f"[视角] {'第三人称' if third_person else '第一人称'}")

        # 手柄轴读取
        if joystick:
            ax_lateral = deadzone(joystick.get_axis(0))    # 左摇杆 X
            ax_forward = -deadzone(joystick.get_axis(1))   # 左摇杆 Y (上为负)
            ax_vertical = -deadzone(joystick.get_axis(3))  # 右摇杆 Y
            # LT / RT (axis 4 & 5 在 Linux Xbox 上常为 -1~1)
            lt = (joystick.get_axis(4) + 1.0) / 2.0 if joystick.get_numaxes() > 4 else 0.0
            rt = (joystick.get_axis(5) + 1.0) / 2.0 if joystick.get_numaxes() > 5 else 0.0
            thrust_gain = 1.0 - 0.5 * lt + 0.5 * rt  # LT 减速，RT 加速
        else:
            # 键盘 fallback: 检测按键按住
            # 注: cv2.waitKey 不支持持续按住检测，用 pygame 键盘代替
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]: ax_forward = 1.0
            if keys[pygame.K_s]: ax_forward = -1.0
            if keys[pygame.K_a]: ax_lateral = -1.0
            if keys[pygame.K_d]: ax_lateral = 1.0
            if keys[pygame.K_SPACE]: ax_vertical = 1.0
            if keys[pygame.K_LSHIFT]: ax_vertical = -1.0

        # ---- 2. 处理重置 ----
        if reset_flag:
            env.safe_reset()
            print(f"[重置] 位置: {env.p[0].cpu().tolist()}")
            continue
        if scene_flag:
            env.randomize_scene()
            env.safe_reset()
            print("[场景] 已随机化并重置")
            continue

        # ---- 3. 将摇杆输入转为机体加速度指令 ----
        # ENU 坐标系: X=东, Y=北, Z=上
        # 利用无人机当前朝向 (R 矩阵) 将机体前/右映射到世界坐标
        R_body = env.R[0]  # (3, 3), 列 = 机体轴在世界系中的方向
        body_x = R_body[:, 0]  # 机体前方 (世界系)
        body_y = R_body[:, 1]  # 机体左方 (世界系)

        # 将前后/左右映射到水平面
        forward_proj = body_x.clone()
        forward_proj[2] = 0.0
        norm = forward_proj.norm()
        if norm > 1e-6:
            forward_proj = forward_proj / norm

        left_proj = body_y.clone()
        left_proj[2] = 0.0
        norm = left_proj.norm()
        if norm > 1e-6:
            left_proj = left_proj / norm

        scale = args.thrust_scale * thrust_gain
        # 推力指令 = 悬停 (g) + 操控增量
        g = env.g_std[0]  # (3,), 标准重力
        act_delta = (ax_forward * forward_proj + ax_lateral * (-left_proj) + 
                     ax_vertical * torch.tensor([0.0, 0.0, 1.0], device=device))
        act_cmd = g + act_delta * scale  # (3,)
        act_cmd = act_cmd.unsqueeze(0)   # (1, 3)

        # ---- 4. 仿真步进 ----
        env.step(act_cmd)

        # ---- 5. 渲染 ----
        rgb, depth = env.render(return_tensor=False)

        # 取第一个 batch
        if rgb.ndim == 4:
            rgb_frame = rgb[0]
        else:
            rgb_frame = rgb

        # BGR for OpenCV
        bgr = cv2.cvtColor((rgb_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # ---- 6. HUD 叠加 ----
        pos = env.p[0].detach().cpu().numpy()
        vel = env.v[0].detach().cpu().numpy()
        spd = np.linalg.norm(vel)
        alt = pos[2]

        # FPS 计算
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed > 0.5:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        hud_lines = [
            f"FPS: {fps_display:.0f}",
            f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"Alt: {alt:.2f}m  Spd: {spd:.2f}m/s",
            f"Gain: {thrust_gain:.1f}x",
            f"View: {'3rd' if third_person else '1st'}",
        ]
        y0 = 25
        for i, line in enumerate(hud_lines):
            cv2.putText(bgr, line, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win_name, bgr)

    # 清理
    cv2.destroyAllWindows()
    pygame.quit()
    print("[退出] 手动控制结束")


if __name__ == '__main__':
    main()
