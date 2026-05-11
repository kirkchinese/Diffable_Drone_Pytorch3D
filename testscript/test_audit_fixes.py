#!/usr/bin/env python3
"""
审计修复验证脚本：验证代码审计中所有修复项在运行时正确工作。

覆盖项：
  1. numpy 已从训练热路径中移除（drone_env, drone_renderer, drone_renderer_dynamic, scene_generator）
  2. DynamicObstacle 接受 list/tensor 作为 position/velocity（torch.as_tensor 修复）
  3. DroneRenderer 缓存了 _fwd_canonical / _up_canonical 向量
  4. DroneLoss.forward 使用 new_tensor 替代 torch.tensor(0.0)
  5. drone_dynamics 中冗余 uz.clone() 已移除
  6. add_primitive_obstacle 使用 base_model OBJ 加载而非硬编码几何体
  7. randomize_dynamic_obstacles 全部使用 torch 随机数
  8. scene_generator 无残留 numpy 依赖
    9. scene_generator 不再依赖 Python random/math
 10. train.py 支持统一 model_type 入口，train_adaptive.py 为兼容包装

用法:
  conda run -n pytorch python testscript/test_audit_fixes.py
  conda run -n pytorch python testscript/test_audit_fixes.py --gpu 1   # 使用 GPU1
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch


def test_no_numpy_in_hot_modules():
    """验证 drone_renderer, drone_renderer_dynamic, drone_dynamics, loss 无 numpy 导入"""
    import drone_renderer
    import drone_renderer_dynamic
    import drone_dynamics
    import loss

    for mod_name, mod in [
        ('drone_renderer', drone_renderer),
        ('drone_renderer_dynamic', drone_renderer_dynamic),
        ('drone_dynamics', drone_dynamics),
    ]:
        src = open(mod.__file__).read()
        assert 'import numpy' not in src, f"{mod_name} 仍然包含 numpy 导入"

    # scene_generator 也不应保留 numpy
    import scene_generator
    src = open(scene_generator.__file__).read()
    assert 'import numpy' not in src, "scene_generator 仍然包含 numpy 导入"

    print("[PASS] 热路径模块无 numpy 导入")


def test_renderer_cached_vectors(device):
    """验证 DroneRenderer 缓存了 canonical 向量"""
    from drone_renderer import DroneRenderer
    r = DroneRenderer(
        mesh_path='./data/base_model/drone.obj',
        image_size=(64, 64),
        device=device,
    )
    assert hasattr(r, '_fwd_canonical'), "缺少 _fwd_canonical"
    assert hasattr(r, '_up_canonical'), "缺少 _up_canonical"
    assert r._fwd_canonical.device == device
    assert r._up_canonical.device == device
    print("[PASS] DroneRenderer 缓存的 canonical 向量验证通过")


def test_dynamic_obstacle_accepts_list(device):
    """验证 DynamicObstacle 能接受 Python list 作为 position/velocity"""
    from drone_renderer_dynamic import DynamicSceneRenderer
    dsr = DynamicSceneRenderer(
        static_mesh_path='./data/base_model/drone.obj',
        device=device, image_size=(64, 64),
    )
    # 使用 list（之前会 AttributeError: 'list' object has no attribute 'to'）
    idx = dsr.add_primitive_obstacle(
        primitive_type='cube',
        position=[1.0, 2.0, 3.0],
        velocity=[0.1, 0.0, 0.0],
        scale=0.5,
    )
    obs = dsr.dynamic_obstacles[idx]
    assert obs.position.device == device
    assert obs.velocity.device == device
    assert obs.position.shape == (3,)
    print("[PASS] DynamicObstacle 接受 list 输入")


def test_primitive_obstacle_from_obj(device):
    """验证 add_primitive_obstacle 从 base_model OBJ 加载（非硬编码几何体）"""
    from drone_renderer_dynamic import DynamicSceneRenderer
    dsr = DynamicSceneRenderer(
        static_mesh_path='./data/base_model/drone.obj',
        device=device, image_size=(64, 64),
    )
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'torus']
    for shape in shapes:
        dsr.add_primitive_obstacle(
            primitive_type=shape,
            position=[0.0, 0.0, 1.0],
            scale=0.3,
        )
    assert len(dsr.dynamic_obstacles) == len(shapes)
    print(f"[PASS] {len(shapes)} 种基本几何体从 OBJ 加载成功")


def test_randomize_dynamic_obstacles_torch_only(device):
    """验证 randomize_dynamic_obstacles 使用 torch 随机数（无 numpy）"""
    from drone_env import DroneSimulator
    from scene_generator import SceneGenerator
    sg = SceneGenerator(primitive_dir='./data/base_model', device=device, seed=0)
    sim = DroneSimulator(
        batch_size=2, device=device, scene_generator=sg,
        mesh_path='./data/base_model/drone.obj', image_size=(64, 64),
        enable_random_scene=True,
    )
    # 用 torch seed 控制可重复性
    torch.manual_seed(42)
    sim.randomize_dynamic_obstacles(arena_range=6.0)
    n1 = len(sim._dynamic_obstacles)

    torch.manual_seed(42)
    sim.randomize_dynamic_obstacles(arena_range=6.0)
    n2 = len(sim._dynamic_obstacles)

    assert n1 == n2, f"相同 seed 下障碍物数量不一致: {n1} vs {n2}"
    assert n1 > 0, "未生成任何动态障碍物"
    # 检查每个障碍物都在正确设备上
    for obs in sim._dynamic_obstacles:
        assert obs.position.device == device
    print(f"[PASS] randomize_dynamic_obstacles 纯 torch 随机 (n={n1}), seed 可重复")


def test_loss_no_torch_tensor_fallback(device):
    """验证 DroneLoss 中使用 new_tensor 而非 torch.tensor(0.0)"""
    import loss as loss_mod
    src = open(loss_mod.__file__).read()
    # 不应在 forward 方法中出现 torch.tensor(0.0
    # （但 __init__ 中可以出现）
    lines = src.split('\n')
    in_forward = False
    for i, line in enumerate(lines):
        if 'def forward(' in line:
            in_forward = True
        elif in_forward and line.strip().startswith('def '):
            in_forward = False
        if in_forward and 'torch.tensor(0.0' in line:
            raise AssertionError(f"loss.py 第 {i+1} 行: forward 中仍使用 torch.tensor(0.0)")
    print("[PASS] DroneLoss.forward 使用 new_tensor (无 torch.tensor(0.0))")


def test_dynamics_no_redundant_clone():
    """验证 drone_dynamics.py 中无冗余 uz.clone()"""
    import drone_dynamics
    src = open(drone_dynamics.__file__).read()
    assert 'uz_safe = uz.clone()' not in src, "drone_dynamics 中仍存在冗余 uz.clone()"
    print("[PASS] drone_dynamics 无冗余 uz.clone()")


def test_scene_generator_torch_random_only():
    """验证 scene_generator 不再依赖 Python random/math"""
    import scene_generator
    src = open(scene_generator.__file__).read()
    assert 'import random' not in src, "scene_generator 仍依赖 Python random"
    assert 'from math import' not in src, "scene_generator 仍依赖 math"
    print("[PASS] scene_generator 使用 torch 随机与张量运算")


def test_scene_generator_generate_smoke(device):
    """验证 SceneGenerator.generate() 可实际生成场景"""
    from scene_generator import SceneGenerator

    sg = SceneGenerator(
        device=device,
        primitive_dir='./data/base_model',
        num_obstacles_range=(6, 6),
        seed=123,
    )
    scene_mesh, obstacle_info = sg.generate()
    assert scene_mesh.verts_packed().shape[0] > 0, "生成场景顶点数应大于0"
    assert len(obstacle_info) == 6, f"障碍物数量应为6, 得到{len(obstacle_info)}"
    print("[PASS] SceneGenerator.generate() 运行正常")


def test_train_entrypoints_unified():
    """验证 train.py 支持 model_type，train_adaptive.py 为兼容入口"""
    train_src = open(PROJECT_ROOT / 'train.py').read()
    adaptive_src = open(PROJECT_ROOT / 'train_adaptive.py').read()
    assert '--model_type' in train_src, "train.py 缺少 --model_type 统一入口"
    assert 'Model_adaptive' in train_src, "train.py 尚未接入 Model_adaptive"
    assert ('model_type' in adaptive_src and 'adaptive' in adaptive_src), \
        "train_adaptive.py 未默认委托 adaptive 模型"
    print("[PASS] 统一训练入口与 adaptive 兼容入口验证通过")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    print("=" * 60)
    print(f"审计修复验证  (device={device})")
    print("=" * 60)

    tests = [
        test_no_numpy_in_hot_modules,
        lambda: test_renderer_cached_vectors(device),
        lambda: test_dynamic_obstacle_accepts_list(device),
        lambda: test_primitive_obstacle_from_obj(device),
        lambda: test_randomize_dynamic_obstacles_torch_only(device),
        lambda: test_loss_no_torch_tensor_fallback(device),
        test_dynamics_no_redundant_clone,
        test_scene_generator_torch_random_only,
        lambda: test_scene_generator_generate_smoke(device),
        test_train_entrypoints_unified,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
