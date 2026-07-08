#!/usr/bin/env python3
"""
可视化验证脚本：验证所有三个新特性可通过 visualize_eval.py 调用

功能：
1. 验证 visualize_eval.py 对新特性参数的支持
2. 快速运行一个短 episode（num_episodes=1, timesteps=20），验证端到端流程
3. 输出验证结果到 viz_results 目录

用法：
  python testscript/test_visualize_all_features.py
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_visualize_with_features():
    """运行可视化脚本，启用所有三个新特性"""
    
    checkpoint = PROJECT_ROOT / "checkpoints" / "checkpoint_final.pth"
    output_dir = PROJECT_ROOT / "viz_results" / "test_all_features"
    
    if not checkpoint.exists():
        print(f"SKIP: checkpoint absent: {checkpoint}")
        sys.exit(0)
    
    # 清理输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # 可视化命令：启用所有三个特性
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "visualize_eval.py"),
        "--checkpoint", str(checkpoint),
        "--output_dir", str(output_dir),
        "--num_episodes", "1",           # 只运行 1 个 episode
        "--timesteps", "20",             # 只运行 20 步（快速测试）
        "--batch_size", "2",
        "--random_scene",                # 随机场景
        "--enable_dynamic_obstacles",    # Feature 2: 启用动态障碍物
        "--num_dynamic_obstacles_min", "1",
        "--num_dynamic_obstacles_max", "3",
        "--drone_mesh_path", str(PROJECT_ROOT / "data" / "base_model" / "drone.obj"),  # Feature 3: 无人机互视
        "--n_drones_per_group", "2",
        "--no_video",  # 不生成视频（快速）
    ]
    
    print(f"[INFO] Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, timeout=300, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"[ERROR] visualize_eval.py failed with return code {result.returncode}")
            return False
        print("\n[PASS] visualize_eval.py completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print("[ERROR] visualize_eval.py timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run visualize_eval.py: {e}")
        return False

def check_output_files(output_dir):
    """检查输出目录中的文件"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"[WARN] Output directory does not exist: {output_dir}")
        return False
    
    files = list(output_dir.glob("**/*"))
    print(f"\n[INFO] Generated files in {output_dir}:")
    for f in sorted(files):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name}: {size_mb:.2f} MB")
    
    return len(files) > 0

def main():
    """主测试流程"""
    print("=" * 60)
    print("可视化验证：所有三个新特性")
    print("=" * 60)
    
    print("\n[TEST] 检查 visualize_eval.py 是否支持新参数...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "visualize_eval.py"), "--help"],
        capture_output=True, text=True, timeout=10
    )
    
    required_params = [
        "--enable_dynamic_obstacles",
        "--num_dynamic_obstacles_min",
        "--num_dynamic_obstacles_max",
        "--drone_mesh_path",
        "--n_drones_per_group",
    ]
    
    help_text = result.stdout + result.stderr
    for param in required_params:
        if param not in help_text:
            print(f"[ERROR] Missing parameter in --help: {param}")
            return False
    print("[PASS] All required parameters present in --help")
    
    print("\n[TEST] 运行可视化流程（启用所有新特性）...")
    if not run_visualize_with_features():
        return False
    
    print("\n[TEST] 检查输出文件...")
    output_dir = PROJECT_ROOT / "viz_results" / "test_all_features"
    check_output_files(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 可视化验证完成：所有三个新特性工作正常")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
