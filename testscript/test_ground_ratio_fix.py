"""
验证 ground_ratio 修复: 对比 ground_ratio=0.3 (BUG) vs ground_ratio=0.6 (修复) 下的SR
加载旧 checkpoint, 分别用两种 ground_ratio 跑 5 个 episode, 对比SR
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

def run_test(ground_ratio, ckpt_path, config_path, gpu, n_episodes=5):
    """用指定 ground_ratio 跑n个episode, 返回平均SR"""
    from train import DroneTrainer, _ArgParser
    
    # 构造命令行参数
    parser = _ArgParser(fromfile_prefix_chars='@')
    # 先从 train.py 的 parse_args 获取所有参数定义
    from train import parse_args as _parse
    sys.argv = [
        'train.py',
        f'@{config_path}',
        '--save_dir', '/tmp/test_gr_fix',
        '--gpu', str(gpu),
        '--resume', ckpt_path,
        f'--ground_ratio', str(ground_ratio),
        '--num_iters', '1',
        '--batch_size', '16',  # 缩小batch以适配RTX 3060
    ]
    args = _parse()
    
    trainer = DroneTrainer(args)
    
    srs = []
    for ep in range(n_episodes):
        with torch.no_grad():
            _, metrics, _, extra = trainer.run_episode(iteration=0)
        sr = metrics['success_rate']
        srs.append(sr)
        print(f"  Episode {ep+1}: SR={sr:.2%}")
    
    avg_sr = sum(srs) / len(srs)
    print(f"  -> 平均SR = {avg_sr:.2%} (ground_ratio={ground_ratio})")
    
    # Cleanup
    del trainer
    torch.cuda.empty_cache()
    
    return avg_sr


if __name__ == '__main__':
    GPU = 1  # RTX 3060
    
    # 使用单机checkpoint (参数偏移更干净: 仅ground_ratio变化)
    ckpt = 'checkpoints/single_run_20260322/checkpoint_final.pth'
    config = 'configs/single_agent.args'
    
    if not os.path.exists(ckpt):
        print(f"Checkpoint 不存在: {ckpt}")
        sys.exit(1)
    
    print(f"=" * 60)
    print(f"对照实验: ground_ratio=0.3 (BUG) vs 0.6 (修复)")
    print(f"Checkpoint: {ckpt}")
    print(f"=" * 60)
    
    print(f"\n--- ground_ratio=0.3 (BUG: 70%悬浮障碍物) ---")
    sr_bug = run_test(0.3, ckpt, config, GPU, n_episodes=5)
    
    print(f"\n--- ground_ratio=0.6 (修复: 40%悬浮, 60%接地) ---")
    sr_fix = run_test(0.6, ckpt, config, GPU, n_episodes=5)
    
    print(f"\n{'='*60}")
    print(f"结果对比:")
    print(f"  ground_ratio=0.3 (BUG):  SR = {sr_bug:.2%}")
    print(f"  ground_ratio=0.6 (修复): SR = {sr_fix:.2%}")
    print(f"  差值: {sr_fix - sr_bug:+.2%}")
    print(f"{'='*60}")
