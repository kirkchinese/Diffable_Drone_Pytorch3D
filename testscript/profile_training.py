"""
训练循环性能剖析 — 使用 torch.profiler 和手动计时。在 gpu1 (RTX 3060) 上运行。
直接调用 trainer.run_episode() 避免手动复现训练循环。
Usage: conda run -n pytorch python testscript/profile_training.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def main():
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    sys.argv = [
        'train.py',
        '--batch_size', '32',
        '--num_iters', '6',
        '--timesteps', '100',
        '--lr', '1e-3',
        '--grad_decay', '0.4',
        '--ctl_dt', '0.0667',
        '--enable_dynamic_obstacles',
        '--reach_radius', '0.1',
        '--random_scene',
        '--safe_spawn',
        '--safe_clearance', '1.0',
        '--force_cross_map',
        '--spawn_z_max', '3.0',
        '--arena_range', '20.0',
        '--init_p_range', '20.0',
        '--num_obstacles_min', '35',
        '--num_obstacles_max', '50',
        '--obstacle_scale_min', '0.3',
        '--obstacle_scale_max', '1.5',
        '--ground_ratio', '0.6',
        '--cluster_ratio', '0.3',
        '--image_height', '48',
        '--image_width', '64',
        '--hfov', '90',
        '--depth_min', '0.3',
        '--depth_max', '24.0',
        '--num_samples', '50000',
        '--subdivide_times', '0',
        '--coef_v', '1.0',
        '--coef_v_pred', '2.0',
        '--coef_collide', '3.0',
        '--coef_obj_avoidance', '2.0',
        '--coef_d_acc', '0.01',
        '--coef_d_jerk', '0.001',
        '--coef_ground_affinity', '0.1',
        '--coef_lateral', '0.0',
        '--window_size', '30',
        '--margin_min', '0.3',
        '--margin_max', '0.8',
        '--noise_std', '0.04',
        '--cam_angle', '10',
        '--cam_rand_rpy', '2.0',
        '--cam_rand_xy', '0.02',
        '--random_init_yaw',
        '--n_drones_per_group', '8',
        '--save_dir', '/tmp/profile_test',
        '--save_freq', '99999',
        '--gpu', '1',
        '--loss_v_mode', 'adaptive',
        '--adaptive_decay_rate', '2',
    ]

    from train import parse_args, DroneTrainer
    args = parse_args()

    print("初始化 Trainer ...")
    trainer = DroneTrainer(args)

    # 预热
    print("预热 2 个 episode ...")
    for i in range(2):
        loss, metrics, _, _ = trainer.run_episode(iteration=99999)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        print(f"  warmup {i}: loss={loss.item():.3f}")
    torch.cuda.synchronize(device)

    # === 手动计时: 整体 episode ===
    NUM_EP = 4
    ep_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []

    print(f"\n{'='*60}")
    print(f"手动计时: {NUM_EP} episodes, batch={args.batch_size}, "
          f"timesteps={args.timesteps}")
    print(f"{'='*60}\n")

    for ep in range(NUM_EP):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        loss, metrics, _, _ = trainer.run_episode(iteration=99999)

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        trainer.optimizer.zero_grad()
        loss.backward()

        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optimizer.step()

        torch.cuda.synchronize(device)
        t3 = time.perf_counter()

        fwd_ms = (t1 - t0) * 1000
        bwd_ms = (t2 - t1) * 1000
        opt_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        opt_times.append(opt_ms)
        ep_times.append(total_ms)

        print(f"  ep {ep}: total={total_ms:.0f}ms  "
              f"(fwd={fwd_ms:.0f} bwd={bwd_ms:.0f} opt={opt_ms:.0f})  "
              f"loss={loss.item():.3f}")

    avg_total = sum(ep_times) / len(ep_times)
    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    avg_opt = sum(opt_times) / len(opt_times)

    print(f"\n{'='*60}")
    print(f"平均 episode 耗时:")
    print(f"  forward (run_episode): {avg_fwd:8.1f} ms  ({avg_fwd/avg_total*100:.1f}%)")
    print(f"  backward:              {avg_bwd:8.1f} ms  ({avg_bwd/avg_total*100:.1f}%)")
    print(f"  optimizer step:        {avg_opt:8.1f} ms  ({avg_opt/avg_total*100:.1f}%)")
    print(f"  total:                 {avg_total:8.1f} ms")
    print(f"  吞吐量: {1000/avg_total:.2f} ep/s")

    # === 内循环分段计时 (monkey-patch) ===
    print(f"\n{'='*60}")
    print(f"内循环分段计时 (render / knn / step / policy):")
    print(f"{'='*60}\n")

    import functools

    class TimingAccumulator:
        def __init__(self):
            self.data = {}
        def reset(self):
            self.data = {}
        def add(self, key, ms):
            self.data.setdefault(key, []).append(ms)

    timer = TimingAccumulator()

    # Patch env.render
    orig_render = trainer.env.render
    @functools.wraps(orig_render)
    def timed_render(*a, **kw):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        r = orig_render(*a, **kw)
        torch.cuda.synchronize(device)
        timer.add('render', (time.perf_counter() - t) * 1000)
        return r
    trainer.env.render = timed_render

    # Patch env.combined_vec_to_nearest
    orig_knn = trainer.env.combined_vec_to_nearest
    @functools.wraps(orig_knn)
    def timed_knn(*a, **kw):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        r = orig_knn(*a, **kw)
        torch.cuda.synchronize(device)
        timer.add('knn', (time.perf_counter() - t) * 1000)
        return r
    trainer.env.combined_vec_to_nearest = timed_knn

    # Patch env.step
    orig_step = trainer.env.step
    @functools.wraps(orig_step)
    def timed_step(*a, **kw):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        r = orig_step(*a, **kw)
        torch.cuda.synchronize(device)
        timer.add('env_step', (time.perf_counter() - t) * 1000)
        return r
    trainer.env.step = timed_step

    # Patch policy.infer
    orig_infer = trainer.policy.infer
    @functools.wraps(orig_infer)
    def timed_infer(*a, **kw):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        r = orig_infer(*a, **kw)
        torch.cuda.synchronize(device)
        timer.add('policy_infer', (time.perf_counter() - t) * 1000)
        return r
    trainer.policy.infer = timed_infer

    # Patch env.inter_drone_distances
    orig_dd = trainer.env.inter_drone_distances
    @functools.wraps(orig_dd)
    def timed_dd(*a, **kw):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        r = orig_dd(*a, **kw)
        torch.cuda.synchronize(device)
        timer.add('drone_dist', (time.perf_counter() - t) * 1000)
        return r
    trainer.env.inter_drone_distances = timed_dd

    detail_eps = 2
    for ep in range(detail_eps):
        timer.reset()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss, metrics, _, _ = trainer.run_episode(iteration=99999)
        torch.cuda.synchronize(device)
        fwd_ms = (time.perf_counter() - t0) * 1000

        # backward (for completeness)
        trainer.optimizer.zero_grad()
        torch.cuda.synchronize(device)
        tb = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        bwd_ms = (time.perf_counter() - tb) * 1000

        print(f"Episode {ep}: fwd={fwd_ms:.0f}ms, bwd={bwd_ms:.0f}ms")
        measured_total = 0
        for key in ['render', 'knn', 'drone_dist', 'env_step', 'policy_infer']:
            vals = timer.data.get(key, [])
            total_ms = sum(vals)
            count = len(vals)
            per_call = total_ms / count if count else 0
            pct = total_ms / fwd_ms * 100 if fwd_ms > 0 else 0
            measured_total += total_ms
            print(f"  {key:20s}: {total_ms:8.1f}ms total  "
                  f"({count} calls, {per_call:.2f}ms/call)  [{pct:.1f}%]")
        overhead_ms = fwd_ms - measured_total
        overhead_pct = overhead_ms / fwd_ms * 100 if fwd_ms > 0 else 0
        print(f"  {'python_overhead':20s}: {overhead_ms:8.1f}ms  [{overhead_pct:.1f}%]")

    # === GPU 内存使用 ===
    print(f"\n{'='*60}")
    print(f"GPU 内存使用:")
    print(f"{'='*60}")
    alloc = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    print(f"  Allocated: {alloc:.0f} MB")
    print(f"  Reserved:  {reserved:.0f} MB")
    print(f"\n建议: 如果 Reserved 远小于 GPU 显存, 可尝试增大 batch_size")


if __name__ == '__main__':
    main()
