"""
诊断脚本：验证各损失项的量级，分析 SR 下降根因。

重点检查：
1. BUG: collision_history 形状 (T,S,B) 传入 compute_navigation_metrics_torch 时维度处理错误
2. 各损失项的量级和加权占比  
3. 初始帧的 inter_drone 距离是否为负（即"出生即碰撞"）
4. loss_drone_collide 的实际量级
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("TEST 1: collision_history 维度 BUG 验证")
print("=" * 70)

# 模拟: T=200步, S=5子步, B=64无人机
T, S, B = 200, 5, 64
# 假设 10% 的(时步,子步,无人机)组合发生碰撞
collision_history_3d = torch.rand(T, S, B, device=device) < 0.10

print(f"collision_history shape: {collision_history_3d.shape}")
print(f"碰撞率: {collision_history_3d.float().mean():.3f}")

# 当前代码 (BUG): .any(dim=0) on (T, S, B) → (S, B)
collision_free_buggy = ~collision_history_3d.any(dim=0)  # (S, B)
print(f"\n[BUG] collision_free shape: {collision_free_buggy.shape}  ← 应该是 ({B},) 但实际是 ({S}, {B})")

# 假设 reached_target 是 (B,)
reached_target = torch.ones(B, device=device, dtype=torch.bool)
success_buggy = collision_free_buggy & reached_target  # broadcasts to (S, B)
sr_buggy = success_buggy.float().mean().item()
print(f"[BUG] SR (inflated): {sr_buggy:.3f}")

# 正确实现: flatten(0,1) then .any(dim=0)
collision_free_correct = ~collision_history_3d.flatten(0, 1).any(dim=0)  # (B,)
success_correct = collision_free_correct & reached_target  # (B,)
sr_correct = success_correct.float().mean().item()
print(f"[CORRECT] collision_free shape: {collision_free_correct.shape}")
print(f"[CORRECT] SR: {sr_correct:.3f}")
print(f"差异: SR_buggy - SR_correct = {sr_buggy - sr_correct:.3f}")

# 极端情况：只有一个子步碰撞
collision_only_substep3 = torch.zeros(T, S, B, device=device, dtype=torch.bool)
collision_only_substep3[50, 3, :32] = True  # 50% 无人机在t=50,s=3碰撞一次

cf_bug = ~collision_only_substep3.any(dim=0)  # (S, B) — s=3 有 32 个 False，其余全 True
cf_correct = ~collision_only_substep3.flatten(0, 1).any(dim=0)  # (B,) — 那 32 架全 False
sr_bug = (cf_bug & reached_target).float().mean().item()
sr_correct2 = (cf_correct & reached_target).float().mean().item()
print(f"\n极端情况（50%无人机仅1次碰撞）:")
print(f"  BUG SR:     {sr_bug:.3f}  (4/5子步仍为True → 虚高)")
print(f"  CORRECT SR: {sr_correct2:.3f}")

print("\n" + "=" * 70)
print("TEST 2: inter_drone 出生距离 vs margin")
print("=" * 70)

# 模拟 n_drones_per_group=8, batch=64, margin 范围 0.3-0.8
B = 64
G = 8
n_groups = B // G
margin = torch.rand(B, device=device) * 0.5 + 0.3  # [0.3, 0.8]

# 出生间距 = 1.0m (用户参数)
# 最差情况：同一高度，欧氏距离 = 椭球距离 = 1.0
spawn_inter_dist = 1.0

margin_grouped = margin[:n_groups * G].view(n_groups, G)
# 两无人机间 margin 之和
margin_sum_max = (margin_grouped.max(dim=1).values + margin_grouped.max(dim=1).values).max().item()
margin_sum_avg = (margin_grouped.mean(dim=1) * 2).mean().item()
margin_sum_min = (margin_grouped.min(dim=1).values + margin_grouped.min(dim=1).values).min().item()

print(f"spawn_inter_distance: {spawn_inter_dist:.1f}m (欧氏/椭球)")
print(f"margin_sum — min: {margin_sum_min:.2f}, avg: {margin_sum_avg:.2f}, max: {margin_sum_max:.2f}")
print(f"有效距离 min = {spawn_inter_dist - margin_sum_max:.2f}m")
print(f"有效距离 avg = {spawn_inter_dist - margin_sum_avg:.2f}m")
if spawn_inter_dist - margin_sum_max < 0:
    print(f"⚠️  出生即碰撞! 最差情况下有效距离为负")
else:
    print(f"✓ 最差情况下有效距离仍为正")

# 推荐出生间距
recommended = margin_sum_max + 0.5
print(f"\n推荐 min_spawn_inter_distance: ≥ {recommended:.1f}m")

print("\n" + "=" * 70)
print("TEST 3: drone_collide 损失量级分析")
print("=" * 70)

ctl_dt = 1.0 / 15  # ~0.0667s
coef_drone_collide = 5.0

# 场景1：出生即接触 (距离 ≈ 0)
dist = torch.tensor([0.0, 0.05, 0.1, 0.3, 0.5, 1.0], device=device)
softplus_val = F.softplus(dist.mul(-32))
print("softplus(-32*d) 在不同距离下的值:")
for d, s in zip(dist.tolist(), softplus_val.tolist()):
    print(f"  d={d:.2f}m → softplus = {s:.4f}")

# 场景2: 两无人机相向飞行 1 m/s (典型)
v_rel = 1.0  # m/s
v_to_drone = max(v_rel / ctl_dt, 1.0)  # ≈ 15
dist_at_contact = 0.0
loss_per_pair = F.softplus(torch.tensor(-32 * dist_at_contact)).item() * v_to_drone
print(f"\n典型碰撞场景:")
print(f"  v_rel=1.0 m/s → v_to_drone = {v_to_drone:.1f}")
print(f"  dist=0.0 → loss_per_pair = {loss_per_pair:.2f}")
print(f"  coef * loss = {coef_drone_collide * loss_per_pair:.2f}")

# 场景3: 近距离但没接触 (dist=0.3)
dist_near = 0.3
loss_near = F.softplus(torch.tensor(-32 * dist_near)).item() * v_to_drone
print(f"\n近距离未碰撞:")
print(f"  dist=0.3m → loss_per_pair = {loss_near:.4f}")
print(f"  coef * loss = {coef_drone_collide * loss_near:.4f}")

print("\n" + "=" * 70)
print("TEST 4: 各损失项典型量级对比")
print("=" * 70)

# 模拟各损失项的典型值 (从训练日志推算)
# 用户命令: coef_v=1.0, coef_v_pred=2.0, coef_collide=3.0, coef_obj_avoidance=2.0
# coef_d_acc=0.01, coef_d_jerk=0.001, coef_ground_affinity=0.5, coef_lateral=0.5
# coef_drone_collide=5.0 (default)
print(f"{'损失项':<25} {'系数':>6} {'典型原始值':>12} {'加权值':>12} {'占比':>8}")
print("-" * 70)

# 典型原始值 (from first few iters in training log):
# L=3.158 = total, v=0.115/1.0, collide=0.060/3.0=0.020, obj_avoidance=0.127/2.0=0.064
items = [
    ('loss_v (adaptive)', 1.0, 0.115),
    ('loss_v_pred', 2.0, 1.5),  # 通常是最大的
    ('loss_collide', 3.0, 0.020),
    ('loss_obj_avoidance', 2.0, 0.064),
    ('loss_drone_collide', 5.0, 0.50),   # 估计: 多机近距离
    ('loss_d_acc', 0.01, 10.0),
    ('loss_d_jerk', 0.001, 50.0),
    ('loss_ground_affinity', 0.5, 0.0),  # 新实现：正常高度=0
    ('loss_lateral', 0.5, 0.15),
]
total = sum(c * v for _, c, v in items)
for name, coef, raw in items:
    weighted = coef * raw
    pct = weighted / total * 100
    print(f"{name:<25} {coef:>6.3f} {raw:>12.4f} {weighted:>12.4f} {pct:>7.1f}%")
print("-" * 70)
print(f"{'合计':<25} {'':>6} {'':>12} {total:>12.4f} {'100.0':>7}%")

# 关键比较
collide_weighted = 5.0 * 0.50
v_weighted = 1.0 * 0.115
print(f"\n⚠️  drone_collide 加权值 ({collide_weighted:.2f}) 是 loss_v ({v_weighted:.3f}) 的 {collide_weighted/v_weighted:.0f} 倍")
print("   → 如果 drone_collide 方差大，可能淹没导航梯度")

print("\n" + "=" * 70)
print("TEST 5: loss.py 内部 success_rate 指标验证")
print("=" * 70)

distance = torch.randn(T, S, B, device=device) * 0.5 + 1.0  # 大部分 > 0
distance[10, 2, 5] = -0.1  # 仅1个碰撞
sr_internal = torch.all(distance.flatten(0, 1) > 0).float().item()
print(f"shape: {distance.shape}, 碰撞数: 1/{distance.numel()}")
print(f"loss.py 内部 success_rate = {sr_internal:.0f}  ← 任意1个碰撞就为0")
print("(此指标在 train.py 中被 compute_navigation_metrics_torch 的结果覆盖)")

print("\n✅ 诊断完成")
