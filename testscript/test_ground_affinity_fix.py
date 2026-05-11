"""
验证 ground_affinity 天花板式修复的数学正确性。

检查项:
1. z < z_ceiling → loss=0, gradient=0 
2. z > z_ceiling → loss>0, gradient pushes z down toward ceiling
3. z = z_ceiling → loss=0 (边界值)
4. 修复前后在典型飞行高度范围内的损失对比
5. 模拟损失分量权重平衡
"""

import torch
import sys
sys.path.insert(0, '.')
from loss import DroneLoss

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        print(f'  ✓ {name}')
        passed += 1
    else:
        print(f'  ✗ {name}')
        failed += 1

# ============================================================
print("=" * 60)
print("Test 1: 天花板以下 → 损失=0, 梯度=0")
print("=" * 60)

z_ceiling = 5.0
losser = DroneLoss(coef_ground_affinity=0.5, ga_z_ceiling=z_ceiling)

# 模拟 T=200, B=64 的位置历史, z ∈ [1, 3] (典型飞行高度)
T, B = 200, 64
p = torch.randn(T, B, 3)
p[..., 2] = torch.clamp(p[..., 2].abs() + 1.0, 1.0, 3.0)
p.requires_grad_(True)

loss_ga = (p[..., 2] - z_ceiling).relu().pow(2).mean()
loss_ga.backward()

check('z∈[1,3], z_ceiling=5 → loss_ga == 0', loss_ga.item() == 0.0)
check('z∈[1,3] → 梯度全为零', p.grad.abs().max().item() == 0.0)

# ============================================================
print("\nTest 2: 天花板以上 → 损失>0, 梯度指向下方")
print("=" * 60)

p2 = torch.randn(T, B, 3)
p2[..., 2] = torch.clamp(p2[..., 2].abs() + 6.0, 6.0, 8.0)  # z ∈ [6, 8]
p2.requires_grad_(True)

loss_ga2 = (p2[..., 2] - z_ceiling).relu().pow(2).mean()
loss_ga2.backward()

check('z∈[6,8], z_ceiling=5 → loss_ga > 0', loss_ga2.item() > 0)
# 梯度方向应为正(惩罚增加z)，所以模型应学到减小z
grad_z = p2.grad[..., 2]
check('z>ceiling → z 方向梯度 > 0 (惩罚更高)', grad_z.min().item() > 0)
# x, y 方向梯度应为 0
check('z>ceiling → x 方向梯度 = 0', p2.grad[..., 0].abs().max().item() == 0.0)
check('z>ceiling → y 方向梯度 = 0', p2.grad[..., 1].abs().max().item() == 0.0)

# ============================================================
print("\nTest 3: 边界值 z = z_ceiling → 损失=0")
print("=" * 60)

p3 = torch.randn(T, B, 3)
p3[..., 2] = z_ceiling  # 精确在天花板
p3.requires_grad_(True)

loss_ga3 = (p3[..., 2] - z_ceiling).relu().pow(2).mean()
loss_ga3.backward()

check('z == z_ceiling → loss = 0', loss_ga3.item() == 0.0)
check('z == z_ceiling → 梯度 = 0', p3.grad.abs().max().item() == 0.0)

# ============================================================
print("\nTest 4: 修复前后损失对比")
print("=" * 60)

# 典型飞行参数: z ∈ [1, 3]
z_vals = torch.linspace(0.5, 8.0, 16)
print(f'  {"z":>6}  {"旧loss":>10}  {"新loss":>10}  {"新/旧":>8}')
for z in z_vals:
    old_loss = max(z.item(), 0) ** 2  # old: relu(z)^2
    new_loss = max(z.item() - z_ceiling, 0) ** 2  # new: relu(z-ceiling)^2
    ratio = new_loss / old_loss if old_loss > 0 else 0
    print(f'  {z.item():6.1f}  {old_loss:10.4f}  {new_loss:10.4f}  {ratio:8.4f}')

check('z=2.0 → 旧loss=4.0, 新loss=0.0', 
      max(2.0, 0)**2 == 4.0 and max(2.0 - 5.0, 0)**2 == 0.0)
check('z=6.0 → 旧loss=36.0, 新loss=1.0 (仍有惩罚)', 
      max(6.0, 0)**2 == 36.0 and max(6.0 - 5.0, 0)**2 == 1.0)

# ============================================================
print("\nTest 5: 通过完整 DroneLoss.forward() 验证")
print("=" * 60)

# 构造最小测试数据
device = 'cpu'
T, B = 50, 8
p_hist = torch.randn(T, B, 3)
p_hist[..., 2] = 2.0  # 正常高度

v_hist = torch.randn(T, B, 3) * 0.5
target_v_hist = torch.randn(T, B, 3) * 2.0
act_hist = torch.randn(T + 2, B, 3) * 0.1
vec_to_obj_hist = torch.randn(T, B, 3)
vec_to_obj_hist = vec_to_obj_hist / vec_to_obj_hist.norm(dim=-1, keepdim=True) * 3.0
v_preds = torch.randn(T, B, 3) * 0.5
env_margin = torch.full((B,), 0.5)

losser_new = DroneLoss(
    coef_ground_affinity=0.5,
    ga_z_ceiling=5.0,
    ctl_dt=1/15,
    window_size=30,
    loss_v_mode='adaptive',
)

loss, metrics = losser_new.forward(
    p_hist, v_hist, target_v_hist, act_hist,
    vec_to_obj_hist, v_preds, env_margin,
)
ga_loss = metrics['loss_ground_affinity']
check('完整forward: z=2.0, ceiling=5 → ga_loss=0', ga_loss.item() == 0.0)

# 高度超过天花板
p_hist_high = p_hist.clone()
p_hist_high[..., 2] = 7.0  # 超过天花板
loss_high, metrics_high = losser_new.forward(
    p_hist_high, v_hist, target_v_hist, act_hist,
    vec_to_obj_hist, v_preds, env_margin,
)
ga_loss_high = metrics_high['loss_ground_affinity']
check('完整forward: z=7.0, ceiling=5 → ga_loss=4.0', abs(ga_loss_high.item() - 4.0) < 0.01)
check('完整forward: z=7.0 时总损失 > z=2.0', loss_high.item() > loss.item())

# ============================================================
print("\nTest 6: 模拟损失平衡验证")
print("=" * 60)

# 使用用户的参数，验证修复后 loss_v 占比提升
import csv
import os
csv_path = 'checkpoints/trainbase_grop8_run20260321-1/metrics.csv'
if os.path.exists(csv_path):
    with open(csv_path) as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    
    # 计算修复后的损失分布
    N = min(20, len(rows))
    coefs = {'v':1.0, 'collide':3.0, 'obj_avoidance':2.0, 
             'ground_affinity':0.5, 'drone_collide':5.0, 'v_pred':2.0}
    
    old_total = 0
    old_ga_contrib = 0
    old_v_contrib = 0
    for i in range(N):
        for name, c in coefs.items():
            val = c * float(rows[i].get(f'loss_{name}', 0))
            old_total += val
            if name == 'ground_affinity':
                old_ga_contrib += val
            elif name == 'v':
                old_v_contrib += val
    
    # 修复后: ga 贡献 → 0 (z < ceiling)
    new_total = old_total - old_ga_contrib
    old_v_pct = 100 * old_v_contrib / old_total
    new_v_pct = 100 * old_v_contrib / new_total
    
    print(f'  修复前: loss_v 占比 = {old_v_pct:.1f}%')
    print(f'  修复后: loss_v 占比 = {new_v_pct:.1f}%')
    print(f'  信噪比提升: {new_v_pct / old_v_pct:.2f}x')
    
    check(f'修复后导航信号占比 > 35%', new_v_pct > 35)
    check(f'修复后导航信号倍增', new_v_pct > old_v_pct * 1.3)
else:
    print('  (CSV 不存在，跳过此项)')

# ============================================================
print(f'\n{"=" * 60}')
print(f'结果: {passed} 通过, {failed} 失败')
print(f'{"=" * 60}')
if failed > 0:
    sys.exit(1)
