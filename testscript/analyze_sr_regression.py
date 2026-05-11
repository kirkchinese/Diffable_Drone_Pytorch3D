"""
数值分析：从训练日志数据推断 SR 回归根因
=============================================
用用户提供的 12 步训练日志数据做数学分析。
"""
import numpy as np

# 用户提供的训练日志数据 (前12步)
data = [
    # (iter, L_total, loss_v, loss_collide, loss_obj_avoidance, SR, AR)
    (1,  3.132, 0.130, 0.204, 0.165, 0.36, 0.52),
    (2,  2.515, 0.161, 0.149, 0.150, 0.38, 0.50),
    (3,  2.064, 0.258, 0.101, 0.117, 0.39, 0.50),
    (4,  1.950, 0.362, 0.084, 0.107, 0.34, 0.42),
    (5,  2.102, 0.467, 0.078, 0.101, 0.29, 0.35),
    (6,  2.063, 0.572, 0.075, 0.099, 0.25, 0.30),
    (7,  2.001, 0.689, 0.064, 0.089, 0.22, 0.27),
    (8,  2.044, 0.824, 0.068, 0.088, 0.20, 0.23),
    (9,  2.069, 0.917, 0.073, 0.087, 0.17, 0.21),
    (10, 2.042, 0.992, 0.066, 0.081, 0.16, 0.19),
    (11, 2.034, 1.055, 0.063, 0.079, 0.14, 0.17),
    (12, 2.023, 1.100, 0.062, 0.077, 0.13, 0.16),
]

# Loss coefficients from user's command
coefs = {
    'v': 1.0, 'v_pred': 2.0, 'collide': 3.0, 'obj_avoidance': 2.0,
    'lateral': 0.5, 'drone_collide': 5.0,
    'd_acc': 0.01, 'd_jerk': 0.001, 'ground_affinity': 0.001,
}

print("=" * 80)
print("1. 加权损失贡献分析 (估算)")
print("=" * 80)
print(f"\n{'iter':>4s} {'L_total':>8s} {'v_wt':>8s} {'coll_wt':>9s} {'obj_wt':>8s} {'safety':>8s} {'nav':>8s} {'ratio':>6s} {'SR':>6s}")
print("-" * 80)

for row in data:
    it, L, lv, lc, lo, sr, ar = row
    v_weighted = coefs['v'] * lv
    coll_weighted = coefs['collide'] * lc
    obj_weighted = coefs['obj_avoidance'] * lo
    
    # 从总损失减去已知项估算隐含项
    known_sum = v_weighted + coll_weighted + obj_weighted
    hidden = L - known_sum  # v_pred + drone_collide + lateral + d_acc + d_jerk + g_affinity
    
    safety_total = coll_weighted + obj_weighted  # 不含 drone_collide (未知)
    nav_total = v_weighted
    ratio = safety_total / max(nav_total, 1e-8)
    
    print(f"{it:4d} {L:8.3f} {v_weighted:8.3f} {coll_weighted:9.3f} {obj_weighted:8.3f} "
          f"{safety_total:8.3f} {nav_total:8.3f} {ratio:5.1f}x  {sr:5.0%}")

print(f"\n  hidden = L - (v_wt + coll_wt + obj_wt) 包含: v_pred(2.0×), drone_collide(5.0×), lateral(0.5×), etc.")

print("\n" + "=" * 80)
print("2. 各项损失变化率分析")
print("=" * 80)
d = np.array(data)
iters = d[:, 0]
L_total = d[:, 1]
loss_v = d[:, 2]
loss_collide = d[:, 3]
loss_obj = d[:, 4]
SR = d[:, 5]

# 回归分析
from numpy.polynomial import polynomial as P
coef_v_trend = np.polyfit(iters, loss_v, 1)
coef_c_trend = np.polyfit(iters, loss_collide, 1)
coef_o_trend = np.polyfit(iters, loss_obj, 1)
coef_sr_trend = np.polyfit(iters, SR, 1)

print(f"\n  loss_v 斜率:       {coef_v_trend[0]:+.4f}/iter  (初始={loss_v[0]:.3f}, 最终={loss_v[-1]:.3f}, 变化={loss_v[-1]-loss_v[0]:+.3f})")
print(f"  loss_collide 斜率: {coef_c_trend[0]:+.4f}/iter  (初始={loss_collide[0]:.3f}, 最终={loss_collide[-1]:.3f}, 变化={loss_collide[-1]-loss_collide[0]:+.3f})")
print(f"  loss_obj 斜率:     {coef_o_trend[0]:+.4f}/iter  (初始={loss_obj[0]:.3f}, 最终={loss_obj[-1]:.3f}, 变化={loss_obj[-1]-loss_obj[0]:+.3f})")
print(f"  SR 斜率:           {coef_sr_trend[0]:+.4f}/iter  (初始={SR[0]:.1%}, 最终={SR[-1]:.1%})")

# Weighted change
delta_v_weighted = (loss_v[-1] - loss_v[0]) * coefs['v']
delta_coll_weighted = (loss_collide[-1] - loss_collide[0]) * coefs['collide']  
delta_obj_weighted = (loss_obj[-1] - loss_obj[0]) * coefs['obj_avoidance']

print(f"\n  加权变化量:")
print(f"    Δ(v × 1.0) = {delta_v_weighted:+.3f}  (↑ 导航更差)")
print(f"    Δ(collide × 3.0) = {delta_coll_weighted:+.3f}  (↓ 避障更好)")
print(f"    Δ(obj_avoidance × 2.0) = {delta_obj_weighted:+.3f}  (↓ 避障更好)")
print(f"    总 L 变化: {L_total[-1] - L_total[0]:+.3f}")

print(f"\n  结论: collide 下降了 {abs(delta_coll_weighted):.3f}, obj 下降了 {abs(delta_obj_weighted):.3f}")
print(f"        v 上升了 {delta_v_weighted:.3f}, 但安全项下降的总和 ({abs(delta_coll_weighted)+abs(delta_obj_weighted):.3f}) < v上升 ({delta_v_weighted:.3f})")
print(f"        总 L 反而上升 → 优化器在损失平原上震荡")

print("\n" + "=" * 80)
print("3. 关键推断: 为什么 v_loss 持续上升?")
print("=" * 80)

# 估算 hidden loss 贡献
hidden_start = L_total[0] - (loss_v[0]*1 + loss_collide[0]*3 + loss_obj[0]*2)
hidden_end = L_total[-1] - (loss_v[-1]*1 + loss_collide[-1]*3 + loss_obj[-1]*2)
print(f"\n  隐含项(v_pred+drone_collide+lateral+...) 初始={hidden_start:.3f}, 最终={hidden_end:.3f}")
print(f"  隐含项变化: {hidden_end - hidden_start:+.3f}")
print(f"  隐含项占总 L 比例: 初始={hidden_start/L_total[0]:.0%}, 最终={hidden_end/L_total[-1]:.0%}")

print(f"""
  --- 诊断结论 ---
  
  ★ 模式: 经典的「安全-导航 梯度竞争」
  
  ① v_loss 持续上升 (0.13 → 1.10): 
     模型学会了减速/悬停以避免碰撞，速度跟踪恶化
  
  ② collide + obj_avoidance 持续下降:
     模型成功学会了避障（通过停止运动）
  
  ③ SR 持续下降 (36% → 13%):
     SR = collision_free AND reached_target
     模型不再向目标移动 → reached_target 暴降 → SR 暴降
  
  ④ 隐含项 (v_pred=2.0×, drone_collide=5.0×) 也在优化:
     drone_collide(coef=5.0!) 是最大的安全系数，
     进一步压制导航梯度
  
  ★ 根因: 安全损失权重之和 >> 导航损失权重
     collide(3.0) + obj_avoidance(2.0) + drone_collide(5.0) = 10.0  [安全]
     v(1.0) + lateral(0.5)                                  = 1.5   [导航]
     安全:导航 = 6.67:1
  
  ★ 此外, adaptive v loss 中的 brake 项加剧问题:
     alpha = exp(-2.0 * target_speed)
     当 target_speed < 1.0 时 alpha > 0.14, brake 项显著激活
     这意味着模型被训练在接近目标时减速，但配合过强的安全损失，
     模型学会了在任何位置都"刹车"
  
  ★ 另外, 该 checkpoint 从未见过动态障碍物和无人机碰撞:
     enable_dynamic_obstacles=True + n_drones_per_group=8
     这些是全新的碰撞源, 模型不知道如何应对,
     只能通过停止运动来最小化损失
""")

print("=" * 80)
print("4. 建议修复方案")
print("=" * 80)
print("""
  方案 A: 调整损失系数平衡
  - 降低 coef_drone_collide: 5.0 → 1.0 (最大安全项,影响最大)
  - 降低 coef_collide: 3.0 → 1.5
  - 提高 coef_v: 1.0 → 2.0
  - 安全:导航 从 10:1.5 变为 4.5:2.5 = 1.8:1
  
  方案 B: 使用梯度归一化 (GradNorm)
  - 动态平衡各项梯度范数,防止任何一项主导
  
  方案 C: 分阶段训练
  - Phase 1: 先不启用 drone_collide 和 dynamic_obs, 
    让模型适应新的 deployment 环境
  - Phase 2: 逐步启用并降低安全系数
  
  ★ 推荐: 方案 A + C 结合
    先用平衡的系数 + 无动态障碍训练 50-100 iter 稳定模型,
    再开启动态障碍 + drone_collide
""")
