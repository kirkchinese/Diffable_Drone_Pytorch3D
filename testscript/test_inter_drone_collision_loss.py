import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_inter_drone_collision_loss():
    from loss import DroneLoss

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    losser = DroneLoss(ctl_dt=1 / 15)

    T, B = 4, 3
    zeros = torch.zeros(T, B, 3, device=device)
    vec_to_obj = torch.full((T, 2, B, 3), 2.0, device=device)
    v_preds = torch.zeros(T, B, 3, device=device)
    env_margin = torch.full((B,), 0.3, device=device)
    g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)

    # 无人机间距离历史 (T, S, B)，已减去 margin
    # 正值 = 安全距离，负值 = 碰撞
    safe_dist = torch.full((T, 2, B), 0.8, device=device)  # 安全，距离 > 0
    collide_dist = safe_dist.clone()
    collide_dist[..., 0] = -0.1  # 无人机 0 发生碰撞

    # 无无人机间碰撞
    _, metrics_safe = losser.forward(
        p_history=zeros,
        v_history=zeros,
        target_vel_history=zeros,
        act_history=torch.zeros(T + 1, B, 3, device=device),
        vec_to_obj_history=vec_to_obj,
        v_preds=v_preds,
        env_margin=env_margin,
        env_g_std=g_std,
        inter_drone_dist_history=safe_dist,
    )
    # 有无人机间碰撞
    _, metrics_collide = losser.forward(
        p_history=zeros,
        v_history=zeros,
        target_vel_history=zeros,
        act_history=torch.zeros(T + 1, B, 3, device=device),
        vec_to_obj_history=vec_to_obj,
        v_preds=v_preds,
        env_margin=env_margin,
        env_g_std=g_std,
        inter_drone_dist_history=collide_dist,
    )

    # 验证 inter-drone 碰撞损失增大（机间惩罚在 loss_drone_collide，非静态障碍的 loss_collide）
    assert float(metrics_collide['loss_drone_collide']) > float(metrics_safe['loss_drone_collide']), \
        '发生无人机碰撞时 inter-drone collision loss 未增大'
    print('[PASS] DroneLoss 正确注入 inter-drone collision penalty')


if __name__ == '__main__':
    test_inter_drone_collision_loss()
