"""E3D-7 共享任务定义：动态高度跟踪（三脚本统一，防约定漂移）。

Stage 0 静态任务审计发现（坑#2 应验）：反馈策略几乎完全掩盖两类失配（eval loss 仅
1.2×，残留只有稳态小偏置）——静态站立下三老师对比度不足。对策：任务改为**正弦高度
跟踪** z*(t) = rest + 0.04 + A·sin(2πt/T)，观测加 [sin φ, cos φ] 相位：
  · 无积分作用的 MLP 反馈对持续瞬态留持久跟踪误差；
  · 跟踪的**前馈时机/幅值只能从正确动力学模型学到**——模型正确性在此兑现；
  · 高度调制使 f_n 时变 → M_force 的扰动力矩也时变，纯反馈更难掩盖；
  · 最低目标 rest+0.015 仍高于被动平衡（保留 E3D-3 坑#6 的"策略必要性"）。
失配定义与 E3D-6 完全一致（叙事连续）。损失权重沿用 E3D-3，z* 换为时变。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
from srbd_standing import level_error  # noqa: E402
from residual_3d import standing_step_hooked  # noqa: E402
from e3d3_standing_train import sample_init, level_to_angle  # noqa: E402

ACT_SCALE = 0.10
TRACK_MID = 0.04           # 轨迹中心 = rest + 0.04（与 E3D-3 目标一致）
TRACK_AMP = 0.03           # 正弦振幅 (m)；最低 rest+0.01 仍高于被动平衡
PERIOD_S = 0.6             # 周期 (s)
H_TRAIN = 600              # 训练视野 = 1 周期（相位 obs + 随机初态覆盖全相位）
H_EVAL = 2400              # 评测视野 = 4 周期（长 rollout 暴露漂移，E3D-3 坑#8）
W_H = 20.0                 # 高度跟踪权重。**校准记录（第一版任务的 bug）**：沿用 E3D-3 的
#  w_h=4 时，"不跟踪"的高度罚 4·A²/2≈0.0013 低于跟踪所需速度罚 0.05·v_peak²/2≈0.004
#  → 最优策略是不跟踪（审计实测基线 RMSE 38.7mm > 不跟的 17.7mm）。w_h=20、T=0.6s 后：
#  不跟罚 20·A²/2=0.009 > 跟踪速度代价≈0.0025，跟踪才是最优。


class PolicyDyn(nn.Module):
    """obs 11 = [z−z*(t), tilt(2), v(3), w(3), sinφ, cosφ] → 4 腿伸长。"""

    def __init__(self, obs=11, hid=32, act=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act))

    def forward(self, obs):
        return ACT_SCALE * torch.tanh(self.net(obs))


def z_star_phase(t_step: int, cfg):
    """步号 → (z*(t), φ)。"""
    phi = 2.0 * np.pi * (t_step * cfg.dt) / PERIOD_S
    return cfg.rest_height + TRACK_MID + TRACK_AMP * np.sin(phi), phi


def observe_dyn(state, z_star: float, phi: float):
    from pytorch3d.transforms import quaternion_to_matrix
    R = quaternion_to_matrix(state.q)
    tilt = R[:, 2, :2]
    B = state.p.shape[0]
    ph = state.p.new_tensor([np.sin(phi), np.cos(phi)]).expand(B, 2)
    return torch.cat([(state.p[:, 2:3] - z_star), tilt, state.v, state.w, ph], dim=-1)


def loss_step(state, a, z_star: float):
    h_err = (state.p[:, 2] - z_star) ** 2
    att = level_error(state.q)
    vel = (state.v ** 2).sum(-1) + 0.1 * (state.w ** 2).sum(-1)
    horiz = (state.p[:, :2] ** 2).sum(-1)
    return (W_H * h_err + 2.0 * att + 0.05 * vel + 0.2 * horiz
            + 1e-3 * (a ** 2).sum(-1)).mean()


def rollout_train(policy, cfg, state, horizon, extra_fn, tbptt: int | None = None):
    """可微训练 rollout：extra_fn(state, a)->(f_extra, dx_body) 区分三老师。
    tbptt=K：每 K 步 detach 状态（截断 BPTT）——E3D-7 视野扫描实测梯度只在 H≲150 可用
    （长视野被轨迹发散淹没、全程 BPTT 在难系统上发散），截断是处方而非妥协。"""
    loss = state.p.new_zeros(())
    for t in range(horizon):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach()
        zs, phi = z_star_phase(t, cfg)
        a = policy(observe_dyn(state, zs, phi))
        fe, dx = extra_fn(state, a)
        state, _ = standing_step_hooked(state, a, cfg, fe, dx)
        loss = loss + loss_step(state, a, zs)
    return loss / horizon


def train(cfg, extra_fn, horizon=H_TRAIN, iters=80, B=64, lr=3e-3, seed=0, clip=1.0,
          tbptt: int | None = None):
    torch.manual_seed(seed)
    pol = PolicyDyn().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = {"loss": [], "gnorm": []}
    for _ in range(iters):
        s = sample_init(cfg, B, gen)
        loss = rollout_train(pol, cfg, s, horizon, extra_fn, tbptt)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(pol.parameters(), clip).item()
        opt.step()
        hist["loss"].append(loss.item()); hist["gnorm"].append(gnorm)
    return pol, hist


def policy_grad(pol, cfg, extra_fn, seed=11, B=32, horizon=H_TRAIN):
    """固定策略下该老师给的 ∇θL（展平向量），确定性（固定初态批）。"""
    for p in pol.parameters():
        p.grad = None
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, B, gen)
    loss = rollout_train(pol, cfg, s, horizon, extra_fn)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in pol.parameters()]).detach()


@torch.no_grad()
def rollout_eval(policy, cfg, extra_fn, horizon=H_EVAL, seed=7, B=32):
    """评测 rollout（任意系统）：loss / 跟踪RMSE / 轨迹 / 前馈签名(前后腿差)。"""
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, B, gen)
    loss = 0.0
    zs, zt, atts, xs, ff = [], [], [], [], []
    for t in range(horizon):
        z_star, phi = z_star_phase(t, cfg)
        a = (policy(observe_dyn(s, z_star, phi)) if policy is not None
             else torch.zeros(B, 4, device=cfg.device, dtype=cfg.dtype))
        fe, dx = extra_fn(s, a)
        s, _ = standing_step_hooked(s, a, cfg, fe, dx)
        loss += loss_step(s, a, z_star).item()
        zs.append(s.p[:, 2].mean().item()); zt.append(z_star)
        atts.append(np.degrees(level_to_angle(level_error(s.q).mean().item())))
        xs.append(s.p[:, 0].mean().item())
        ff.append((a[:, :2].mean() - a[:, 2:].mean()).item())
    zs, zt = np.array(zs), np.array(zt)
    half = horizon // 2
    return dict(loss=loss / horizon, zs=zs, zt=zt, atts=np.array(atts),
                xs=np.array(xs),
                track_rmse=float(np.sqrt(((zs - zt)[half:] ** 2).mean())),
                ff=float(np.mean(ff[half:])),
                final_tilt=float(np.array(atts)[half:].mean()),
                final_x=float(abs(xs[-1])))
