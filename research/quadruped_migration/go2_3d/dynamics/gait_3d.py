"""预定 trot 步态的足端轨迹生成 + SRBD 步态步进（E3D-4）。

设计要点（对应预注册坑）：
  1. **推进 = 摩擦速度伺服**〔推导〕：SRBD+运动学足模型中体上所有力来自接触。支撑足在
     体系内以 vx_cmd 匀速后扫 → 足世界速度 ≈ v_body − vx_cmd → 正则库仑摩擦
     −μf_n·tanh(v_f/v_ε) 自动"慢了推、快了拽"。跟踪良好时足世界速度≈0 ⇒ **锚定涌现**
     （支撑期足端世界漂移为诊断量），锥占用仅正比于速度误差，梯度友好。
  2. **足速解析合成，绝不位置差分**（2D-F4 的 1/dt 摩擦梯度爆炸）：
     v_foot = v_com + ω_w×(R·p_b) + R·ṗ_b，ṗ_b 由相位闭式给出。
  3. **相位闭式、无记忆**：支撑 x 从 +Lx/2 线性扫到 −Lx/2，摆动余弦返回 + 正弦抬腿；
     边界位置连续（速率在离地瞬间不连续，但该瞬间接触力≈0，无害）。
  4. 接触切换由预定调度+平滑接触自然处理（不学习触发）。

动作空间 a∈R⁸ = 每腿 [ΔLx(改落足/扫速,运动学通道, ±0.04m), Δext(改支撑深度,力/高度
通道, ±0.03m)]——与双头残差的通道结构天然对齐（E3D-4b 用）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix

from floating_base_srbd import FloatingBaseState, srbd_step
from contact_3d import foot_contact_force_world
from srbd_standing import StandingConfig

# trot：对角腿对同相。腿序 FL,FR,RL,RR。
PHASE_OFF = (0.0, 0.5, 0.5, 0.0)


@dataclass(frozen=True)
class GaitConfig:
    vx_cmd: float = 0.3        # 前向速度指令 [m/s]
    period: float = 0.4        # 步态周期 [s]
    duty: float = 0.5          # 支撑占空比
    h_swing: float = 0.04      # 抬腿高度 [m]
    ext0: float = 0.035        # 支撑期标称伸长（足低于站立标称的深度）[m]
    dLx_max: float = 0.04      # 策略落足修正限幅 [m]
    dext_max: float = 0.03     # 策略伸长修正限幅 [m]

    @property
    def t_stance(self) -> float:
        return self.duty * self.period

    @property
    def lx0(self) -> float:    # 标称扫程 = vx_cmd · T_stance
        return self.vx_cmd * self.t_stance


def foot_plan(t_step, a: torch.Tensor, cfg: StandingConfig, g: GaitConfig):
    """相位闭式足端规划（体系），**按样本相位全张量化**（残差训练需逐样本相位）。
    t_step: 标量 或 (B,) 张量；a:(B,8)=每腿[ΔLx,Δext]（tanh 限幅在内）。
    返回 p_b:(B,4,3)、pdot_b:(B,4,3)、stance:(B,4) bool、phase φ:(B,)。"""
    B = a.shape[0]
    dLx = g.dLx_max * torch.tanh(a[:, 0::2])          # (B,4)
    dext = g.dext_max * torch.tanh(a[:, 1::2])        # (B,4)
    Lx = g.lx0 + dLx                                   # (B,4) 每腿扫程
    if not torch.is_tensor(t_step):
        t_step = a.new_full((B,), float(t_step))
    phi_g = (t_step.to(a.dtype) * cfg.dt / g.period) % 1.0          # (B,)
    off = a.new_tensor(PHASE_OFF)                                    # (4,)
    phi = (phi_g[:, None] + off) % 1.0                               # (B,4)
    st = phi < g.duty                                                # (B,4)
    fr = cfg.foot_rel_com                                            # (4,3)
    # 支撑：线性后扫（摩擦速度伺服）
    s_st = phi / g.duty
    x_st = (0.5 - s_st) * Lx
    xd_st = -Lx / g.t_stance                                         # ≈ −vx_cmd
    # 摆动：端点速率连续样条 + 正弦抬腿。f(0)=0,f(1)=1,f'(0)=f'(1)=−(1−duty)/duty
    # → **着地回撤**（触地瞬间世界速度≈0，消除每步制动脉冲——标准步态做法）
    s_sw = ((phi - g.duty) / (1.0 - g.duty)).clamp(0.0, 1.0)
    k = (1.0 - g.duty) / g.duty
    a3, a2 = -2.0 * (1.0 + k), 3.0 * (1.0 + k)
    f = a3 * s_sw ** 3 + a2 * s_sw ** 2 - k * s_sw
    fp = 3 * a3 * s_sw ** 2 + 2 * a2 * s_sw - k
    x_sw = (-0.5 + f) * Lx
    xd_sw = (fp / ((1 - g.duty) * g.period)) * Lx
    z_sw_lift = g.h_swing * torch.sin(np.pi * s_sw)
    zd_sw = g.h_swing * np.pi * torch.cos(np.pi * s_sw) / ((1 - g.duty) * g.period)
    # 合成
    x = torch.where(st, x_st, x_sw)
    xdot = torch.where(st, xd_st, xd_sw)
    z0 = fr[None, :, 2] - g.ext0 - dext                              # (B,4)
    z = torch.where(st, z0, z0 + z_sw_lift)
    zdot = torch.where(st, torch.zeros_like(zd_sw), zd_sw)
    p_b = torch.stack([fr[None, :, 0] + x,
                       fr[None, :, 1].expand(B, 4), z], dim=-1)      # (B,4,3)
    pdot_b = torch.stack([xdot, torch.zeros_like(xdot), zdot], dim=-1)
    return p_b, pdot_b, st, phi_g


def gait_step(state: FloatingBaseState, t_step: int, a: torch.Tensor,
              cfg: StandingConfig, g: GaitConfig, mode: str = "smooth",
              f_extra: torch.Tensor | None = None,
              dx_body: torch.Tensor | None = None):
    """一步：足规划→世界足位/解析足速→接触→wrench→srbd_step。
    f_extra/dx_body = E3D-6/7 同款双通道 hook（E3D-4b 失配/残差用）。"""
    p_b, pdot_b, stance, phi = foot_plan(t_step, a, cfg, g)
    if dx_body is not None:
        p_b = p_b + dx_body
    R = quaternion_to_matrix(state.q)                          # (B,3,3)
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, p_b)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    # 足速 = 刚体部分 + 指令部分（解析，绝不位置差分）
    foot_v = (state.v[:, None, :]
              + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
              + torch.einsum("bij,bkj->bki", R, pdot_b))
    out = foot_contact_force_world(foot_w, foot_v, cfg.contact, mode=mode)
    f_each = out["f_world"] if f_extra is None else out["f_world"] + f_extra
    tau_world = torch.cross(r, f_each, dim=-1).sum(1)
    tau_body = torch.einsum("bji,bj->bi", R, tau_world)
    nxt = srbd_step(state, cfg.mass, cfg.I_body, cfg.I_body_inv, cfg.dt,
                    f_world=f_each.sum(1), tau_body=tau_body)
    cone = torch.linalg.norm(out["f_t"], dim=-1) / out["mu_fn"].squeeze(-1).clamp_min(1e-9)
    info = dict(f_n=out["f_n"].squeeze(-1), cone=cone, foot_world=foot_w,
                foot_v=foot_v, stance=stance, phase=phi)
    return nxt, info
