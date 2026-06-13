"""自适应步态基建（E3D-8）：把固定调度的 trot 换成**相位作状态**的可调步态。

E3D-4a 的固定步态把 period/duty/PHASE_OFF 写死、相位由 t_step 闭式给出——这是个
**梯度选择**（接触调度对参数平滑可微，绕开 2D-F4 的 1/dt 摩擦梯度爆炸）。本模块把相位
提升为一个连续状态 φ，由瞬时频率 ω 推进，从而频率/步态可被策略或耦合动力学调制，
而**接触调度仍是 φ 的平滑函数 → 梯度保真不变**。这是 E3D-8 闸门要验证的命题。

核心区别于 gait_3d.foot_plan：足端体系速度 ṗ_b 现在按**瞬时相位率** φ̇ 解析缩放
（ẋ = (dx/dφ)·φ̇），而非固定的 1/period。频率变 → 扫腿速度自动跟着变，摩擦速度伺服
推进（E3D-4a 机制）仍成立。配套发现：频率调制下，为保持指令速度，标称扫程必须
Lx0 = vx_cmd·duty/ω 随 ω 缩放——速度-频率-步幅自洽。

通道：foot_geom（相位驱动足端几何）→ gait_dynamics_step（接触+wrench+srbd）。
相位推进由各臂控制器在外部完成（fixed 常 ω / phaseA 策略调 ω / hard 硬接触负控）。
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix

from floating_base_srbd import FloatingBaseState, srbd_step
from contact_3d import foot_contact_force_world
from srbd_standing import StandingConfig
from gait_3d import GaitConfig, PHASE_OFF


def foot_geom(phi: torch.Tensor, phidot: torch.Tensor, Lx: torch.Tensor,
              dext: torch.Tensor, cfg: StandingConfig, g: GaitConfig):
    """相位驱动的足端体系轨迹（gait_3d.foot_plan 的相位状态泛化）。

    入参全 (B,4)：phi 每腿相位∈[0,1)、phidot 每腿相位率 [cycle/s]、Lx 每腿扫程 [m]、
    dext 每腿支撑深度修正 [m]（均**已限幅**，与 foot_plan 内部 tanh 不同，标度在外做）。
    返回 p_b (B,4,3)、pdot_b (B,4,3)、stance (B,4) bool。
    速度按 ṗ = (d p/d φ)·φ̇ 解析合成（绝不位置差分）。"""
    fr = cfg.foot_rel_com                                   # (4,3)
    st = phi < g.duty                                       # (B,4)
    # 支撑：线性后扫，dx/dφ = -Lx/duty
    s_st = phi / g.duty
    x_st = (0.5 - s_st) * Lx
    xd_st = -(Lx / g.duty) * phidot
    # 摆动：端点速率连续三次样条（与 foot_plan 同系数），dx/dφ = Lx·f'(s)/(1-duty)
    s_sw = ((phi - g.duty) / (1.0 - g.duty)).clamp(0.0, 1.0)
    k = (1.0 - g.duty) / g.duty
    a3, a2 = -2.0 * (1.0 + k), 3.0 * (1.0 + k)
    f = a3 * s_sw ** 3 + a2 * s_sw ** 2 - k * s_sw
    fp = 3 * a3 * s_sw ** 2 + 2 * a2 * s_sw - k
    x_sw = (-0.5 + f) * Lx
    xd_sw = (Lx * fp / (1.0 - g.duty)) * phidot
    z_lift = g.h_swing * torch.sin(np.pi * s_sw)
    zd_lift = (g.h_swing * np.pi * torch.cos(np.pi * s_sw) / (1.0 - g.duty)) * phidot
    x = torch.where(st, x_st, x_sw)
    xdot = torch.where(st, xd_st, xd_sw)
    z0 = fr[None, :, 2] - g.ext0 - dext                    # (B,4)
    z = torch.where(st, z0, z0 + z_lift)
    zdot = torch.where(st, torch.zeros_like(zd_lift), zd_lift)
    p_b = torch.stack([fr[None, :, 0] + x, fr[None, :, 1].expand_as(x), z], dim=-1)
    pdot_b = torch.stack([xdot, torch.zeros_like(xdot), zdot], dim=-1)
    return p_b, pdot_b, st


def gait_dynamics_step(state: FloatingBaseState, phi: torch.Tensor, phidot: torch.Tensor,
                       Lx: torch.Tensor, dext: torch.Tensor,
                       cfg: StandingConfig, g: GaitConfig, mode: str = "smooth"):
    """一步：相位足端几何→世界足位/解析足速→接触→wrench→srbd_step（gait_step 的相位版）。
    mode='hard' = 接触事件式硬切换（E3D-8 负控，复刻 E3D-4a 梯度分水岭）。"""
    p_b, pdot_b, stance = foot_geom(phi, phidot, Lx, dext, cfg, g)
    R = quaternion_to_matrix(state.q)
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, p_b)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = (state.v[:, None, :]
              + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
              + torch.einsum("bij,bkj->bki", R, pdot_b))
    out = foot_contact_force_world(foot_w, foot_v, cfg.contact, mode=mode)
    f_each = out["f_world"]
    tau_world = torch.cross(r, f_each, dim=-1).sum(1)
    tau_body = torch.einsum("bji,bj->bi", R, tau_world)
    f_tot = f_each.sum(1)
    nxt = srbd_step(state, cfg.mass, cfg.I_body, cfg.I_body_inv, cfg.dt,
                    f_world=f_tot, tau_body=tau_body)
    cone = torch.linalg.norm(out["f_t"], dim=-1) / out["mu_fn"].squeeze(-1).clamp_min(1e-9)
    info = dict(f_n=out["f_n"].squeeze(-1), cone=cone, foot_world=foot_w,
                foot_v=foot_v, stance=stance)
    return nxt, info


# --------------------------------------------------------------------------- #
# 相位率 → 标称扫程：频率调制下保持指令速度的自洽关系（速度=频率×步幅）。
# --------------------------------------------------------------------------- #
def nominal_lx(omega: torch.Tensor, cfg: StandingConfig, g: GaitConfig) -> torch.Tensor:
    """支撑足体系后扫速度 = -(Lx/duty)·ω；令其 ≈ -vx_cmd（足世界速度≈0，锚定）
    ⇒ Lx0 = vx_cmd·duty/ω。ω=1/period 时退回 g.lx0。omega:(B,) → (B,)。"""
    return g.vx_cmd * g.duty / omega


def trot_phase(phi_g: torch.Tensor) -> torch.Tensor:
    """全局相位 (B,) → 每腿 trot 相位 (B,4)，对角同相（PHASE_OFF）。"""
    off = phi_g.new_tensor(PHASE_OFF)
    return (phi_g[:, None] + off) % 1.0


# --------------------------------------------------------------------------- #
# 耦合 CPG（E3D-8b 步态自涌现）：每腿一个相位振荡器，经可学习耦合 K 互相牵引，
# 锁相后的相位模式（哪些腿同/反相）即步态——不由 PHASE_OFF 写死，而**对称破缺涌现**。
# 闸门(E3D-8)已证耦合是平滑可微的 → 用 BPTT 训 K（步态本身）梯度保真。
# --------------------------------------------------------------------------- #
def cpg_phase_rate(phi: torch.Tensor, omega: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Kuramoto 相位率：φ̇_i = ω + Σ_j K_ij·sin(2π(φ_j − φ_i))。
    phi:(B,4) cycles；omega:(B,) 或标量 cycle/s；K:(4,4) 耦合 [cycle/s]。返回 (B,4)。
    锁相不定 Δ_ij（不预设目标相位）→ 模式由 K 的结构决定，可学习涌现。"""
    diff = phi[:, None, :] - phi[:, :, None]        # (B,4,4): [.,i,j]=φ_j−φ_i
    coup = (K[None] * torch.sin(2 * np.pi * diff)).sum(dim=2)   # Σ_j (B,4)
    if omega.dim() == 1:
        omega = omega[:, None]
    return omega + coup


def trot_coupling(strength: float = 4.0) -> torch.Tensor:
    """参照用的 trot 耦合（手设，emerge 阶段的"已知答案"对照）：腿序 FL,FR,RL,RR，
    对角对 (FL,RR)/(FR,RL) 同相、两对反相。K_ij>0 拉同相、<0 推反相。"""
    same = [(0, 3), (3, 0), (1, 2), (2, 1)]          # 对角对 → 同相
    opp = [(0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)]  # 邻 → 反相
    K = torch.zeros(4, 4)
    for i, j in same:
        K[i, j] = strength
    for i, j in opp:
        K[i, j] = -strength
    return K
