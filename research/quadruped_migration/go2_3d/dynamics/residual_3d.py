"""Dual-channel residual hooks + heads on the Go2 SRBD twin (E3D-6).

2D 阶段（R3/R8/R9/R10/R11）确立的核心论点："残差 a = a_phys + r_φ 必须放在误差实际发生
的通道——力误差→接触力残差，几何误差→落足(运动学)残差；放错通道即使前向能拟合，梯度
也会坏。" 本模块把这两条通道作为 SRBD 站立环境的注入点（同时充当**已知失配**与**残差头
输出**），供 E3D-6a（通道匹配 2×2）与 E3D-6b（双头自动路由）使用。

通道定义（frame 纪律沿用 E3D-1/2：world↔body 只在力矩求和一处发生）：
  force 通道  f_extra (B,4,3)  WORLD 系、作用于足端接触点的额外力，与接触力同点合成
              （进同一个 r×f 力矩求和，wrench 自洽）。
  kin  通道   dx_body (B,4,3)  BODY 系足端几何偏移：foot_w += R·dx，杠杆与刚体足速
              v = v_com + ω×r 用偏移后的 r —— 失配与残差都走同一几何路径。

一步加速度（回归目标；不积分）：
  a_lin = g + Σf/m            (WORLD)
  a_ang = I⁻¹(τ_body − w×Iw)  (BODY)

已知失配（"真实系统" = SRBD + 失配，ground truth 在手）：
  M_force: 载荷比例固定向切向力 −κ·f_n·x̂（κ=0.4）。纯力通道——kin 偏移只能改 f_n
           （法向）与经速度的摩擦，造不出固定方向切向力。
  M_kin:   足端几何偏移 δ=[0.02,0,−0.012] m（BODY）。纯运动学通道。
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from floating_base_srbd import FloatingBaseState, _matvec  # noqa: E402
from contact_3d import foot_contact_force_world  # noqa: E402
from srbd_standing import StandingConfig, foot_world  # noqa: E402

KAPPA = 0.4                                    # M_force 强度
KIN_OFF = (0.02, 0.0, -0.012)                  # M_kin 偏移 (BODY, m)
F_SCALE, K_SCALE = 60.0, 0.04                  # 头输出 tanh 限幅：N / m


# --------------------------------------------------------------------------- #
def foot_world_offset(state: FloatingBaseState, leg_ext: torch.Tensor,
                      cfg: StandingConfig, dx_body: torch.Tensor | None):
    """foot_world 的带体系偏移版：foot_w += R·dx，足速杠杆同步用偏移后的 r。"""
    R = quaternion_to_matrix(state.q)
    fr = cfg.foot_rel_com.unsqueeze(0).expand(state.p.shape[0], 4, 3).clone()
    fr[:, :, 2] = fr[:, :, 2] - leg_ext
    if dx_body is not None:
        fr = fr + dx_body
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, fr)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = state.v[:, None, :] + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
    return foot_w, foot_v


def accel(state: FloatingBaseState, leg_ext: torch.Tensor, cfg: StandingConfig,
          f_extra: torch.Tensor | None = None, dx_body: torch.Tensor | None = None):
    """一步广义加速度 (B,6)=[a_lin(WORLD), a_ang(BODY)]，含双通道注入。全程可微。"""
    foot_w, foot_v = foot_world_offset(state, leg_ext, cfg, dx_body)
    out = foot_contact_force_world(foot_w, foot_v, cfg.contact)
    f_each = out["f_world"] if f_extra is None else out["f_world"] + f_extra
    R = quaternion_to_matrix(state.q)
    r_world = foot_w - state.p[:, None, :]
    tau_world = torch.cross(r_world, f_each, dim=-1).sum(1)
    tau_body = torch.einsum("bji,bj->bi", R, tau_world)
    f_tot = f_each.sum(1)
    g = state.p.new_tensor([0.0, 0.0, -9.81])
    a_lin = g + f_tot / cfg.mass
    Iw = _matvec(cfg.I_body, state.w)
    a_ang = _matvec(cfg.I_body_inv, tau_body - torch.cross(state.w, Iw, dim=-1))
    return torch.cat([a_lin, a_ang], dim=-1)


# --------------------------------------------------------------------------- #
def mismatch(kind: str, state: FloatingBaseState, leg_ext: torch.Tensor,
             cfg: StandingConfig):
    """真实系统的已知失配 → (f_extra, dx_body)。"""
    if kind == "force":
        foot_w, foot_v = foot_world(state, leg_ext, cfg)
        f_n = foot_contact_force_world(foot_w, foot_v, cfg.contact)["f_n"]   # (B,4,1)
        xhat = state.p.new_tensor([1.0, 0.0, 0.0]).expand_as(foot_w)
        return -KAPPA * f_n * xhat, None
    if kind == "kin":
        dx = state.p.new_tensor(KIN_OFF).expand(state.p.shape[0], 4, 3)
        return None, dx
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
def head_obs(state: FloatingBaseState, leg_ext: torch.Tensor, cfg: StandingConfig):
    """头输入 (B,33)：体系重力方向(3)+体系线速度(3)+w(3)+leg_ext(4)+足相对COM体系(12)
    +gap(4)+足法向速度vn(4)。vn 必须给——f_n 含速度阻尼项 k_d·gate·srelu(−vn)，缺它力头
    表示不全（已踩过的坑）。全部由标称几何算出（不依赖头自身输出，无不动点回路）。"""
    R = quaternion_to_matrix(state.q)
    down = state.p.new_tensor([0.0, 0.0, -1.0])
    pg = torch.einsum("bji,j->bi", R, down)                       # R^T·(-ẑ)
    v_body = torch.einsum("bji,bj->bi", R, state.v)
    fr = cfg.foot_rel_com.unsqueeze(0).expand(state.p.shape[0], 4, 3).clone()
    fr[:, :, 2] = fr[:, :, 2] - leg_ext
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, fr)
    gap = foot_w[:, :, 2] - cfg.contact.ground_z                  # (B,4)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = state.v[:, None, :] + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
    vn = foot_v[:, :, 2]                                          # (B,4) 足法向速度
    return torch.cat([pg, v_body, state.w, leg_ext, fr.reshape(-1, 12), gap, vn], dim=-1)


class ResidualHead(nn.Module):
    """MLP(head_obs 33)→(B,4,3)。channel='force' 输出世界系力(N)，'kin' 输出体系偏移(m)。"""

    def __init__(self, channel: str, scale: float, in_dim: int = 33, hid: int = 64):
        super().__init__()
        assert channel in ("force", "kin")
        self.channel, self.scale = channel, scale
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ELU(),
                                 nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: FloatingBaseState, leg_ext: torch.Tensor,
                cfg: StandingConfig) -> torch.Tensor:
        out = self.scale * torch.tanh(self.net(head_obs(state, leg_ext, cfg)))
        return out.view(-1, 4, 3)

    def extras(self, state, leg_ext, cfg):
        out = self(state, leg_ext, cfg)
        return (out, None) if self.channel == "force" else (None, out)


class DualHead(nn.Module):
    """力头 + 运动学头（独立 MLP，零初始化末层 → 对称起步）。"""

    def __init__(self):
        super().__init__()
        self.fh = ResidualHead("force", F_SCALE)
        self.kh = ResidualHead("kin", K_SCALE)

    def extras(self, state, leg_ext, cfg):
        return self.fh(state, leg_ext, cfg), self.kh(state, leg_ext, cfg)


# --------------------------------------------------------------------------- #
def stack_states(pool: list[FloatingBaseState]) -> FloatingBaseState:
    return FloatingBaseState(*[torch.cat([getattr(s, k) for s in pool], 0)
                               for k in ("p", "q", "v", "w")])


def index_state(s: FloatingBaseState, i: int) -> FloatingBaseState:
    return FloatingBaseState(s.p[i:i + 1], s.q[i:i + 1], s.v[i:i + 1], s.w[i:i + 1])
