"""步态任务上的双通道残差基建（E3D-4b）：一步加速度 + 步态头观测 + 失配。

与 E3D-6/7 同一套方法论搬到 trot：失配定义不变（M_force=−κ·f_n·x̂ 载荷比例切向力，
M_kin=足端几何偏移 δ），hook 走 gait_3d.gait_step 的 f_extra/dx_body。
步态下的关键区别：纵向力失配直接打击速度跟踪、且 ΔLx 有对抗权限（可控通道）——
E3D-7"失配×可控子空间交集"机制预测闭环对比在此变可判。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix

from floating_base_srbd import FloatingBaseState, _matvec
from contact_3d import foot_contact_force_world
from srbd_standing import StandingConfig
from gait_3d import GaitConfig, foot_plan

KAPPA = 0.4
KIN_OFF = (0.02, 0.0, -0.012)
F_SCALE, K_SCALE = 60.0, 0.04


def _foot_world_v(state, t_step, a, cfg, g, dx_body=None):
    p_b, pdot_b, stance, phi = foot_plan(t_step, a, cfg, g)
    if dx_body is not None:
        p_b = p_b + dx_body
    R = quaternion_to_matrix(state.q)
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, p_b)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = (state.v[:, None, :]
              + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
              + torch.einsum("bij,bkj->bki", R, pdot_b))
    return foot_w, foot_v, p_b, R


def gait_accel(state: FloatingBaseState, t_step: int, a: torch.Tensor,
               cfg: StandingConfig, g: GaitConfig,
               f_extra=None, dx_body=None):
    """一步广义加速度 (B,6)=[a_lin WORLD, a_ang BODY]，含双通道注入。全程可微。"""
    foot_w, foot_v, _, R = _foot_world_v(state, t_step, a, cfg, g, dx_body)
    out = foot_contact_force_world(foot_w, foot_v, cfg.contact)
    f_each = out["f_world"] if f_extra is None else out["f_world"] + f_extra
    r = foot_w - state.p[:, None, :]
    tau_body = torch.einsum("bji,bj->bi", R, torch.cross(r, f_each, dim=-1).sum(1))
    grav = state.p.new_tensor([0.0, 0.0, -9.81])
    a_lin = grav + f_each.sum(1) / cfg.mass
    Iw = _matvec(cfg.I_body, state.w)
    a_ang = _matvec(cfg.I_body_inv, tau_body - torch.cross(state.w, Iw, dim=-1))
    return torch.cat([a_lin, a_ang], dim=-1)


def gait_mismatch(kind: str, state, t_step, a, cfg, g,
                  kappa: float = KAPPA, kin_off=KIN_OFF):
    """真实系统失配 → (f_extra, dx_body)。默认与 E3D-6/7 完全一致；
    kappa/kin_off 可调供强度扫描（E3D-4c 统计加固）。"""
    if kind == "force":
        foot_w, foot_v, _, _ = _foot_world_v(state, t_step, a, cfg, g)
        f_n = foot_contact_force_world(foot_w, foot_v, cfg.contact)["f_n"]
        xhat = state.p.new_tensor([1.0, 0.0, 0.0]).expand_as(foot_w)
        return -kappa * f_n * xhat, None
    if kind == "kin":
        return None, state.p.new_tensor(kin_off).expand(state.p.shape[0], 4, 3)
    raise ValueError(kind)


def gait_head_obs(state, t_step, a, cfg, g):
    """残差头输入 (B,29)：体系重力(3)+v_b(3)+w(3)+足体系位置(12)+gap(4)+vn(4)。
    由标称几何算出（不依赖头输出），含 f_n 所需的 gap+vn 信息。"""
    foot_w, foot_v, p_b, R = _foot_world_v(state, t_step, a, cfg, g)
    pg = torch.einsum("bji,j->bi", R, state.p.new_tensor([0.0, 0.0, -1.0]))
    v_b = torch.einsum("bji,bj->bi", R, state.v)
    gap = foot_w[:, :, 2] - cfg.contact.ground_z
    vn = foot_v[:, :, 2]
    return torch.cat([pg, v_b, state.w, p_b.reshape(-1, 12), gap, vn], dim=-1)


class GaitResidualHead(nn.Module):
    def __init__(self, channel: str, scale: float, in_dim: int = 29, hid: int = 64):
        super().__init__()
        assert channel in ("force", "kin")
        self.channel, self.scale = channel, scale
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ELU(),
                                 nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, t_step, a, cfg, g):
        out = self.scale * torch.tanh(self.net(gait_head_obs(state, t_step, a, cfg, g)))
        return out.view(-1, 4, 3)


class GaitDualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fh = GaitResidualHead("force", F_SCALE)
        self.kh = GaitResidualHead("kin", K_SCALE)

    def extras(self, state, t_step, a, cfg, g):
        return self.fh(state, t_step, a, cfg, g), self.kh(state, t_step, a, cfg, g)


class StructuredDual(nn.Module):
    """结构化参数残差（E3D-4b 收官判别实验）：力头只学标量 κ̂（基底 −f_n·x̂），
    运动学头只学常量 δ̂∈R³。共 4 个参数 vs 自由 MLP 的 ~万级——若它兑现收益而
    MLP 头不能，则败因=头的拟合/梯度质量，非修正概念；同时复刻 2D"结构化 vs
    神经残差"边界（结构对了 4 参数胜过自由网络）。"""

    def __init__(self):
        super().__init__()
        self.kappa = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(3))

    def extras(self, state, t_step, a, cfg, g):
        foot_w, foot_v, _, _ = _foot_world_v(state, t_step, a, cfg, g)
        f_n = foot_contact_force_world(foot_w, foot_v, cfg.contact)["f_n"]
        xhat = state.p.new_tensor([1.0, 0.0, 0.0]).expand_as(foot_w)
        fe = -self.kappa * f_n * xhat
        dx = self.delta.expand(state.p.shape[0], 4, 3)
        return fe, dx
