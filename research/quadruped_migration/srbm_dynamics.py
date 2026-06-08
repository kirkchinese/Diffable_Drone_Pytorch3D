"""
平面单刚体模型 (Planar Single Rigid Body Model, SRBM) —— 第二阶段迁移载体
==========================================================================

在 Hopper(第一阶段)的基础上,加入**机身姿态**与**切向摩擦**,得到能反映四足核心
动力学结构、又仍足够简单可微的最小模型。对应 Song 2024 的浮动基 SRBD 的 2D 退化。

为什么是这个模型
----------------
它**最小地**保留了四足区别于无人机点质量的两个本质特征:
  1. **姿态由接触力矩产生**: I·θ̈ = Σ rᵢ × Fᵢ —— 不能像无人机那样代数反解(微分平坦性丢失);
  2. **切向摩擦**: 前向推进只能靠地面摩擦,接触切换 + 摩擦锥引入新的非光滑。

状态 / 控制
----------
  state x = [px, pz, θ, vx, vz, ω]   (机身质心位置 / 俯仰角 / 线速度 / 角速度), 支持 batch (B,)
  control u = 每条腿的伸长量 ext_k     (棱柱腿; 差动伸长 -> 俯仰力矩 -> 姿态控制)

接触(penalty)
-------------
  足端 = 机身经 R(θ) 投影 + 腿伸长。足端穿地 -> 平滑/硬法向力(复用第一阶段) + 平滑库仑摩擦。
  摩擦: f_t = -μ·f_n·tanh(v_foot_x / v_eps)   (可微、有界、过零点光滑)

约定: x 水平, z 向上, θ 逆时针为正; R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]] body->world。
复用无人机 g_decay(同 [[quadruped-migration-research]] 第一阶段)。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from slip_dynamics import g_decay  # noqa: E402  (后者已复用无人机 drone_dynamics.g_decay)

G = 9.81


@dataclass(frozen=True)
class SRBMParams:
    m: float = 10.0          # 机身质量 (kg)
    I: float = 0.35          # 绕 y 的转动惯量 (kg·m²)
    half_len: float = 0.25   # 前/后足在机身 x 方向的偏移 ±a (m)
    leg_nominal: float = 0.32  # 腿标称长度 h0 (m, 机身下方)
    k_n: float = 6000.0      # 法向接触刚度
    k_d: float = 30.0        # 法向接触阻尼
    eps: float = 0.01        # 平滑接触宽度
    mu: float = 0.9          # 摩擦系数
    v_eps: float = 0.05      # 摩擦平滑速度尺度 (m/s)


def _R(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return c, s  # 返回分量, 手写 2x2 旋转以省内存


def foot_world(px, pz, theta, ext, base_x, leg_nominal, foot_dx=0.0):
    """单足世界坐标。foot_local = (base_x + foot_dx, -(leg_nominal+ext))，经 R(θ) 投影 + 平移。

    foot_dx: 足端在机身 x 方向的额外前后偏移(用于摆腿/落足控制; F1-F3 默认 0 不受影响)。
    """
    c, s = _R(theta)
    lx = base_x + foot_dx
    lz = -(leg_nominal + ext)
    fx = px + c * lx - s * lz
    fz = pz + s * lx + c * lz
    return fx, fz


def contact_force_2d(fx, fz, vfx, vfz, model, p: SRBMParams):
    """足端 penalty 接触力 (世界系)。返回 (Fcx, Fcz)。fz<0 表示穿地。"""
    if model == "smooth":
        pen = p.eps * F.softplus(-fz / p.eps)
        gate = torch.sigmoid(-fz / p.eps)
    else:  # 'soft' / 'hard'(用更大 k_n)
        pen = torch.relu(-fz)
        gate = (fz < 0).to(fz.dtype)
    f_n = torch.clamp(p.k_n * pen - p.k_d * vfz * gate, min=0.0)   # 法向(向上)
    # 平滑库仑摩擦: 有界 |f_t|<=μ f_n, 过零光滑
    f_t = -p.mu * f_n * torch.tanh(vfx / p.v_eps)
    return f_t, f_n


def srbm_accel(state, ext, model, p: SRBMParams, base_x, foot_dx=None):
    """给定状态与腿伸长,算机身加速度 (ax, az, alpha) 与诊断量。

    foot_dx: 可选, 每足在机身 x 方向的额外偏置(列表, 每足一个)。
             用于结构性失配实验(teacher 足端偏移); 默认 None -> 0, 不影响 F1-F4。
    """
    px, pz, th, vx, vz, om = state
    Fx = torch.zeros_like(px)
    Fz = torch.zeros_like(px)
    tau = torch.zeros_like(px)
    diag = {}
    for k, bx in enumerate(base_x):
        e = ext[k]
        dx = 0.0 if foot_dx is None else foot_dx[k]
        fx, fz = foot_world(px, pz, th, e, bx, p.leg_nominal, dx)
        rx, rz = fx - px, fz - pz                      # 力臂 r = foot - com
        vfx = vx - om * rz                             # v_foot = v_com + ω×r ; (ω×r)=(-ω rz, ω rx)
        vfz = vz + om * rx
        Fcx, Fcz = contact_force_2d(fx, fz, vfx, vfz, model, p)
        Fx = Fx + Fcx
        Fz = Fz + Fcz
        tau = tau + (rx * Fcz - rz * Fcx)              # 2D 力矩 = r_x F_z - r_z F_x
        diag[f"foot{k}_z"] = fz
        diag[f"foot{k}_fn"] = Fcz
    ax = Fx / p.m
    az = Fz / p.m - G
    alpha = tau / p.I
    diag["tau"] = tau
    return ax, az, alpha, diag


def rollout_srbm(state0, ext_seq, model="smooth", p: SRBMParams = SRBMParams(),
                 base_x=None, dt=1e-3, grad_decay=1.0):
    """可微 SRBM rollout(半隐式欧拉)。

    state0: tuple of 6 个标量/张量 (px,pz,θ,vx,vz,ω)。
    ext_seq: (n_steps, n_legs, ...) 腿伸长序列(每步每腿)。
    返回 traj dict: 'px','pz','th','vx','vz','om' 各 (n_steps+1, ...), 以及 'tau'。
    """
    if base_x is None:
        base_x = [p.half_len, -p.half_len]
    px, pz, th, vx, vz, om = state0
    n = ext_seq.shape[0]
    out = {kk: [v] for kk, v in zip(["px", "pz", "th", "vx", "vz", "om"],
                                    [px, pz, th, vx, vz, om])}
    taus = []
    decay = grad_decay ** dt
    for t in range(n):
        if grad_decay != 1.0:
            px, pz, th = g_decay(px, decay), g_decay(pz, decay), g_decay(th, decay)
            vx, vz, om = g_decay(vx, decay), g_decay(vz, decay), g_decay(om, decay)
        ext_t = [ext_seq[t, k] for k in range(len(base_x))]
        ax, az, al, diag = srbm_accel((px, pz, th, vx, vz, om), ext_t, model, p, base_x)
        vx = vx + dt * ax
        vz = vz + dt * az
        om = om + dt * al
        px = px + dt * vx
        pz = pz + dt * vz
        th = th + dt * om
        for kk, v in zip(["px", "pz", "th", "vx", "vz", "om"], [px, pz, th, vx, vz, om]):
            out[kk].append(v)
        taus.append(diag["tau"])
    out = {kk: torch.stack(v) for kk, v in out.items()}
    out["tau"] = torch.stack(taus)
    return out


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    p = SRBMParams()

    # 自检1: 对称落地应几乎不俯仰; 差动腿伸长应产生俯仰力矩
    def drop(ext0, ext1, th0=0.0):
        s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.45, device=dev),
              torch.tensor(th0, device=dev), torch.tensor(0.0, device=dev),
              torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev))
        seq = torch.zeros(3000, 2, device=dev)
        seq[:, 0] = ext0; seq[:, 1] = ext1
        return rollout_srbm(s0, seq, model="smooth", p=p, dt=1e-3)

    tr_sym = drop(0.0, 0.0)
    tr_asym = drop(0.05, 0.0)   # 前腿伸长 -> 前侧抬高 -> 俯仰
    print(f"[self-check] symmetric drop  final |θ| = {abs(float(tr_sym['th'][-1])):.4f} rad (应≈0)")
    print(f"[self-check] asym leg-ext    final  θ  = {float(tr_asym['th'][-1]):+.4f} rad (应≠0, 接触力矩致俯仰)")
    print(f"[self-check] settle height   pz = {float(tr_sym['pz'][-1]):.3f} m")

    # 自检2: 可微性 + 姿态对腿伸长有梯度(可微平坦性丢失的反面: θ 确实是被积分出来且可控)
    e = torch.zeros(2000, 2, device=dev, requires_grad=True)
    s0 = (torch.tensor(0.0, device=dev), torch.tensor(0.40, device=dev),
          torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev),
          torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev))
    tr = rollout_srbm(s0, e, model="smooth", p=p, dt=1e-3)
    loss = (tr["th"][-1] - 0.1) ** 2
    loss.backward()
    print(f"[self-check] dLoss/d(ext) norm = {float(e.grad.norm()):.4f} (姿态可由腿伸长经接触力矩可微地控制)")
