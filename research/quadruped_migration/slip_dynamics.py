"""
可微 SLIP / 垂直 Hopper 最小动力学模块
========================================

无人机 -> 四足 可微物理迁移研究 · 第一阶段最小实验载体。

目的
----
在**最小混合系统**（飞行相 <-> 支撑相）上，干净地隔离"接触如何破坏一阶梯度
(First-order Gradient, FoG)"，并与无人机式**光滑、可微平坦**动力学做对照。

为什么用垂直 Hopper
-------------------
四足机器人区别于无人机点质量动力学的**最小不可约特征**就是"足端接触切换"。
一个 1 自由度垂直弹跳质点（SLIP 的退化形式）已经包含：
  - 飞行相: ÿ = -g           (光滑，与无人机自由落体一致)
  - 支撑相: ÿ = -g + f_n/m    (接触力进入)
  - 相切换: y 穿过地面时发生  (非光滑事件 / 混合系统)
而**不引入**多关节、姿态、摩擦等额外复杂度，因此最适合做"病理隔离实验"。

三种接触法向力模型（对应文献）
------------------------------
对应 Bagajo & Schwarke (DiffSim2Real, CoRL-W 2024) Fig.2 与 Xu (SHAC, ICLR 2022) Eq.3：
  - 'hard'   : 阶跃接触力，解析梯度几乎处处为 0   -> FoG 无信息 (uninformative)
  - 'soft'   : 线性罚 (Kelvin-Voigt)，连续但有折点；刚度大时力无界 -> 数值不稳
  - 'smooth' : softplus 平滑罚，C∞ 且梯度有界有信息 -> analytic smoothing 思想

设计约束
--------
- 所有 rollout 函数 torch 可微，支持 batch 维 (B,) 并行 -> GPU 友好的景观扫描。
- 复用无人机项目的 ``g_decay``（梯度衰减自定义 autograd），用来在**同一份代码**上
  演示 Song 2024 的"状态对齐 α 衰减" 与 无人机 GDecay 是同一机制。

作者: 迁移研究 notebook 配套模块
日期: 2026-06-04
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

# ---- 复用无人机项目的梯度衰减自定义 autograd（演示"同一机制"用） ----
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from drone_dynamics import g_decay as _drone_g_decay  # noqa: E402

    HAVE_DRONE_GDECAY = True
except Exception:  # pragma: no cover - 退化为本地等价实现
    HAVE_DRONE_GDECAY = False

    class _GDecay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, param):
            ctx.param = param
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * ctx.param, None

    def _drone_g_decay(x, param):
        return _GDecay.apply(x, param)


def g_decay(x: torch.Tensor, param: float) -> torch.Tensor:
    """逐步梯度衰减：前向恒等，反向把梯度乘 ``param``（与无人机 GDecay 一致）。"""
    return _drone_g_decay(x, param)


G = 9.81  # 重力加速度 (m/s^2)


# =====================================================================
# 1. 接触力定律（用于"力-穿透深度"概念图，复现 DiffSim2Real Fig.2）
# =====================================================================
def contact_force_law(
    d: torch.Tensor,
    model: str,
    F0: float = 50.0,
    k: float = 4000.0,
    eps: float = 0.01,
    sigma: float = 0.02,
):
    """法向接触力 f_n 及其**解析梯度** df_n/dd 作为穿透深度 d 的函数。

    d > 0 表示穿透（接触中），d < 0 表示分离（无接触）。返回 (f_n, df_n/dd)。

    四种处理（Bagajo & Schwarke 2024, Fig.2）：
      - 'hard'  : f = F0·H(d)              -> 阶跃；解析梯度几乎处处 0（仅 d=0 处为 Dirac）
      - 'soft'  : f = relu(k·d)            -> 线性罚；梯度为阶跃 k·H(d)，力随 d 无界
      - 'stoch' : f = E_ξ[F0·H(d+ξ)]       -> 随机平滑（期望力光滑 = F0·Φ(d/σ)），
                  但**逐样本一阶梯度仍为 0**（每个样本看到的还是硬接触）-> FoG 无信息
      - 'smooth': f = F0·σ(d/eps)          -> 解析平滑；梯度 = (F0/eps)·σ'(d/eps) 为有界钟形
    """
    if model == "hard":
        f = F0 * (d > 0).to(d.dtype)
        grad = torch.zeros_like(d)
    elif model == "soft":
        f = torch.relu(k * d)
        grad = k * (d > 0).to(d.dtype)
    elif model == "stoch":
        # 期望力（对硬接触做高斯随机平滑后的均值）：光滑
        f = F0 * 0.5 * (1.0 + torch.erf(d / (sigma * math.sqrt(2.0))))
        # 关键点：逐样本一阶梯度（FoG）仍为 0（阶跃处处导数为 0），故 FoG 无信息
        grad = torch.zeros_like(d)
    elif model == "smooth":
        s = torch.sigmoid(d / eps)
        f = F0 * s
        grad = (F0 / eps) * s * (1.0 - s)
    else:
        raise ValueError(f"unknown contact force model: {model}")
    return f, grad


# =====================================================================
# 2. 垂直 Hopper 可微仿真（用于景观 / BPTT / 偏差实验）
# =====================================================================
def hopper_normal_force(
    y: torch.Tensor,
    ydot: torch.Tensor,
    model: str = "smooth",
    k_n: float = 4000.0,
    k_d: float = 12.0,
    eps: float = 0.01,
) -> torch.Tensor:
    """Hopper 法向力（向上为正）。质点高度 y，y<0 表示穿透地面（penetration = -y）。

    Kelvin-Voigt 弹簧阻尼：f = k_n·penetration - k_d·ydot·gate，clamp(>=0)。
    阻尼项提供能量耗散 -> 恢复系数 < 1 -> 真实弹跳。

      - 'soft' / 'stiff' : penetration = relu(-y)，gate = 1[y<0]（含折点 / 非光滑）
      - 'smooth'         : penetration = eps·softplus(-y/eps)，gate = sigmoid(-y/eps)（C∞）
    """
    if model == "smooth":
        pen = eps * F.softplus(-y / eps)
        gate = torch.sigmoid(-y / eps)
    else:  # 'soft' / 'stiff'（'stiff' 仅用更大的 k_n）
        pen = torch.relu(-y)
        gate = (y < 0).to(y.dtype)
    f = k_n * pen - k_d * ydot * gate
    return torch.clamp(f, min=0.0)


def rollout_hopper(
    y0: torch.Tensor,
    v0: torch.Tensor,
    n_steps: int,
    dt: float = 1e-3,
    model: str = "smooth",
    k_n: float = 4000.0,
    k_d: float = 12.0,
    eps: float = 0.01,
    m: float = 1.0,
    grad_decay: float = 1.0,
):
    """可微 Hopper rollout（半隐式欧拉）。

    y0, v0: 标量或 (B,) batch。返回 (Y, V)，形状 (n_steps+1, ...)。
    grad_decay < 1 时，对 (y, v) 逐步施加 g_decay（衰减因子 grad_decay**dt），
    与无人机训练一致地驯服 BPTT 梯度幅值。
    """
    y, v = y0, v0
    Y, V = [y], [v]
    decay = grad_decay ** dt
    for _ in range(n_steps):
        if grad_decay != 1.0:
            y = g_decay(y, decay)
            v = g_decay(v, decay)
        f = hopper_normal_force(y, v, model=model, k_n=k_n, k_d=k_d, eps=eps)
        a = -G + f / m
        v = v + dt * a       # 半隐式欧拉：先更新速度
        y = y + dt * v       # 再用新速度更新位置
        Y.append(y)
        V.append(v)
    return torch.stack(Y), torch.stack(V)


# =====================================================================
# 3. 无人机式光滑点质量（对照基线：可微平坦、无接触）
# =====================================================================
def rollout_smooth_pointmass(
    y0: torch.Tensor,
    v0: torch.Tensor,
    u_cmd: torch.Tensor,
    n_steps: int,
    dt: float = 0.02,
    ctl_delay: float = 12.0,
    grad_decay: float = 1.0,
):
    """垂直点质量，净加速度指令 u_cmd（重力已折叠，与无人机 act_cmd 同约定）。

    一阶执行器低通 a <- u_cmd·(1-α) + a·α，α = exp(-ctl_delay·dt)，完全光滑。
    返回 Y，形状 (n_steps+1, ...)。这是"无接触光滑动力学"的参照系。
    """
    y, v = y0, v0
    a = torch.zeros_like(y0)
    alpha = math.exp(-ctl_delay * dt)
    Y = [y]
    decay = grad_decay ** dt
    for _ in range(n_steps):
        if grad_decay != 1.0:
            y = g_decay(y, decay)
            v = g_decay(v, decay)
        a = u_cmd * (1.0 - alpha) + a * alpha
        v = v + dt * a
        y = y + dt * v
        Y.append(y)
    return torch.stack(Y)


# =====================================================================
# 4. 辅助：成本、解析/有限差分梯度、景观粗糙度
# =====================================================================
def terminal_height_cost(Y: torch.Tensor, y_target: float) -> torch.Tensor:
    """终端高度跟踪成本 (y_T - y*)^2，对 batch 取每个样本（返回 (B,) 或标量）。"""
    return (Y[-1] - y_target) ** 2


def analytic_grad_wrt_v0(
    cost_fn,
    v0_value: float,
    device: torch.device,
    dtype=torch.float64,
) -> float:
    """对单个 v0 用 autograd 求 dCost/dv0（一阶解析梯度 / FoG）。"""
    v0 = torch.tensor(v0_value, device=device, dtype=dtype, requires_grad=True)
    cost = cost_fn(v0)
    (grad,) = torch.autograd.grad(cost, v0)
    return float(grad.item())


def finite_diff_grad_wrt_v0(
    cost_fn,
    v0_value: float,
    device: torch.device,
    h: float = 1e-4,
    dtype=torch.float64,
) -> float:
    """中心有限差分求 dCost/dv0（作为"真实"梯度的参照，含接触时序变化的真实效应）。"""
    with torch.no_grad():
        vp = torch.tensor(v0_value + h, device=device, dtype=dtype)
        vm = torch.tensor(v0_value - h, device=device, dtype=dtype)
        cp = cost_fn(vp)
        cm = cost_fn(vm)
    return float((cp - cm).item()) / (2.0 * h)


def landscape_roughness(cost_curve: torch.Tensor) -> float:
    """景观"非光滑度"指标：**三阶**差分的 L1 范数 Σ|Δ³cost|。

    选三阶而非二阶是为了**尺度无关地**剥离"光滑但高曲率"的混淆：
    对任意二次曲线（光滑大抛物线）Δ³≡0，而折点 / 尖峰 / 接触时序突变会让 Δ³ 爆发。
    因此该指标对"光滑基线"≈0，对"接触锯齿景观"很大。
    """
    d3 = cost_curve[3:] - 3.0 * cost_curve[2:-1] + 3.0 * cost_curve[1:-2] - cost_curve[:-3]
    return float(d3.abs().sum().item())


if __name__ == "__main__":
    # 自检：硬接触梯度≈0、光滑接触梯度有界；光滑基线解析=有限差分
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # (a) 力定律梯度对照
    d = torch.linspace(-0.05, 0.05, 11, device=dev)
    for mdl in ["hard", "soft", "stoch", "smooth"]:
        f, gr = contact_force_law(d, mdl)
        print(f"[force_law:{mdl:6s}] grad(min,max)=({gr.min():.2f},{gr.max():.2f})")

    # (b) Hopper rollout 可微性自检
    y0 = torch.tensor(1.0, device=dev, requires_grad=True)
    v0 = torch.tensor(0.0, device=dev, requires_grad=True)
    Y, V = rollout_hopper(y0, v0, n_steps=2000, dt=1e-3, model="smooth")
    cost = (Y[-1] - 0.5) ** 2
    cost.backward()
    print(f"[hopper smooth] y_end={Y[-1].item():.3f} dCost/dy0={y0.grad.item():.4f}")

    # (c) 光滑基线自检
    def cost_fn(v0v):
        Yb = rollout_smooth_pointmass(
            torch.tensor(1.0, device=dev, dtype=v0v.dtype), v0v,
            torch.tensor(0.0, device=dev, dtype=v0v.dtype), n_steps=200, dt=0.02,
        )
        return (Yb[-1] - 0.0) ** 2

    ga = analytic_grad_wrt_v0(cost_fn, 1.0, dev)
    gf = finite_diff_grad_wrt_v0(cost_fn, 1.0, dev)
    print(f"[smooth baseline] analytic={ga:.5f} finite_diff={gf:.5f} (应几乎相等)")
    print("self-check done. HAVE_DRONE_GDECAY =", HAVE_DRONE_GDECAY)
