"""
组合残差 [R10]: 力 + 运动学双头, **自动把修正路由到正确的误差通道**
==========================================================================
R3–R9 收束出原理: 残差形式必须**对上误差类型**(力误差→接触力残差; 几何误差→落足残差)。
但真实迁移中**事先不知道**失配是力律(参数)还是几何(接触点)。R10 给残差**两个头**:
  · 力头(force):  R4 约束接触力残差(乘性有界法向 + 平滑摩擦锥), 修正"推多大力"
  · 运动学头(kin): 每足落足 x 偏置 Δx_k, 修正"踩在哪"
并让前向回归**自动分配**修正到该用的头。

三种 teacher(student 一律用 NOMINAL 参数 + 标称几何):
  T_force: 纯参数失配 REAL_BIG(m16/kn14000/μ0.45), 无几何偏  -> 只有力头能治
  T_geom : 纯几何失配(非对称剪刀 δx=0.35, 翻转力臂),  标称参数 -> 只有运动学头能治
  T_both : 参数 + 几何同时失配                                    -> 需两头并用

对照三种 student(同一 combined_accel, 只开不同头):
  C 力头   = combined(kin=None, force=√)   (= R4 接触力残差, 标称落足)
  D 运动学头= combined(kin=√, force=None)   (= R9 落足残差)
  E 双头   = combined(kin=√, force=√)       (自动路由)
预期 3×3: C 治力不治几何; D 治几何不治力; **E 三种 teacher 通吃**, 且头活跃度自动路由。
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import srbm_dynamics as srb
from srbm_dynamics import SRBMParams
from srbm_residual import NOMINAL, sample_XU
from srbm_residual_contact import ContactResidual
from srbm_residual_kinematic import KinematicResidual
from srbm_residual_struct import jac_of, cos_sign
from srbm_residual_extreme import accel_asym

C_C, C_D, C_E = "#55A868", "#DD8452", "#4C72B0"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
G = srb.G
EPS_T = 1e-6
ALPHA_N = 0.6
DGEOM = 0.35                                            # 几何失配(剪刀)大小, 过翻转点 0.25
REAL_BIG = SRBMParams(m=16.0, I=0.65, k_n=14000.0, k_d=65.0, mu=0.45)   # 参数失配(同 R6/R7)


def combined_accel(XU, p, kin=None, force=None, alpha_n=ALPHA_N, return_heads=False,
                   return_reg=False):
    """统一接触加速度: 可选运动学头(落足偏置 Δx) + 可选力头(R4 约束接触力残差)。

    return_reg=True 时额外返回**可微**的头输出平方均值 (kin_sq, force_sq) 供正则化。
    """
    px, pz, th = XU[:, 0], XU[:, 1], XU[:, 2]
    vx, vz, om = XU[:, 3], XU[:, 4], XU[:, 5]
    base_x = [p.half_len, -p.half_len]
    dfoot = kin(XU) if kin is not None else None
    Fx = torch.zeros_like(px); Fz = torch.zeros_like(px); tau = torch.zeros_like(px)
    kin_mag, force_mag, force_logits = [], [], []
    for k, bx in enumerate(base_x):
        ext_k = XU[:, 6 + k]
        dx = torch.zeros_like(px) if dfoot is None else dfoot[:, k]
        fx, fz = srb.foot_world(px, pz, th, ext_k, bx, p.leg_nominal, foot_dx=dx)
        rx, rz = fx - px, fz - pz
        vfx = vx - om * rz; vfz = vz + om * rx
        pen = p.eps * F.softplus(-fz / p.eps)
        gate = torch.sigmoid(-fz / p.eps)
        Fn_phys = torch.clamp(p.k_n * pen - p.k_d * vfz * gate, min=0.0)
        Ft_phys = -p.mu * Fn_phys * torch.tanh(vfx / p.v_eps)
        if force is not None:
            feat = torch.stack([pz, th, vx, vz, om, ext_k, fz, gate, vfz, vfx,
                                Fn_phys, Ft_phys], dim=-1)
            d = force.net(feat); dFn, dFt = d[:, 0], d[:, 1]
            Fn = Fn_phys * (1.0 + alpha_n * torch.tanh(dFn))      # 乘性有界法向
            muFn = p.mu * Fn
            Ft = muFn * torch.tanh((Ft_phys + dFt * gate) / (muFn + EPS_T))  # 平滑摩擦锥
            force_mag.append(((Fn - Fn_phys).abs() / (Fn_phys + 1e-6)))
            force_logits.append(d)
        else:
            Fn, Ft = Fn_phys, Ft_phys
            force_mag.append(torch.zeros_like(px))
        kin_mag.append(dx.abs())
        Fx = Fx + Ft; Fz = Fz + Fn; tau = tau + (rx * Fn - rz * Ft)
    A = torch.stack([Fx / p.m, Fz / p.m - G, tau / p.I], dim=-1)
    if return_reg:
        kin_sq = (dfoot ** 2).mean() if dfoot is not None else A.new_zeros(())
        force_sq = (torch.stack(force_logits, -1) ** 2).mean() if force_logits else A.new_zeros(())
        return A, kin_sq, force_sq
    if return_heads:
        heads = dict(kin=float(torch.stack(kin_mag, -1).mean()),
                     force=float(torch.stack(force_mag, -1).mean()))
        return A, heads
    return A


def make_student(kind, dev):
    """返回 (params_list, student_fn, heads_fn)。kind: C 力头 / D 运动学头 / E 双头。"""
    kin = KinematicResidual().to(dev).to(torch.get_default_dtype()) if kind in ("D", "E") else None
    force = ContactResidual().to(dev).to(torch.get_default_dtype()) if kind in ("C", "E") else None
    params = []
    if kin is not None: params += list(kin.parameters())
    if force is not None: params += list(force.parameters())
    student = lambda XU: combined_accel(XU, NOMINAL, kin=kin, force=force)
    heads = lambda XU: combined_accel(XU, NOMINAL, kin=kin, force=force, return_heads=True)[1]
    return params, student, heads


def scale_for_teacher(teacher_fn, dev):
    big = sample_XU(4096, dev)
    with torch.no_grad():
        nom = combined_accel(big, NOMINAL)
        s = (teacher_fn(big) - nom).std(0)
    return s.clamp_min(1e-6)


def train_student(kind, teacher_fn, n_iters, dev, scale, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    params, student, heads = make_student(kind, dev)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = teacher_fn(XU)
        loss = (((student(XU) - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return student, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 300 if args.quick else 1500
    print(f"device={dev} n_iters={n_iters} DGEOM={DGEOM}", flush=True)

    teachers = {
        "force": lambda XU: accel_asym(XU, REAL_BIG, 0.0),      # 纯参数失配
        "geom":  lambda XU: accel_asym(XU, NOMINAL, DGEOM),     # 纯几何失配(剪刀)
        "both":  lambda XU: accel_asym(XU, REAL_BIG, DGEOM),    # 两者
    }
    eval_XU = sample_XU(512, dev)
    res = {"fwd": {}, "cos": {}, "att": {}, "heads": {}}
    for tname, tfn in teachers.items():
        J_T = jac_of(tfn, eval_XU)
        sc = scale_for_teacher(tfn, dev)
        with torch.no_grad():
            AT = tfn(eval_XU)
        for kind in ["C", "D", "E"]:
            student, heads = train_student(kind, tfn, n_iters, dev, sc, seed=0)
            J = jac_of(student, eval_XU)
            with torch.no_grad():
                fwd = float(((student(eval_XU) - AT) ** 2).mean().sqrt())
            cos_full = cos_sign(J, J_T)[0]
            cos_att = cos_sign(J, J_T, rows=[2], cols=[6, 7])[0]
            res["fwd"].setdefault(kind, {})[tname] = fwd
            res["cos"].setdefault(kind, {})[tname] = cos_full
            res["att"].setdefault(kind, {})[tname] = cos_att
            if kind == "E":
                res["heads"][tname] = heads(eval_XU)
            print(f"  T={tname:6s} S={kind}: fwd={fwd:7.3f} cosFull={cos_full:+.3f} cosAtt={cos_att:+.3f}",
                  flush=True)
    json.dump(res, open(os.path.join(HERE, "results_phase3i.json"), "w"), indent=2, ensure_ascii=False)
    _plot(res, list(teachers))
    print("[R10] figure saved.", flush=True)


def _plot(res, tnames):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    kinds = ["C", "D", "E"]; labels = ["C force-head", "D kinematic-head", "E both (auto-route)"]
    cols = [C_C, C_D, C_E]
    x = np.arange(len(tnames)); w = 0.26
    # (a) 前向 RMS (越低越好)
    for j, kind in enumerate(kinds):
        ax[0].bar(x + (j - 1) * w, [res["fwd"][kind][t] for t in tnames], w, color=cols[j], label=labels[j])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"T_{t}" for t in tnames])
    ax[0].set_yscale("log"); ax[0].set_ylabel("forward RMS (log, lower=better)")
    ax[0].set_title("(a) forward fit per teacher type"); ax[0].legend(fontsize=7)
    # (b) 全梯度余弦 (越高越好)
    for j, kind in enumerate(kinds):
        ax[1].bar(x + (j - 1) * w, [res["cos"][kind][t] for t in tnames], w, color=cols[j], label=labels[j])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"T_{t}" for t in tnames])
    ax[1].set_ylabel("full-gradient cosine to teacher"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("(b) gradient fidelity per teacher type\n(E good on ALL; C/D each fail one)")
    ax[1].legend(fontsize=7)
    # (c) E 的头活跃度 (自动路由): 各 teacher 下 kin 头 vs force 头 (各自归一到自身最大)
    kin_raw = np.array([res["heads"][t]["kin"] for t in tnames])
    frc_raw = np.array([res["heads"][t]["force"] for t in tnames])
    kin_n = kin_raw / (kin_raw.max() + 1e-12)
    frc_n = frc_raw / (frc_raw.max() + 1e-12)
    ax[2].bar(x - w / 2, kin_n, w, color=C_D, label="kinematic head |Δx|")
    ax[2].bar(x + w / 2, frc_n, w, color=C_C, label="force head |ΔFn|/Fn")
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"T_{t}" for t in tnames])
    ax[2].set_ylabel("E head activity (each normalized to its max)")
    ax[2].set_title("(c) E auto-routes: geom→kin head,\nforce→force head, both→both")
    ax[2].legend(fontsize=7)
    fig.suptitle("R10  Combined residual auto-routes correction to the right error channel "
                 "(force vs kinematic)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R10_combined_residual.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
