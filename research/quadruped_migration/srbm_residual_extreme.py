"""
极端几何失配 [R8]: 非对称"剪刀"足端偏置 —— 真正翻转**差动控制力臂**符号
==========================================================================
R5 只测温和**对称**足偏(δx=0.06)。但**对称平移** fd=[δx,δx] 同时移动两脚, 差动
平衡控制力臂差 r_x0−r_x1=(¼+δx)−(−¼+δx)=2·half_len **恒定不变** -> 不翻转
∂θ/∂(差动伸长)(已数值验证: 对称下姿态梯度仅缓降不变号), 不是最狠几何失配。

R8 用**非对称剪刀偏置** fd=[−δx, +δx](前脚后移、后脚前移):
    r_x0 = +half_len − δx,  r_x1 = −half_len + δx = −r_x0
    => 差动控制增益 g_diff ≡ ∂α/∂ext0 − ∂α/∂ext1 ∝ r_x0 − r_x1 = 2(half_len − δx)
       **在 δx=half_len(0.25) 处过零并反号** -> δx>0.25 两脚交叉、平衡控制方向整体翻转。

核心检验(原假设, 反 R5"结构总保护梯度"): C(约束接触力残差)把标称力臂 r_x=±half_len
**焊死在结构里**(残差只缩放 F_n ±60%, 改不了力臂) -> δx>0.25 真实力臂翻转时 C 结构上
**无法翻转控制方向** -> 应与 nominal 一同崩。B(自由加速度残差)直接输出 Δα, 原则上能
重拟合翻转后的动力学 -> 极端区可能**反超** A/C。
=> 预期**交叉**: 小 δx 时 C 最优(R3–R7); 过 0.25 后 B 最优、C 最差。
=> **物理结构价值"有界": 仅当其编码几何近似正确时保护梯度; 失配大到翻转力臂,
   焊死的力臂从"归纳偏置"沦为"错误先验"。**

三指标: g_diff(差动控制增益, 看符号翻转) / 梯度余弦保真 / 闭环平衡部署 loss(看谁先崩)。
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import srbm_dynamics as srb
from srbm_residual import NOMINAL, sample_XU, Residual as AccelResidual
from srbm_residual_constrained import contact_accel_c, ContactResidual
from srbm_residual_struct import jac_of, cos_sign
from srbm_residual_closedloop import train_policy, evaluate

C_NOM, C_B, C_C, C_TEA = "#8C8C8C", "#C44E52", "#55A868", "#4C72B0"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
HALF = NOMINAL.half_len   # 0.25 = 差动力臂过零的临界 δx


def accel_asym(XU, p, dx, model="smooth"):
    """非对称剪刀足偏 fd=[−dx,+dx] 的加速度。dx=0 即标称(无偏)。"""
    state = (XU[:, 0], XU[:, 1], XU[:, 2], XU[:, 3], XU[:, 4], XU[:, 5])
    ext = [XU[:, 6], XU[:, 7]]
    fd = [-float(dx), float(dx)]
    ax, az, al, _ = srb.srbm_accel(state, ext, model, p, [p.half_len, -p.half_len], foot_dx=fd)
    return torch.stack([ax, az, al], dim=-1)


def nominal_accel(XU):
    return accel_asym(XU, NOMINAL, 0.0)


def diff_gain(J):
    """差动姿态控制增益 g_diff = mean(∂α/∂ext0 − ∂α/∂ext1)。符号=平衡控制方向。"""
    return float((J[:, 2, 6] - J[:, 2, 7]).mean())


def train_resid(kind, n_iters, dev, scale, dx, seed=0, lr=2e-3):
    """拟合非对称 teacher(dx)。kind: 'B'=自由加速度残差 / 'C'=R4约束接触力残差。"""
    torch.manual_seed(seed)
    teacher = lambda XU: accel_asym(XU, NOMINAL, dx)
    if kind == "B":
        r = AccelResidual().to(dev).to(torch.get_default_dtype())
        student = lambda XU: nominal_accel(XU) + r(XU)
    else:  # C
        r = ContactResidual().to(dev).to(torch.get_default_dtype())
        student = lambda XU: contact_accel_c(XU, NOMINAL, r, gated=True, cone=True, alpha_n=0.6)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    for _ in range(n_iters):
        XU = sample_XU(512, dev)
        with torch.no_grad():
            A_T = teacher(XU)
        loss = (((student(XU) - A_T) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return student


def scale_for(dx, dev):
    big = sample_XU(4096, dev)
    with torch.no_grad():
        s = (accel_asym(big, NOMINAL, dx) - nominal_accel(big)).std(0)
    return s.clamp_min(1e-6)


def part1_gradient_sweep(dev, n_iters, n_pts):
    eval_XU = sample_XU(512, dev)
    deltas = np.linspace(-0.45, 0.45, n_pts)
    gain = {k: [] for k in ["teacher", "A", "B", "C"]}
    cos = {k: {"att": [], "full": []} for k in ["A", "B", "C"]}
    for dx in deltas:
        teacher_fn = lambda XU, d=dx: accel_asym(XU, NOMINAL, float(d))
        J_T = jac_of(teacher_fn, eval_XU)
        sc = scale_for(float(dx), dev)
        stB = train_resid("B", n_iters, dev, sc, float(dx), seed=1)
        stC = train_resid("C", n_iters, dev, sc, float(dx), seed=1)
        gain["teacher"].append(diff_gain(J_T))
        for key, fn in [("A", nominal_accel), ("B", stB), ("C", stC)]:
            J = jac_of(fn, eval_XU)
            gain[key].append(diff_gain(J))
            cos[key]["att"].append(cos_sign(J, J_T, rows=[2], cols=[6, 7])[0])
            cos[key]["full"].append(cos_sign(J, J_T)[0])
        print(f"  δx={dx:+.3f}  g_diff T={gain['teacher'][-1]:+.2f} A={gain['A'][-1]:+.2f} "
              f"B={gain['B'][-1]:+.2f} C={gain['C'][-1]:+.2f} | att-cos B={cos['B']['att'][-1]:.2f} "
              f"C={cos['C']['att'][-1]:.2f}", flush=True)
    return deltas.tolist(), gain, cos


def part2_closed_loop(dev, dxs, res_iters, pol_iters):
    rows = {k: {"own": [], "deploy": []} for k in ["nominal", "free-res", "constr-res", "teacher"]}
    for dx in dxs:
        teacher_fn = lambda XU, d=dx: accel_asym(XU, NOMINAL, float(d))
        sc = scale_for(float(dx), dev)
        stB = train_resid("B", res_iters, dev, sc, float(dx), seed=0)
        stC = train_resid("C", res_iters, dev, sc, float(dx), seed=0)
        models = {"nominal": nominal_accel, "free-res": stB, "constr-res": stC, "teacher": teacher_fn}
        for name, fn in models.items():
            pol, _ = train_policy(fn, pol_iters, dev, seed=0)
            own, _ = evaluate(pol, fn, dev)
            dep, _ = evaluate(pol, teacher_fn, dev)        # 部署到真实 teacher(δx)
            rows[name]["own"].append(own); rows[name]["deploy"].append(dep)
        print(f"  δx={dx:+.2f}  deploy@teacher  " +
              " ".join(f"{k}={rows[k]['deploy'][-1]:.4f}" for k in rows), flush=True)
    return rows


def _plot(deltas, gain, cos, dxs, cl):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    flip = HALF
    # (a) 差动控制增益 g_diff vs δx —— 符号翻转机理
    gmap = [("teacher", C_TEA, "teacher (true)"), ("A", C_NOM, "A nominal"),
            ("B", C_B, "B free accel-res"), ("C", C_C, "C constr contact-res")]
    for k, c, lab in gmap:
        lw = 2.6 if k == "teacher" else 1.8
        ax[0].plot(deltas, gain[k], color=c, lw=lw, marker="o", ms=3, label=lab)
    for xl in (-flip, flip):
        ax[0].axvline(xl, color="k", lw=1, ls="--", alpha=.6)
    ax[0].axhline(0, color="r", lw=.9, ls=":")
    ax[0].text(flip, 0, " arm flips\n |δx|=half_len", fontsize=7, va="bottom")
    ax[0].set_xlabel("scissor offset δx [m]  (fd=[−δx,+δx])")
    ax[0].set_ylabel(r"diff. control gain $\partial\alpha/\partial e_{\rm diff}$")
    ax[0].set_title("(a) teacher's control sign flips at δx=0.25;\nnominal & C cannot (welded arm)")
    ax[0].legend(fontsize=7)
    # (b) 姿态梯度余弦保真 vs δx
    for k, c, lab in [("A", C_NOM, "A nominal"), ("B", C_B, "B free accel-res"),
                      ("C", C_C, "C constr contact-res")]:
        ax[1].plot(deltas, cos[k]["att"], color=c, lw=2, marker="s", ms=3, label=lab)
    for xl in (-flip, flip):
        ax[1].axvline(xl, color="k", lw=1, ls="--", alpha=.6)
    ax[1].set_xlabel("scissor offset δx [m]"); ax[1].set_ylabel("attitude ∂α/∂ext cosine to teacher")
    ax[1].set_title("(b) attitude-gradient fidelity\n(B≈A: free residual never helps it)")
    ax[1].legend(fontsize=7)
    # (c) 闭环部署 loss vs δx —— 谁先崩
    cmap = {"nominal": C_NOM, "free-res": C_B, "constr-res": C_C, "teacher": C_TEA}
    for name, c in cmap.items():
        ax[2].plot(dxs, cl[name]["deploy"], color=c, lw=2, marker="o", ms=4, label=f"π_{name}")
    ax[2].axvline(flip, color="k", lw=1, ls="--", alpha=.6)
    ax[2].set_yscale("log"); ax[2].set_xlabel("scissor offset δx [m]")
    ax[2].set_ylabel("balance loss deployed on teacher (log)")
    ax[2].set_title("(c) closed-loop collapse: who fails first?\n(past flip: C worst, free-res best non-oracle)")
    ax[2].legend(fontsize=7)
    fig.suptitle("R8  Extreme geometric mismatch (asymmetric scissor): moment-arm sign flip "
                 "bounds the value of physical structure", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R8_extreme_geometry.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 250 if args.quick else 800
    n_pts = 9 if args.quick else 19
    res_iters = 250 if args.quick else 800
    pol_iters = 60 if args.quick else 200
    dxs = [0.0, 0.30] if args.quick else [0.0, 0.15, 0.25, 0.35, 0.45]
    print(f"device={dev} half_len(arm-flip)={HALF} sweep_pts={n_pts} "
          f"res_iters={res_iters} pol_iters={pol_iters}", flush=True)

    print("== Part 1: gradient sweep vs δx (asymmetric scissor) ==", flush=True)
    deltas, gain, cos = part1_gradient_sweep(dev, n_iters, n_pts)
    print("== Part 2: closed-loop deployment vs δx ==", flush=True)
    cl = part2_closed_loop(dev, dxs, res_iters, pol_iters)

    res = {"half_len": HALF, "deltas": deltas, "diff_gain": gain, "grad_cos": cos,
           "closed_loop_dxs": dxs, "closed_loop": cl}
    json.dump(res, open(os.path.join(HERE, "results_phase3g.json"), "w"), indent=2, ensure_ascii=False)
    _plot(deltas, gain, cos, dxs, cl)
    print("[R8] figure saved.", flush=True)


if __name__ == "__main__":
    main()
