"""
结构性失配下的残差梯度保真 [R5]: 足端 x 偏置 δx —— 几何/力矩臂失配
==========================================================================
比缩放型(参数)失配更狠的检验: teacher 的脚实际落在 r_i+δx, student 以为在 r_i。
同样的法向力, 力矩臂 r_x 不同 -> 俯仰力矩 τ=r_x·F_n 的**方向效应**改变,
最易破坏**姿态梯度 ∂α/∂u、∂θ/∂u 的符号**(不只是大小错)。

teacher: srbm_accel(foot_dx=δx)   student: srbm_accel(foot_dx=0)  (同参数, 仅几何差)
对比三模型:
  A. nominal student
  B. student + 加速度残差            (直接补 α, 与几何解耦)
  C. student + R4 约束接触力残差     (补每足接触力, 但用 student 的**错误力臂** r_x)
核心猜想: 力律失配时 C 更优(R3); **几何失配时 C 把错误力臂"焊死"在结构里, B 可能反超**。

指标(尤其姿态): 前向RMS / 全梯度余弦 / 全符号错误率 / **∂α/∂ext 符号错误率** / rollout ∂θ_T/∂δe。
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import srbm_dynamics as srb
from srbm_residual import NOMINAL, sample_XU, Residual as AccelResidual
from srbm_residual_constrained import contact_accel_c, ContactResidual

C_NOM, C_B, C_C, C_TEA = "#8C8C8C", "#4C72B0", "#55A868", "#333333"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DELTA = 0.06   # 主实验: 足端前偏 6cm (基足偏移 ±0.25 的 ~24%)


def accel_struct(XU, p, foot_dx=None, model="smooth"):
    state = (XU[:, 0], XU[:, 1], XU[:, 2], XU[:, 3], XU[:, 4], XU[:, 5])
    ext = [XU[:, 6], XU[:, 7]]
    fd = None if foot_dx is None else [foot_dx, foot_dx]
    ax, az, al, _ = srb.srbm_accel(state, ext, model, p, [p.half_len, -p.half_len], foot_dx=fd)
    return torch.stack([ax, az, al], dim=-1)


def jac_of(fn, XU):
    XU = XU.clone().requires_grad_(True)
    A = fn(XU)
    Js = [torch.autograd.grad(A[:, j].sum(), XU, retain_graph=(j < 2))[0] for j in range(3)]
    return torch.stack(Js, dim=1).detach()


def cos_sign(J_s, J_t, rows=None, cols=None):
    """返回(余弦, 符号错误率)。rows/cols 限定子块(如姿态行 α=2, 控制列 ext=6,7)。"""
    if rows is None:
        s, t = J_s.reshape(J_s.shape[0], -1), J_t.reshape(J_t.shape[0], -1)
    else:
        s = J_s[:, rows][:, :, cols].reshape(J_s.shape[0], -1)
        t = J_t[:, rows][:, :, cols].reshape(J_t.shape[0], -1)
    cos = ((s * t).sum(-1) / (s.norm(dim=-1) * t.norm(dim=-1) + 1e-12)).mean()
    m = t.abs() > 0.05 * t.abs().mean()
    sgn = ((torch.sign(s) != torch.sign(t)) & m).sum().float() / (m.sum() + 1e-9)
    return float(cos), float(sgn)


def train(kind, n_iters, dev, scale, delta, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    teacher = lambda XU: accel_struct(XU, NOMINAL, foot_dx=delta)
    if kind == "B":
        r = AccelResidual().to(dev).to(torch.get_default_dtype())
        student = lambda XU: accel_struct(XU, NOMINAL, None) + r(XU)
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
    return r, student


def theta_grad(accel_fn, dev, n=400, dt=2e-3, de0=0.0, th0=0.12):
    """rollout: ∂θ_T/∂δe, δe=差动腿伸长(前+δe,后-δe) -> 俯仰平衡控制。"""
    de = torch.tensor(de0, device=dev, requires_grad=True)
    z = lambda v: torch.tensor(v, device=dev)
    px, pz, th, vx, vz, om = z(0.0), z(0.31), z(th0), z(0.0), z(0.0), z(0.0)
    base = 0.012
    for _ in range(n):
        ext0, ext1 = base + de, base - de
        XU = torch.stack([px, pz, th, vx, vz, om, ext0, ext1]).unsqueeze(0)
        A = accel_fn(XU)[0]
        vx = vx + dt * A[0]; vz = vz + dt * A[1]; om = om + dt * A[2]
        px = px + dt * vx; pz = pz + dt * vz; th = th + dt * om
    g, = torch.autograd.grad(th, de)
    return float(th.item()), float(g.item())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    n_iters = 250 if args.quick else 1500
    print(f"device={dev} n_iters={n_iters} DELTA={DELTA}")

    eval_XU = sample_XU(512, dev)
    teacher_fn = lambda XU: accel_struct(XU, NOMINAL, foot_dx=DELTA)
    nominal_fn = lambda XU: accel_struct(XU, NOMINAL, None)
    J_T = jac_of(teacher_fn, eval_XU)
    big = sample_XU(4096, dev)
    with torch.no_grad():
        scale = (teacher_fn(big) - nominal_fn(big)).std(dim=0).clamp_min(1e-6)

    rB, studentB = train("B", n_iters, dev, scale, DELTA, seed=0)
    rC, studentC = train("C", n_iters, dev, scale, DELTA, seed=0)
    models = {"A nominal": nominal_fn, "B accel-res": studentB, "C contact-res": studentC}

    res = {}
    for name, fn in models.items():
        J = jac_of(fn, eval_XU)
        cos_all, sgn_all = cos_sign(J, J_T)
        cos_att, sgn_att = cos_sign(J, J_T, rows=[2], cols=[6, 7])   # ∂α/∂ext
        with torch.no_grad():
            fwd = float(((fn(eval_XU) - teacher_fn(eval_XU)) ** 2).mean().sqrt())
        res[name] = dict(fwd_rms=fwd, grad_cos=cos_all, sign_err=sgn_all,
                         att_grad_cos=cos_att, att_sign_err=sgn_att)
        print(f"[{name:14s}] fwd={fwd:6.2f} cos={cos_all:.3f} signErr={sgn_all:.3f} "
              f"| ATT cos={cos_att:+.3f} signErr={sgn_att:.3f}")

    # rollout ∂θ_T/∂δe
    th_g = {}
    th_t, g_t = theta_grad(teacher_fn, dev)
    th_g["teacher"] = g_t
    for name, fn in models.items():
        _, g = theta_grad(fn, dev)
        th_g[name] = g
    print("[rollout ∂θ_T/∂δe] " + " ".join(f"{k}={v:+.3f}" for k, v in th_g.items()))
    res["rollout_dtheta_ddelta_e"] = th_g

    # δx 扫描: 姿态符号错误率 vs δx
    deltas = np.linspace(-0.08, 0.08, 9)
    sweep = {"A": [], "B": [], "C": []}
    for dx in deltas:
        Jt = jac_of(lambda XU: accel_struct(XU, NOMINAL, foot_dx=float(dx)), eval_XU)
        big2 = sample_XU(4096, dev)
        with torch.no_grad():
            sc = (accel_struct(big2, NOMINAL, float(dx)) - accel_struct(big2, NOMINAL, None)).std(0).clamp_min(1e-6)
        sweep["A"].append(cos_sign(jac_of(nominal_fn, eval_XU), Jt, [2], [6, 7])[0])  # 姿态余弦
        rBs, stB = train("B", n_iters // 2, dev, sc, float(dx), seed=1)
        rCs, stC = train("C", n_iters // 2, dev, sc, float(dx), seed=1)
        sweep["B"].append(cos_sign(jac_of(stB, eval_XU), Jt, [2], [6, 7])[0])
        sweep["C"].append(cos_sign(jac_of(stC, eval_XU), Jt, [2], [6, 7])[0])
    res["sweep_delta"] = deltas.tolist(); res["sweep_att_cos"] = sweep
    json.dump(res, open(os.path.join(HERE, "results_phase3d.json"), "w"), indent=2, ensure_ascii=False)

    # ---------- 图 R5 ----------
    names = list(models.keys()); cols = [C_NOM, C_B, C_C]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    # (a) 全梯度 vs 姿态梯度 余弦保真
    x = np.arange(len(names)); w = 0.35
    ax[0].bar(x - w / 2, [res[n]["grad_cos"] for n in names], w, color=cols, alpha=.55, label="full J")
    ax[0].bar(x + w / 2, [res[n]["att_grad_cos"] for n in names], w, color=cols, hatch="//",
              edgecolor="k", label="attitude ∂α/∂ext")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_ylim(0.85, 1.0); ax[0].set_ylabel("gradient cosine to true J"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) accel-res (B) corrupts attitude gradient\n(<nominal!); contact-res (C) preserves it")
    # (b) rollout ∂θ_T/∂δe
    keys = ["teacher", "A nominal", "B accel-res", "C contact-res"]
    ax[1].bar(range(4), [th_g[k] for k in keys], color=[C_TEA, C_NOM, C_B, C_C])
    ax[1].axhline(0, color="k", lw=.6); ax[1].axhline(th_g["teacher"], color=C_TEA, ls="--", lw=1)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["teacher\n(true)", "A", "B", "C"], fontsize=8)
    ax[1].set_ylabel(r"$\partial\theta_T/\partial\delta e$ (balance ctrl)")
    ax[1].set_title("(b) rollout attitude-control gradient\n(sign/scale vs true)")
    # (c) δx 扫描
    for k, c, lab in [("A", C_NOM, "A nominal"), ("B", C_B, "B accel-res"), ("C", C_C, "C contact-res")]:
        ax[2].plot(deltas, sweep[k], color=c, lw=2, marker="o", ms=3, label=lab)
    ax[2].axvline(0, color="k", lw=.6, ls=":")
    ax[2].set_xlabel("foot x-offset δx [m] (structural mismatch)")
    ax[2].set_ylabel("attitude ∂α/∂ext cosine")
    ax[2].set_title("(c) attitude-gradient fidelity vs mismatch size"); ax[2].legend(fontsize=8)
    fig.suptitle("R5  Structural (geometric) mismatch: foot-placement offset corrupts ATTITUDE "
                 "gradient direction", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R5_structural_mismatch.png"), bbox_inches="tight")
    plt.close(fig)
    print("[R5] figure saved.")


if __name__ == "__main__":
    main()
