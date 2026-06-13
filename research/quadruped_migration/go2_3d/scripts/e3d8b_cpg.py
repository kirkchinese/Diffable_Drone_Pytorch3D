"""E3D-8b: 步态自涌现（耦合 CPG）+ 感觉反馈（Tegotae）〔实验〕

闸门(E3D-8)证明：连续相位 + 平滑接触 → 梯度保真，绿灯。这里把"步态本身"交出去：
不写死 PHASE_OFF，给每腿一个相位振荡器，经**可学习耦合 K** 互相牵引，锁相后的相位模式
即步态——由 K 的结构**对称破缺涌现**。因为耦合平滑可微（闸门已证），可用 BPTT 直接训 K。

三个论点：
  emerge   纯振荡器：手设 trot-K 锁出对角步态、随机 K 锁出别的/不锁——机制演示（无身体）。
  train    K 作可学习参数，端到端训速度任务：步态模式从随机耦合**涌现**，读出锁相相位偏移、
           与 trot 比对、对照固定 trot 基线（涌现的能不能跟写死的打平）。
  tegotae  相位推进被足端载荷调制（载重→慢推→久留支撑，感觉反馈 CPG）：扰动下比开环
           CPG 更稳/滑移更小？兑现则反馈有值，否则诚实记空。

stage: emerge / train / tegotae。判据沿用研究线：梯度保真已由闸门保证，这里看涌现质量
与闭环表现，绝不只看"训得下去"。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
import _plotstyle  # noqa: E402
_plotstyle.use_cjk()
from srbd_standing import build_standing_config  # noqa: E402
from gait_3d import GaitConfig, PHASE_OFF  # noqa: E402
from gait_adaptive import (gait_dynamics_step, nominal_lx, cpg_phase_rate,  # noqa: E402
                           trot_coupling)
from e3d4_gait_train import sample_init, loss_step  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
LEG_NAMES = ["FL", "FR", "RL", "RR"]


# ======================================================================= #
# Stage emerge：纯振荡器锁相（机制演示，无身体）
# ======================================================================= #
def integrate_cpg(K, phi0, omega0, dt=1e-3, steps=1500):
    """积分耦合 CPG（无身体），返回相位轨迹 (steps,4)（单样本）。"""
    phi = phi0.clone()[None]                       # (1,4)
    om = torch.tensor([omega0])
    traj = [phi[0].clone()]
    for _ in range(steps):
        phi = (phi + cpg_phase_rate(phi, om, K) * dt) % 1.0
        traj.append(phi[0].clone())
    return torch.stack(traj).numpy()


def rel_offsets(traj, tail=300):
    """末段相对 leg0 的锁相偏移（cycles, mod 1）+ 是否锁相（偏移方差小）。"""
    last = traj[-tail:]
    d = (last - last[:, 0:1]) % 1.0
    off = np.angle(np.exp(1j * 2 * np.pi * d).mean(0)) / (2 * np.pi) % 1.0   # 圆均值
    # 锁相度：相邻偏移变化的标准差（小=锁住）
    lock = float(np.std(np.diff((traj[-tail:] - traj[-tail:, 0:1]) % 1.0, axis=0)))
    return off, lock


def stage_emerge():
    omega0 = 1.0 / GaitConfig().period
    torch.manual_seed(0)
    phi0 = torch.rand(4)
    cases = {"trot-K (手设)": trot_coupling(4.0),
             "random-K #1": (torch.rand(4, 4) - 0.5) * 6,
             "zero-K (无耦合)": torch.zeros(4, 4)}
    trot_ref = np.array(PHASE_OFF)
    fig, ax = plt.subplots(1, len(cases), figsize=(15, 4.4), sharey=True)
    out = {}
    for i, (name, K) in enumerate(cases.items()):
        traj = integrate_cpg(K, phi0, omega0)
        off, lock = rel_offsets(traj)
        err = float(np.min([np.abs(((off - np.roll(trot_ref, s) + 0.5) % 1.0) - 0.5).mean()
                            for s in range(4)]))  # 对腿序循环不变的 trot 距离
        out[name] = dict(offsets=off.tolist(), lock=lock, trot_dist=err)
        t = np.arange(traj.shape[0]) * 1e-3
        d = (traj - traj[:, 0:1]) % 1.0
        for leg in range(4):
            ax[i].plot(t, d[:, leg], label=f"{LEG_NAMES[leg]}", lw=1.3)
        ax[i].set_title(f"{name}\n锁相偏移 {np.round(off,2)}\n离trot {err:.3f} lock {lock:.0e}")
        ax[i].set_xlabel("t (s)")
        if i == 0:
            ax[i].set_ylabel("相位差 φ_i−φ_0 (cycle)"); ax[i].legend(fontsize=8, ncol=2)
        for ph in trot_ref:
            ax[i].axhline(ph, color="k", ls=":", lw=0.6, alpha=0.5)
    fig.suptitle("E3D-8b emerge：耦合 → 锁相 → 步态。手设 trot-K 涌现对角步态(虚线=trot相位)",
                 fontsize=12)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    fig.savefig(FIG / "e3d8b_emerge.png", dpi=110, bbox_inches="tight")
    (RESULTS / "e3d8b_emerge.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        print(f"  {name:16s}: 偏移 {np.round(r['offsets'],3)}  离trot {r['trot_dist']:.3f}  "
              f"lock {r['lock']:.1e}")
    print(f"saved {FIG / 'e3d8b_emerge.png'}")


# ======================================================================= #
# Stage train / tegotae：可学习 K 端到端
# ======================================================================= #
class PolicyCPG(nn.Module):
    """obs 17 = [vx_b−vx*,vy_b,vz_b,tilt(2),w(3),z−z_ref, sin2πφ(4), cos2πφ(4)] → R⁸ 腿修正。"""

    def __init__(self, obs=17, hid=32, act=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act))

    def forward(self, o):
        return self.net(o)


def observe_cpg(state, phi, g, z_ref):
    R = quaternion_to_matrix(state.q)
    v_b = torch.einsum("bji,bj->bi", R, state.v)
    tilt = R[:, 2, :2]
    s = torch.sin(2 * np.pi * phi); c = torch.cos(2 * np.pi * phi)
    return torch.cat([v_b[:, 0:1] - g.vx_cmd, v_b[:, 1:2], v_b[:, 2:3], tilt,
                      state.w, state.p[:, 2:3] - z_ref, s, c], dim=-1)


def cpg_rollout_step(pol, K, state, phi, cfg, g, z_ref, omega0,
                     tegotae=False, fn_prev=None, sigma=0.6):
    B = state.p.shape[0]
    a = pol(observe_cpg(state, phi, g, z_ref))
    dLx = g.dLx_max * torch.tanh(a[:, 0:8:2])
    dext = g.dext_max * torch.tanh(a[:, 1:8:2])
    Lx = g.lx0 + dLx
    om = state.p.new_full((B,), omega0)
    phidot = cpg_phase_rate(phi, om, K)                  # (B,4)
    if tegotae and fn_prev is not None:
        W = cfg.mass.item() * 9.81
        N = (fn_prev / W).clamp(0.0, 1.5)                # 归一化载荷
        phidot = phidot * (1.0 - sigma * N)              # 载重→慢推→久留支撑
    nxt, info = gait_dynamics_step(state, phi, phidot, Lx, dext, cfg, g)
    phi = (phi + phidot * cfg.dt) % 1.0
    return nxt, phi, a, info


def cpg_rollout(pol, K, cfg, g, z_ref, state, phi, horizon, omega0,
                tbptt=None, tegotae=False, sigma=0.6):
    loss = state.p.new_zeros(())
    fn_prev = None
    for t in range(horizon):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach(); phi = phi.detach()
            if fn_prev is not None:
                fn_prev = fn_prev.detach()
        state, phi, a, info = cpg_rollout_step(pol, K, state, phi, cfg, g, z_ref,
                                               omega0, tegotae, fn_prev, sigma)
        fn_prev = info["f_n"]
        loss = loss + loss_step(state, a, g, z_ref)
    return loss / horizon


def train_cpg(cfg, g, z_ref, iters, omega0, k_init, tegotae=False, sigma=0.6,
              B=32, H=600, tbptt=150, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = PolicyCPG().to(cfg.device, cfg.dtype)
    K = nn.Parameter(k_init.to(cfg.device, cfg.dtype))
    opt = torch.optim.Adam(list(pol.parameters()) + [K], lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = []
    for _ in range(iters):
        s = sample_init(cfg, g, z_ref, B, gen)
        phi = torch.rand(B, 4, generator=gen, device=cfg.device, dtype=cfg.dtype)
        loss = cpg_rollout(pol, K, cfg, g, z_ref, s, phi, H, omega0,
                           tbptt=tbptt, tegotae=tegotae, sigma=sigma)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + [K], clip)
        opt.step(); hist.append(loss.item())
    return pol, K.detach(), hist


@torch.no_grad()
def eval_cpg(pol, K, cfg, g, z_ref, omega0, horizon=2000, seed=7, B=32,
             tegotae=False, sigma=0.6, push=None):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, g, z_ref, B, gen)
    phi = torch.rand(B, 4, generator=gen, device=cfg.device, dtype=cfg.dtype)
    fn_prev = None
    vxs, slips, rels, fell = [], [], [], 0
    for t in range(horizon):
        if push is not None and t == horizon // 3:        # 中途横/纵推扰动
            s.v[:, 0] = s.v[:, 0] + push
        s, phi, a, info = cpg_rollout_step(pol, K, s, phi, cfg, g, z_ref,
                                           omega0, tegotae, fn_prev, sigma)
        fn_prev = info["f_n"]
        vxs.append(s.v[:, 0].mean().item())
        st = info["stance"].float()
        if st.sum() > 0:
            slip = info["foot_v"][..., :2].norm(dim=-1)
            slips.append(((slip * st).sum() / st.sum()).item())
        # **逐样本**相对偏移（每样本绝对相位不同，绝不先跨批平均绝对相位）
        rels.append(((phi - phi[:, 0:1]) % 1.0).detach().cpu().numpy())   # (B,4)
    vxs = np.array(vxs)
    half = horizon // 2
    rel_tail = np.array(rels[-300:]).reshape(-1, 4)                 # (300·B,4)
    zc = np.exp(1j * 2 * np.pi * rel_tail).mean(0)                  # (4,) 圆均值
    off_mean = np.angle(zc) / (2 * np.pi) % 1.0
    lock = float(np.abs(zc).mean())                                 # 1=锁死, 0=散
    ref = np.array([0.0, 0.5, 0.5, 0.0])                            # trot 相对 FL 唯一表示
    trot_dist = float(np.abs(((off_mean - ref + 0.5) % 1.0) - 0.5).mean())
    return dict(vx_rmse=float(np.sqrt(((vxs - g.vx_cmd)[half:] ** 2).mean())),
                vx_mean=float(vxs[half:].mean()), slip_mean=float(np.mean(slips)),
                offsets=off_mean.tolist(), trot_dist=trot_dist, lock=lock, vxs=vxs)


def stage_train(cfg, g, z_ref, iters):
    omega0 = 1.0 / g.period
    t0 = time.time()
    torch.manual_seed(1)
    k_rand = (torch.rand(4, 4) - 0.5) * 2.0          # 随机小耦合起步（无 trot 先验）
    res = {}
    # 涌现臂：可学习 K 从随机起步
    pol, K, hist = train_cpg(cfg, g, z_ref, iters, omega0, k_rand)
    torch.save({"pol": pol.state_dict(), "K": K}, MODELS / "e3d8b_cpg.pt")
    ev = eval_cpg(pol, K, cfg, g, z_ref, omega0)
    res["cpg_emergent"] = dict(loss_final=hist[-1], hist=hist,
                               **{k: v for k, v in ev.items() if not isinstance(v, np.ndarray)})
    res["cpg_emergent"]["vxs"] = ev["vxs"].tolist()
    res["cpg_emergent"]["K"] = K.cpu().numpy().tolist()
    print(f"  cpg(涌现) loss→{hist[-1]:.4f}  vx {ev['vx_mean']:.3f}/{g.vx_cmd} "
          f"(RMSE {ev['vx_rmse']*1e3:.0f}mm/s)  涌现偏移 {np.round(ev['offsets'],2)} "
          f"离trot {ev['trot_dist']:.3f} 锁相度 {ev['lock']:.2f} [{time.time()-t0:.0f}s]")
    # 对照臂：固定 trot 耦合（不学 K，只学策略）——涌现 vs 写死
    pol2, K2, hist2 = train_cpg(cfg, g, z_ref, iters, omega0, trot_coupling(4.0))
    # 起点是 trot 且强耦合，作"已知好答案"参照。K 仍可学（看是否被任务推离 trot）。
    ev2 = eval_cpg(pol2, K2, cfg, g, z_ref, omega0)
    torch.save({"pol": pol2.state_dict(), "K": K2}, MODELS / "e3d8b_cpg_trotinit.pt")
    res["cpg_trotinit"] = dict(loss_final=hist2[-1],
                               **{k: v for k, v in ev2.items() if not isinstance(v, np.ndarray)})
    res["cpg_trotinit"]["K"] = K2.cpu().numpy().tolist()
    print(f"  cpg(trot起) loss→{hist2[-1]:.4f}  vx {ev2['vx_mean']:.3f} "
          f"(RMSE {ev2['vx_rmse']*1e3:.0f}mm/s)  偏移 {np.round(ev2['offsets'],2)} "
          f"离trot {ev2['trot_dist']:.3f} 锁相度 {ev2['lock']:.2f} [{time.time()-t0:.0f}s]")
    (RESULTS / "e3d8b_train.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    a.plot(hist, color="tab:green", label="cpg 涌现(随机K起)")
    a.plot(hist2, color="tab:olive", label="cpg trot起")
    a.set_yscale("log"); a.set_xlabel("iter"); a.set_ylabel("loss")
    a.set_title("可学习耦合端到端训练"); a.legend(fontsize=8)
    a = ax[1]
    off = np.array(res["cpg_emergent"]["offsets"]); ot = np.array(PHASE_OFF)
    x = np.arange(4); w = 0.35
    a.bar(x - w/2, off, w, color="tab:green", label="涌现锁相偏移")
    a.bar(x + w/2, ot, w, color="k", alpha=0.4, label="trot 参照")
    a.set_xticks(x); a.set_xticklabels(LEG_NAMES); a.set_ylabel("相位偏移 (cycle)")
    a.set_title(f"涌现步态 vs trot (离trot {res['cpg_emergent']['trot_dist']:.3f})")
    a.legend(fontsize=8)
    a = ax[2]
    t = np.arange(len(res["cpg_emergent"]["vxs"])) * cfg.dt
    a.plot(t, res["cpg_emergent"]["vxs"], color="tab:green", label="cpg 涌现")
    a.axhline(g.vx_cmd, color="k", ls=":", label="vx*")
    a.set_xlabel("t (s)"); a.set_ylabel("vx (m/s)")
    a.set_title(f"涌现步态速度跟踪 RMSE {res['cpg_emergent']['vx_rmse']*1e3:.0f}mm/s")
    a.legend(fontsize=8)
    fig.suptitle("E3D-8b train：步态模式从可学习耦合涌现，BPTT 梯度保真（闸门已证）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "e3d8b_train.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d8b_train.png'}")


def stage_tegotae(cfg, g, z_ref, iters):
    omega0 = 1.0 / g.period
    t0 = time.time()
    # 公平起点=稳定 trot 耦合（非退化涌现 K）；温和 σ（重载脚相位率 ×(1−0.3·1.5)=0.55，
    # 非 σ=0.6 的近停滞 0.1）——给载荷反馈一个公平的机会再下结论。
    K0 = trot_coupling(4.0).to(cfg.device, cfg.dtype)
    SIGMA = 0.3
    res = {}
    for name, teg in [("open", False), ("tegotae", True)]:
        pol, K, hist = train_cpg(cfg, g, z_ref, iters, omega0, K0.clone(),
                                 tegotae=teg, sigma=SIGMA, seed=2)
        evn = eval_cpg(pol, K, cfg, g, z_ref, omega0, tegotae=teg, sigma=SIGMA)
        evp = eval_cpg(pol, K, cfg, g, z_ref, omega0, tegotae=teg, sigma=SIGMA, push=0.25)
        res[name] = dict(loss_final=hist[-1],
                         rmse=evn["vx_rmse"] * 1e3, slip=evn["slip_mean"] * 1e3,
                         rmse_push=evp["vx_rmse"] * 1e3, slip_push=evp["slip_mean"] * 1e3)
        print(f"  {name:8s} loss→{hist[-1]:.4f}  RMSE {res[name]['rmse']:.0f}mm/s "
              f"滑移 {res[name]['slip']:.0f} | 扰动后 RMSE {res[name]['rmse_push']:.0f} "
              f"滑移 {res[name]['slip_push']:.0f} [{time.time()-t0:.0f}s]")
    better = (res["tegotae"]["rmse_push"] < res["open"]["rmse_push"] and
              res["tegotae"]["slip_push"] <= res["open"]["slip_push"] * 1.02)
    res["verdict"] = ("感觉反馈有值：扰动下 tegotae 跟踪/滑移更好" if better else
                      "诚实记空：本设定下载荷反馈未显著优于开环 CPG")
    print(f"  → {res['verdict']}")
    (RESULTS / "e3d8b_tegotae.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    arms = ["open", "tegotae"]; cols = ["tab:gray", "tab:purple"]
    a = ax[0]; x = np.arange(2); w = 0.35
    a.bar(x - w/2, [res[k]["rmse"] for k in arms], w, color=cols, label="平稳")
    a.bar(x + w/2, [res[k]["rmse_push"] for k in arms], w, color=cols, alpha=0.5,
          label="扰动后")
    a.set_xticks(x); a.set_xticklabels(["开环CPG", "Tegotae"]); a.set_ylabel("vx RMSE (mm/s)")
    a.set_title("载荷反馈相位调制：跟踪鲁棒性"); a.legend(fontsize=8)
    a = ax[1]
    a.bar(x - w/2, [res[k]["slip"] for k in arms], w, color=cols, label="平稳")
    a.bar(x + w/2, [res[k]["slip_push"] for k in arms], w, color=cols, alpha=0.5,
          label="扰动后")
    a.set_xticks(x); a.set_xticklabels(["开环CPG", "Tegotae"]); a.set_ylabel("支撑足滑移 (mm/s)")
    a.set_title("锚定质量（支撑足世界滑移）"); a.legend(fontsize=8)
    fig.suptitle("E3D-8b tegotae：相位被足端载荷调制（载重久留支撑）" + "\n" + res["verdict"],
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e3d8b_tegotae.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d8b_tegotae.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["emerge", "train", "tegotae"])
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=140)
    args = ap.parse_args()
    g = GaitConfig()
    if args.stage == "emerge":
        print("E3D-8b [emerge] 纯振荡器锁相（无身体）")
        stage_emerge()
        return
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-8b [{args.stage}] ({args.device}) vx*={g.vx_cmd} T={g.period}s")
    if args.stage == "train":
        stage_train(cfg, g, z_ref, args.iters)
    else:
        stage_tegotae(cfg, g, z_ref, args.iters)


if __name__ == "__main__":
    main()
