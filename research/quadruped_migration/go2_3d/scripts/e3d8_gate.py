"""E3D-8 闸门：自适应步态保不保梯度保真？〔实验〕

固定步态（period/duty/PHASE_OFF 写死）是个**梯度选择**：接触调度对参数平滑可微。
"自适应步态"最朴素的实现把接触时机交给硬事件 → 把离散接触请回梯度图 → 复刻 2D-F4 病。
本闸门检验命题：把相位提升为连续状态、由策略调制频率 ω（phaseA），是否**保持固定步态的
梯度保真**；而硬接触切换（hard，事件式）是否如预测崩坏。判据**不需 oracle**——自适应
参数化引入的失真，用损失面的**有限差分真梯度 vs 解析 BPTT 梯度的一致性**直接测。

三臂（唯一变量=步态参数化，策略架构同一）：
  fixed   常 ω=1/period + 平滑接触（≡E3D-4a 基线，相位作状态以共享代码）
  phaseA  策略调 ω + 平滑接触（自适应；预测：保真≈fixed）
  hard    策略调 ω + 硬接触切换（负控；预测：梯度爆/保真崩，复刻 E3D-4a 分水岭）

判据：①全程 BPTT 梯度范数（fixed/phaseA 有界、hard 爆）；②FD-vs-解析梯度 cos×视野
（短视野 phaseA≈fixed≈1、hard 显著低）。两者都过 → 自适应在平滑接触侧不损梯度保真，
绿灯 Stage B（耦合 CPG 步态自涌现）。

stage: grad（闸门主判据）/ train（自适应是真的：训得动 + ω 随跟踪误差调制）。
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
from gait_3d import GaitConfig  # noqa: E402
from gait_adaptive import gait_dynamics_step, nominal_lx, trot_phase  # noqa: E402
from e3d4_gait_train import sample_init, loss_step  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
ARMS = ["fixed", "phaseA", "hard"]
ACOL = dict(fixed="tab:gray", phaseA="tab:green", hard="tab:red")
OMEGA_REL = 0.5     # 频率调制范围 ω=ω0·(1±0.5)


class AdaptivePolicy(nn.Module):
    """obs 11（与 e3d4 同：[vx_b−vx*,vy_b,vz_b,tilt(2),w(3),z−z_ref,sinφ,cosφ]）
    → R⁹=[每腿(ΔLx,Δext)×4, Δω_raw]。fixed 臂忽略 Δω（ω 恒定）。"""

    def __init__(self, obs=11, hid=32, act=9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act))

    def forward(self, o):
        return self.net(o)


def observe_ad(state, phi_g, g, z_ref):
    """相位作状态版 observe：phi_g (B,) 张量。"""
    R = quaternion_to_matrix(state.q)
    v_b = torch.einsum("bji,bj->bi", R, state.v)
    tilt = R[:, 2, :2]
    ph = torch.stack([torch.sin(2 * np.pi * phi_g), torch.cos(2 * np.pi * phi_g)], dim=-1)
    return torch.cat([v_b[:, 0:1] - g.vx_cmd, v_b[:, 1:2], v_b[:, 2:3], tilt,
                      state.w, state.p[:, 2:3] - z_ref, ph], dim=-1)


def step_arm(pol, state, phi_g, cfg, g, z_ref, arm):
    """一步自适应步态：策略→(ω, Lx, dext)→相位足端动力学。返回 nxt, phi_g', a, omega。"""
    omega0 = 1.0 / g.period
    B = state.p.shape[0]
    a = pol(observe_ad(state, phi_g, g, z_ref))
    if arm == "fixed":
        omega = state.p.new_full((B,), omega0)
    else:
        omega = omega0 * (1.0 + OMEGA_REL * torch.tanh(a[:, 8]))
    dLx = g.dLx_max * torch.tanh(a[:, 0:8:2])
    dext = g.dext_max * torch.tanh(a[:, 1:8:2])
    Lx = nominal_lx(omega, cfg, g)[:, None] + dLx
    phi = trot_phase(phi_g)
    phidot = omega[:, None].expand(B, 4)
    mode = "hard" if arm == "hard" else "smooth"
    nxt, info = gait_dynamics_step(state, phi, phidot, Lx, dext, cfg, g, mode=mode)
    phi_g = (phi_g + omega * cfg.dt) % 1.0
    info["omega"] = omega
    return nxt, phi_g, a, info


def rollout(pol, cfg, g, z_ref, state, phi_g, horizon, arm, tbptt=None):
    loss = state.p.new_zeros(())
    for t in range(horizon):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach(); phi_g = phi_g.detach()
        state, phi_g, a, _ = step_arm(pol, state, phi_g, cfg, g, z_ref, arm)
        loss = loss + loss_step(state, a, g, z_ref)
    return loss / horizon


def sample_sp(cfg, g, z_ref, B, gen):
    s = sample_init(cfg, g, z_ref, B, gen)
    phi_g = torch.rand(B, generator=gen, device=cfg.device, dtype=cfg.dtype)
    return s, phi_g


# --------------------------------------------------------------------------- #
def _flat(pol):
    return torch.cat([p.detach().reshape(-1) for p in pol.parameters()])


def _set_flat(pol, flat):
    with torch.no_grad():
        i = 0
        for p in pol.parameters():
            n = p.numel(); p.copy_(flat[i:i + n].view_as(p)); i += n


def fd_fidelity(pol, cfg, g, z_ref, state, phi_g, arm, H, n_dirs=8, eps=1e-3, seed=0):
    """FD-vs-解析梯度保真：解析方向导 g·u vs 中心差分 (L(θ+εu)−L(θ−εu))/2ε，
    n_dirs 个随机单位方向上比 cos 与相对误差。损失面光滑→cos→1；硬切换→解析失真→cos↓。"""
    for p in pol.parameters():
        p.grad = None
    loss = rollout(pol, cfg, g, z_ref, state, phi_g, H, arm)
    loss.backward()
    gflat = torch.cat([p.grad.reshape(-1) for p in pol.parameters()]).detach()
    if not torch.isfinite(gflat).all():
        return float("nan"), float("inf"), float("inf")
    flat0 = _flat(pol)
    n = flat0.numel()
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 7)
    an, fd = [], []
    for _ in range(n_dirs):
        u = torch.randn(n, generator=gen, device=cfg.device, dtype=cfg.dtype)
        u = u / u.norm()
        an.append((gflat * u).sum().item())
        with torch.no_grad():
            _set_flat(pol, flat0 + eps * u)
            lp = rollout(pol, cfg, g, z_ref, state, phi_g, H, arm).item()
            _set_flat(pol, flat0 - eps * u)
            lm = rollout(pol, cfg, g, z_ref, state, phi_g, H, arm).item()
        fd.append((lp - lm) / (2 * eps))
    _set_flat(pol, flat0)
    an, fd = np.array(an), np.array(fd)
    cos = float(an @ fd / (np.linalg.norm(an) * np.linalg.norm(fd) + 1e-12))
    rel = float(np.mean(np.abs(an - fd) / (np.abs(fd) + 1e-9)))
    gn = float(np.linalg.norm(gflat.cpu().numpy()))
    return cos, rel, gn


def stage_grad(cfg, g, z_ref, n_pts=4):
    t0 = time.time()
    pts = []
    for sd in range(n_pts):
        torch.manual_seed(sd)
        pts.append(AdaptivePolicy().to(cfg.device, cfg.dtype))
    gen = torch.Generator(device=cfg.device).manual_seed(123)
    s0, phi0 = sample_sp(cfg, g, z_ref, 16, gen)   # 固定批，三臂同初态

    # ---- 判据①：全程 BPTT 梯度范数分水岭（H=800=2 周期，无 tbptt）----
    print("[判据①] 全程 BPTT(H=800) 梯度范数（fixed/phaseA 应有界、hard 应爆）:")
    gnorm800 = {}
    for arm in ARMS:
        gs = []
        for pol in pts:
            for p in pol.parameters():
                p.grad = None
            loss = rollout(pol, cfg, g, z_ref, s0, phi0, 800, arm)
            loss.backward()
            gf = torch.cat([p.grad.reshape(-1) for p in pol.parameters()])
            gs.append(gf.norm().item() if torch.isfinite(gf).all() else float("inf"))
        gnorm800[arm] = gs
        med = np.median([x for x in gs if np.isfinite(x)]) if any(np.isfinite(gs)) else float("inf")
        print(f"  {arm:7s}: |∇θL| 中位={med:.2e}  各点={[f'{x:.1e}' for x in gs]}")

    # ---- 判据②：FD-vs-解析梯度保真 × 视野 ----
    Hs = [50, 150, 300]
    print(f"\n[判据②] FD-vs-解析梯度 cos / 相对误差 × 视野 (n_pts={n_pts}, n_dirs=8): "
          f"[{time.time()-t0:.0f}s]")
    fid = {arm: {"H": Hs, "cos": [], "rel": [], "gn": []} for arm in ARMS}
    for arm in ARMS:
        for H in Hs:
            cs, rs, gns = [], [], []
            for k, pol in enumerate(pts):
                c, r, gn = fd_fidelity(pol, cfg, g, z_ref, s0, phi0, arm, H, seed=k)
                if np.isfinite(c):
                    cs.append(c); rs.append(r)
                gns.append(gn)
            cos = float(np.median(cs)) if cs else float("nan")
            rel = float(np.median(rs)) if rs else float("inf")
            fid[arm]["cos"].append(cos); fid[arm]["rel"].append(rel)
            fid[arm]["gn"].append(float(np.median([x for x in gns if np.isfinite(x)]))
                                  if any(np.isfinite(gns)) else float("inf"))
            print(f"  {arm:7s} H={H:3d}: cos={cos:+.3f}  rel={rel:.2f}  "
                  f"[{time.time()-t0:.0f}s]")

    # ---- 判定 ----
    def med_fin(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return np.median(xs) if xs else float("inf")
    gn_fixed = med_fin(gnorm800["fixed"]); gn_phaseA = med_fin(gnorm800["phaseA"])
    gn_hard = med_fin(gnorm800["hard"])
    # 短视野(H≤150) 保真：phaseA 接近 fixed，hard 明显更差
    cos_fix_s = np.nanmean(fid["fixed"]["cos"][:2])
    cos_pha_s = np.nanmean(fid["phaseA"]["cos"][:2])
    cos_hard_s = np.nanmean(fid["hard"]["cos"][:2])
    norm_ok = np.isfinite(gn_phaseA) and gn_phaseA < 5 * max(gn_fixed, 1e-12)
    fid_ok = cos_pha_s > 0.9 * cos_fix_s
    foil_ok = (not np.isfinite(gn_hard)) or gn_hard > 10 * gn_phaseA or cos_hard_s < cos_pha_s - 0.1
    passed = norm_ok and fid_ok and foil_ok
    verdict = ("✅ 闸门通过：自适应频率(phaseA)在平滑接触侧不损梯度保真"
               f"(短视野 cos {cos_pha_s:.2f}≈fixed {cos_fix_s:.2f}、梯度范数有界)，"
               f"硬切换(hard)如预测崩坏(范数{gn_hard:.1e}/cos {cos_hard_s:.2f})——"
               "绿灯 Stage B 耦合 CPG 步态涌现。"
               if passed else
               f"⚠ 需核对：norm_ok={norm_ok} fid_ok={fid_ok} foil_ok={foil_ok}")
    print(f"\n判定: {verdict}")

    out = dict(gnorm800={k: [None if not np.isfinite(x) else x for x in v]
                         for k, v in gnorm800.items()},
               fidelity=fid, verdict=verdict, passed=bool(passed))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3d8_gate.json").write_text(json.dumps(out, indent=2))

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    meds = [med_fin(gnorm800[arm]) for arm in ARMS]
    meds_plot = [m if np.isfinite(m) else max(med_fin(gnorm800["phaseA"]) * 1e3, 1e3)
                 for m in meds]
    a.bar(ARMS, meds_plot, color=[ACOL[x] for x in ARMS])
    for i, m in enumerate(meds):
        a.text(i, meds_plot[i], "∞(爆)" if not np.isfinite(m) else f"{m:.1e}",
               ha="center", va="bottom", fontsize=8)
    a.set_yscale("log"); a.set_ylabel("|∇θL| (中位)")
    a.set_title("① 全程 BPTT(H=800) 梯度范数\n固定/自适应有界 · 硬切换爆")
    a = ax[1]
    for arm in ARMS:
        a.plot(Hs, fid[arm]["cos"], "o-", color=ACOL[arm], label=arm)
    a.axhline(1.0, color="k", ls=":", lw=1, label="完美(=1)")
    a.set_xscale("log"); a.set_xticks(Hs); a.set_xticklabels(Hs)
    a.set_xlabel("BPTT 视野 H (步)"); a.set_ylabel("cos(解析∇, FD真∇)")
    a.set_title("② 梯度保真 × 视野\n自适应≈固定 · 硬切换塌"); a.legend(fontsize=8)
    a = ax[2]
    for arm in ARMS:
        rel = [min(r, 1e3) for r in fid[arm]["rel"]]
        a.plot(Hs, rel, "s-", color=ACOL[arm], label=arm)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(Hs); a.set_xticklabels(Hs)
    a.set_xlabel("BPTT 视野 H (步)"); a.set_ylabel("方向导相对误差 (↓)")
    a.set_title("② 同上(相对误差视角)"); a.legend(fontsize=8)
    fig.suptitle("E3D-8 闸门：连续相位自适应步态保持梯度保真，硬接触切换复刻 E3D-4a 分水岭",
                 fontsize=12)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    outp = FIG / "e3d8_gate.png"
    fig.savefig(outp, dpi=110, bbox_inches="tight")
    print(f"saved {outp}\nsaved {RESULTS / 'e3d8_gate.json'}")


# --------------------------------------------------------------------------- #
def train_arm(cfg, g, z_ref, arm, iters, B=32, H=600, tbptt=150, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = AdaptivePolicy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = []
    for _ in range(iters):
        s, phi = sample_sp(cfg, g, z_ref, B, gen)
        loss = rollout(pol, cfg, g, z_ref, s, phi, H, arm, tbptt=tbptt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), clip)
        opt.step(); hist.append(loss.item())
    return pol, hist


@torch.no_grad()
def eval_arm(pol, cfg, g, z_ref, arm, horizon=2000, seed=7, B=32):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s, phi = sample_sp(cfg, g, z_ref, B, gen)
    vxs, omes, verr = [], [], []
    for t in range(horizon):
        verr.append((s.v[:, 0] - g.vx_cmd).mean().item())
        s, phi, a, info = step_arm(pol, s, phi, cfg, g, z_ref, arm)
        vxs.append(s.v[:, 0].mean().item())
        omes.append(info["omega"].mean().item())
    vxs, omes = np.array(vxs), np.array(omes)
    half = horizon // 2
    omega0 = 1.0 / g.period
    return dict(vx_rmse=float(np.sqrt(((vxs - g.vx_cmd)[half:] ** 2).mean())),
                vx_mean=float(vxs[half:].mean()),
                omega_mean=float(omes[half:].mean()), omega0=omega0,
                omega_std=float(omes[half:].std()), vxs=vxs, omes=omes)


def stage_train(cfg, g, z_ref, iters):
    t0 = time.time()
    res = {}
    for arm in ["fixed", "phaseA"]:
        pol, hist = train_arm(cfg, g, z_ref, arm, iters)
        torch.save(pol.state_dict(), MODELS / f"e3d8_{arm}.pt")
        ev = eval_arm(pol, cfg, g, z_ref, arm)
        res[arm] = dict(loss_final=hist[-1], **{k: v for k, v in ev.items()
                                                if not isinstance(v, np.ndarray)})
        res[arm + "_traj"] = dict(vxs=ev["vxs"].tolist(), omes=ev["omes"].tolist())
        print(f"  {arm:7s} loss→{hist[-1]:.4f}  vx {ev['vx_mean']:.3f}/{g.vx_cmd} "
              f"(RMSE {ev['vx_rmse']*1e3:.0f}mm/s)  ω {ev['omega_mean']:.2f}/{ev['omega0']:.2f} "
              f"(±{ev['omega_std']:.2f}) [{time.time()-t0:.0f}s]")
    MODELS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e3d8_train.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    t = np.arange(len(res["phaseA_traj"]["vxs"])) * cfg.dt
    a = ax[0]
    a.plot(t, res["fixed_traj"]["vxs"], color=ACOL["fixed"], label="fixed")
    a.plot(t, res["phaseA_traj"]["vxs"], color=ACOL["phaseA"], label="phaseA")
    a.axhline(g.vx_cmd, color="k", ls=":", label="vx*")
    a.set_xlabel("t (s)"); a.set_ylabel("vx (m/s)"); a.set_title("速度跟踪"); a.legend(fontsize=8)
    a = ax[1]
    a.plot(t, res["phaseA_traj"]["omes"], color=ACOL["phaseA"], label="phaseA ω(t)")
    a.axhline(1.0 / g.period, color="tab:gray", ls="--", label="ω0=1/T")
    a.set_xlabel("t (s)"); a.set_ylabel("步态频率 ω (cycle/s)")
    a.set_title("自适应频率：ω 随跟踪状态调制"); a.legend(fontsize=8)
    fig.suptitle("E3D-8 自适应是真的：phaseA 训得动且主动调制步态频率", fontsize=12)
    fig.tight_layout()
    outp = FIG / "e3d8_train.png"
    fig.savefig(outp, dpi=110, bbox_inches="tight")
    print(f"saved {outp}\nsaved {RESULTS / 'e3d8_train.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["grad", "train"])
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=120)
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-8 [{args.stage}] ({args.device})  vx*={g.vx_cmd} T={g.period}s duty={g.duty}")
    if args.stage == "grad":
        stage_grad(cfg, g, z_ref)
    else:
        stage_train(cfg, g, z_ref, args.iters)


if __name__ == "__main__":
    main()
