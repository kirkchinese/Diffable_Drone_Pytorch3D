"""E3D-4a: 预定 trot 步态 3D 前向速度跟踪——可微训练（2D-F4 的 3D 版）。

任务：平地 trot，跟踪 vx_cmd=0.3 m/s + 维持高度/姿态，从带扰动初态。
动作 a∈R⁸=每腿[ΔLx 落足/扫速修正, Δext 支撑深度修正]（gait_3d 内 tanh 限幅）。

对照（F4 复刻 + E3D-7 处方）：
  smooth+TBPTT150（主，3 seeds）/ smooth+全程BPTT / hard+TBPTT150（同配方同裁剪）。
验收：①smooth 训得动、hard 梯度爆/训不动（3D-F4 梯度分水岭）；②锥占用有界（推进=
摩擦速度伺服，占用∝速度误差）；③支撑足世界滑移小（锚定涌现的诊断）；④开环步态
（a=0）为参照，策略应在扰动下收紧跟踪。
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
import _plotstyle
_plotstyle.use_cjk()
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config, level_error  # noqa: E402
from floating_base_srbd import FloatingBaseState  # noqa: E402
from gait_3d import GaitConfig, gait_step  # noqa: E402
from e3d3_standing_train import level_to_angle  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix, euler_angles_to_matrix, \
    matrix_to_quaternion  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
H_TRAIN = 800       # 2 步态周期
H_EVAL = 3200       # 8 周期
SEEDS = [0, 1, 2]


class GaitPolicy(nn.Module):
    """obs 11 = [vx_b−vx*, vy_b, vz_b, tilt(2), w(3), z−z_ref, sinφ, cosφ] → a∈R⁸(未限幅)。"""

    def __init__(self, obs=11, hid=32, act=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act))

    def forward(self, o):
        return self.net(o)


def observe(state, phi, g, z_ref):
    R = quaternion_to_matrix(state.q)
    v_b = torch.einsum("bji,bj->bi", R, state.v)
    tilt = R[:, 2, :2]
    B = state.p.shape[0]
    ph = state.p.new_tensor([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)]).expand(B, 2)
    return torch.cat([v_b[:, 0:1] - g.vx_cmd, v_b[:, 1:2], v_b[:, 2:3], tilt,
                      state.w, state.p[:, 2:3] - z_ref, ph], dim=-1)


def loss_step(state, a, g, z_ref):
    vx = state.v[:, 0]
    return (8.0 * (vx - g.vx_cmd) ** 2
            + 6.0 * (state.p[:, 2] - z_ref) ** 2
            + 2.0 * level_error(state.q)
            + 0.3 * (state.v[:, 1] ** 2 + state.v[:, 2] ** 2)
            + 0.05 * (state.w ** 2).sum(-1)
            + 1e-3 * (a ** 2).sum(-1)).mean()


def sample_init(cfg, g, z_ref, B, gen):
    dev, dt = cfg.device, cfg.dtype
    ang = torch.zeros(B, 3, device=dev, dtype=dt)
    ang[:, 0] = (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.12
    ang[:, 1] = (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.12
    q = matrix_to_quaternion(euler_angles_to_matrix(ang, "XYZ"))
    p = torch.zeros(B, 3, device=dev, dtype=dt)
    p[:, 2] = z_ref + (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.04
    v = torch.zeros(B, 3, device=dev, dtype=dt)
    v[:, 0] = 0.5 * g.vx_cmd + (torch.rand(B, generator=gen, device=dev, dtype=dt) - 0.5) * 0.2
    w = (torch.rand(B, 3, generator=gen, device=dev, dtype=dt) - 0.5) * 0.4
    return FloatingBaseState(p, q, v, w)


def rollout_train(pol, cfg, g, z_ref, state, horizon, mode, tbptt):
    loss = state.p.new_zeros(())
    for t in range(horizon):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach()
        phi = (t * cfg.dt / g.period) % 1.0
        a = pol(observe(state, phi, g, z_ref)) if pol is not None \
            else state.p.new_zeros(state.p.shape[0], 8)
        state, _ = gait_step(state, t, a, cfg, g, mode=mode)
        loss = loss + loss_step(state, a, g, z_ref)
    return loss / horizon


def train(cfg, g, z_ref, mode, tbptt, iters, B=48, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = GaitPolicy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = {"loss": [], "gnorm": []}
    for _ in range(iters):
        s = sample_init(cfg, g, z_ref, B, gen)
        loss = rollout_train(pol, cfg, g, z_ref, s, H_TRAIN, mode, tbptt)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(pol.parameters(), clip).item()
        opt.step()
        hist["loss"].append(loss.item()); hist["gnorm"].append(gnorm)
    return pol, hist


@torch.no_grad()
def evaluate(pol, cfg, g, z_ref, label, horizon=H_EVAL, seed=7, B=32):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, g, z_ref, B, gen)
    loss = 0.0
    vxs, zs, atts, cones, slips = [], [], [], [], []
    for t in range(horizon):
        phi = (t * cfg.dt / g.period) % 1.0
        a = pol(observe(s, phi, g, z_ref)) if pol is not None \
            else s.p.new_zeros(B, 8)
        s2, info = gait_step(s, t, a, cfg, g)
        loss += loss_step(s2, a, g, z_ref).item()
        vxs.append(s2.v[:, 0].mean().item())
        zs.append(s2.p[:, 2].mean().item())
        atts.append(np.degrees(level_to_angle(level_error(s2.q).mean().item())))
        st = info["stance"].float()                    # (B,4) 支撑掩码
        if st.sum() > 0:
            cones.append(((info["cone"] * st).sum() / st.sum()).item())
            # 支撑足世界滑移（锚定涌现诊断）：真实足世界速度（含指令项 R·ṗ_b）
            slip = info["foot_v"][..., :2].norm(dim=-1)
            slips.append(((slip * st).sum() / st.sum()).item())
        s = s2
    vxs = np.array(vxs)
    half = horizon // 2
    res = dict(loss=loss / horizon,
               vx_rmse=float(np.sqrt(((vxs - g.vx_cmd)[half:] ** 2).mean())),
               vx_mean=float(vxs[half:].mean()),
               cone_mean=float(np.mean(cones)), cone_max=float(np.max(cones)),
               slip_mean=float(np.mean(slips)),
               tilt_end=float(np.mean(atts[-400:])),
               vxs=vxs, zs=np.array(zs), atts=np.array(atts))
    print(f"   [{label:18s}] loss={res['loss']:.4f}  vx {res['vx_mean']:.3f}/{g.vx_cmd} "
          f"(RMSE {res['vx_rmse']*1e3:.0f}mm/s)  cone均/峰={res['cone_mean']:.2f}/"
          f"{res['cone_max']:.2f}  支撑滑移={res['slip_mean']*1e3:.0f}mm/s  "
          f"tilt={res['tilt_end']:.2f}°")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=120)
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004     # 双足支撑穿透补偿后的步态参考高度
    print(f"E3D-4a trot 速度跟踪 ({args.device}): vx*={g.vx_cmd} T={g.period}s "
          f"duty={g.duty} Lx0={g.lx0*1e3:.0f}mm z_ref={z_ref:.3f}")

    # 坑#4: 训前梯度自检 smooth vs hard
    for mode in ("smooth", "hard"):
        torch.manual_seed(0)
        pol = GaitPolicy().to(cfg.device, cfg.dtype)
        gen = torch.Generator(device=cfg.device).manual_seed(1)
        s = sample_init(cfg, g, z_ref, 16, gen)
        loss = rollout_train(pol, cfg, g, z_ref, s, H_TRAIN, mode, None)
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in pol.parameters())).item()
        print(f"[grad sanity] {mode}: 全程BPTT(H={H_TRAIN}) |g|={gn:.2e}")

    runs = {}
    t0 = time.time()
    for seed in SEEDS:
        runs[("smooth+tbptt", seed)] = train(cfg, g, z_ref, "smooth", 150,
                                             args.iters, seed=seed)
        print(f"  smooth+tbptt seed{seed} loss→{runs[('smooth+tbptt', seed)][1]['loss'][-1]:.4f}"
              f" [{time.time()-t0:.0f}s]")
    for tag, mode, tb in [("smooth+full", "smooth", None), ("hard+tbptt", "hard", 150)]:
        runs[(tag, 0)] = train(cfg, g, z_ref, mode, tb, args.iters, seed=0)
        print(f"  {tag} seed0 loss→{runs[(tag, 0)][1]['loss'][-1]:.4f} [{time.time()-t0:.0f}s]")
    for key, (pol, _) in runs.items():
        torch.save(pol.state_dict(), MODELS / f"gait_{key[0].replace('+','_')}_s{key[1]}.pt")

    print("\n[eval] 长 rollout 8 周期 (开环步态 a=0 为参照):")
    evals = {"openloop": evaluate(None, cfg, g, z_ref, "openloop a=0")}
    for key, (pol, _) in runs.items():
        evals[f"{key[0]}|s{key[1]}"] = evaluate(pol, cfg, g, z_ref, f"{key[0]} s{key[1]}")

    # ---- 可视化 ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    a = ax[0, 0]
    for key in [("smooth+tbptt", 0), ("smooth+full", 0), ("hard+tbptt", 0)]:
        a.plot(runs[key][1]["loss"], label=key[0])
    a.set_yscale("log"); a.set_title("训练 loss"); a.set_xlabel("iter"); a.legend(fontsize=8)
    a = ax[0, 1]
    for key in [("smooth+tbptt", 0), ("smooth+full", 0), ("hard+tbptt", 0)]:
        a.plot(runs[key][1]["gnorm"], label=key[0], alpha=0.8)
    a.set_yscale("log"); a.set_title("梯度范数(裁剪前)"); a.set_xlabel("iter"); a.legend(fontsize=8)
    t = np.arange(H_EVAL) * cfg.dt
    a = ax[0, 2]
    a.plot(t, evals["openloop"]["vxs"], color="tab:gray", label="openloop")
    a.plot(t, evals["smooth+tbptt|s0"]["vxs"], color="tab:green", label="smooth+tbptt")
    a.plot(t, evals["hard+tbptt|s0"]["vxs"], color="tab:red", label="hard+tbptt", alpha=0.7)
    a.axhline(g.vx_cmd, color="k", ls=":", label="vx*")
    a.set_title("真实 vx(t)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)
    a = ax[1, 0]
    a.plot(t, evals["openloop"]["zs"], color="tab:gray", label="openloop")
    a.plot(t, evals["smooth+tbptt|s0"]["zs"], color="tab:green", label="smooth+tbptt")
    a.axhline(z_ref, color="k", ls=":", label="z_ref")
    a.set_title("COM z(t)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(t, evals["openloop"]["atts"], color="tab:gray", label="openloop")
    a.plot(t, evals["smooth+tbptt|s0"]["atts"], color="tab:green", label="smooth+tbptt")
    a.set_yscale("log"); a.set_title("tilt(t) (deg)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)
    a = ax[1, 2]
    names = ["openloop"] + [f"smooth+tbptt|s{s}" for s in SEEDS]
    a.bar(range(len(names)), [evals[n]["vx_rmse"] * 1e3 for n in names],
          color=["tab:gray"] + ["tab:green"] * 3)
    a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=20, fontsize=7)
    a.set_ylabel("vx RMSE (mm/s)"); a.set_title("速度跟踪 RMSE (后半程)")
    fig.suptitle("E3D-4a: trot velocity tracking through smooth contact (3D version of 2D-F4)",
                 fontsize=13)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    out = FIG / "e3d4_gait_train.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    summary = dict(
        gait=dict(vx_cmd=g.vx_cmd, period=g.period, duty=g.duty, lx0=g.lx0,
                  h_swing=g.h_swing, ext0=g.ext0), z_ref=z_ref,
        runs={f"{k[0]}|s{k[1]}": dict(loss_final=h["loss"][-1],
                                      gnorm_median=float(np.median(h["gnorm"])))
              for k, (_, h) in runs.items()},
        eval={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
              for k, v in evals.items()})
    (RESULTS / "e3d4_gait.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}\nsaved {RESULTS / 'e3d4_gait.json'}")


if __name__ == "__main__":
    main()
