"""E3D-5: MuJoCo 全身 Go2 外部验证——真实未知失配下的收官管线（验证路径②）。

与 E3D-4b 的关键区别：失配不再是我们注入的已知量，而是 SRBD 孪生与全身 MuJoCo 的
**真实差距**（腿惯量~27%总质量、PD 跟踪滞后、真接触、dt 2ms…）。孪生与 MuJoCo 共享
同一控制器栈（gait_3d 足端规划→解析 IK→PD），残差解释的就是动力学本体之差。

阶段（--stage，独立落盘）：
  align    : 开环步态 + 标称孪生基线策略直接上 MuJoCo——走不走得动、差距多大；
             渲染视频（非专家核验的"眼见为实"件）。
  fit      : MuJoCo 闭环数据 →（孪生加速度 + 残差）拟合 MuJoCo 实测加速度
             （Δv/dt——这是对真实系统的测量，不是梯度路径，差分合法）；
             双头 MLP 与结构化头同拟，看真实失配长什么样（通道归因）。
  transfer : nominal 孪生 vs corrected 孪生训出的策略全部部署 MuJoCo，
             vx 跟踪 + 存活 + 视频（无 oracle 臂——真实系统不可微，这正是问题本身）。
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
import imageio
import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config, level_error  # noqa: E402
from floating_base_srbd import FloatingBaseState  # noqa: E402
from gait_3d import GaitConfig, foot_plan  # noqa: E402
from mj_go2 import MJGo2, MJGo2Full  # noqa: E402
from e3d4_gait_train import GaitPolicy, observe  # noqa: E402
from e3d4b_residual_gait import MODELS, load_nominal  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
VIDEOS = HERE.parent / "videos"
TWIN_DT = 1e-3
ORACLE = "menagerie"          # 由 main() 按 --oracle 设置；"menagerie"|"full"
SUF = ""                       # 输出文件后缀（full oracle 加 _full）


def _make_mj(cfg, kp, kd):
    if ORACLE == "full":
        return MJGo2Full(kp=100.0, kv=3.0, c_body=cfg.c_body.cpu().numpy())
    return MJGo2(kp=kp, kd=kd, c_body=cfg.c_body.cpu().numpy())


def rollout_mj(policy, cfg, g, z_ref, seconds=4.0, kp=300.0, kd=5.0,
               video_path=None, noise=0.0, seed=0, record_data=False, ground_mu=None,
               slope_deg=0.0):
    """策略（或 None=开环）在 MuJoCo 上滚 seconds 秒。
    返回 metrics 字典；record_data=True 时附 (states, t_steps, actions, next_v, next_w)。
    ground_mu 不为 None 时把所有 geom 的滑动摩擦设为该值（E3D-11 摩擦扫描）。
    slope_deg≠0 时倾斜重力（地板仍平=坡面对齐系，+x 为上坡）模拟斜坡（E3D-11b 摩擦×坡度）。"""
    mj = _make_mj(cfg, kp, kd)
    mj.reset()
    if ground_mu is not None:
        mj.m.geom_friction[:, 0] = ground_mu
    if slope_deg:
        th = np.radians(slope_deg)
        mj.m.opt.gravity[:] = [-9.81 * np.sin(th), 0.0, -9.81 * np.cos(th)]
    gen = torch.Generator().manual_seed(seed)
    # 初始化：基座放 home，关节按 t=0 足端目标 IK
    a0 = torch.zeros(1, 8)
    p_b0, _, _, _ = foot_plan(0, a0, cfg, g)
    mj.set_base([0, 0, float(z_ref) + 0.01], [1, 0, 0, 0])
    mj.set_joints_ik(p_b0[0].numpy() + mj.c_body)
    n_steps = int(seconds / mj.dt)
    frames = []
    renderer = cam = None
    if video_path:
        renderer = mujoco.Renderer(mj.m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance, cam.elevation, cam.azimuth = 1.6, -15, 120
    vx, zs, tilts, fell_at = [], [], [], None
    data = dict(S=[], T=[], A=[], V2=[], W2=[]) if record_data else None
    for k in range(n_steps):
        t_sec = k * mj.dt
        t_twin = int(round(t_sec / TWIN_DT))
        st = mj.state()
        phi = (t_sec / g.period) % 1.0
        if policy is None:
            a = torch.zeros(1, 8)
        else:
            with torch.no_grad():
                a = policy(observe(st, phi, g, z_ref))
        if noise:
            a = a + noise * torch.randn(a.shape, generator=gen)
        p_b, _, _, _ = foot_plan(t_twin, a, cfg, g)
        q_des = mj.foot_targets_to_qdes(p_b[0].numpy())
        mj.pd_step(q_des)
        st2 = mj.state()
        if record_data and k >= 50 and k % 3 == 0 and fell_at is None:
            data["S"].append(st); data["T"].append(t_twin); data["A"].append(a)
            data["V2"].append(st2.v); data["W2"].append(st2.w)
        vx.append(float(st2.v[0, 0])); zs.append(float(st2.p[0, 2]))
        tilt = float(level_error(st2.q)[0])
        tilts.append(np.degrees(np.arccos(np.clip(1 - tilt, -1, 1))))
        if fell_at is None and tilts[-1] > 45:
            fell_at = t_sec
        if renderer and k % 16 == 0:                       # 約 30 fps
            renderer.update_scene(mj.d, camera=cam)
            frames.append(renderer.render())
    if renderer:
        VIDEOS.mkdir(exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30, quality=7)
        print(f"  video → {video_path}")
    vx = np.array(vx); half = len(vx) // 2
    out = dict(vx_mean=float(vx[half:].mean()),
               vx_rmse=float(np.sqrt(((vx - g.vx_cmd)[half:] ** 2).mean())),
               z_mean=float(np.mean(zs[half:])), tilt_end=float(np.mean(tilts[-300:])),
               fell_at=fell_at, vxs=vx.tolist()[::8])
    if record_data:
        out["_data"] = data
    return out


def stage_align(cfg, g, z_ref):
    print("[align] 同一控制器栈直接上 MuJoCo（kp=300 kd=5——按开环跟踪RMSE选定的工作点）")
    res = {}
    r = rollout_mj(None, cfg, g, z_ref, video_path=VIDEOS / f"e3d5_openloop{SUF}.mp4")
    res["openloop"] = {k: v for k, v in r.items() if k != "vxs"}
    res["openloop_vxs"] = r["vxs"]
    print(f"  开环: vx {r['vx_mean']:.3f}/{g.vx_cmd} RMSE {r['vx_rmse']*1e3:.0f}mm/s "
          f"z {r['z_mean']:.3f} tilt {r['tilt_end']:.1f}° fell={r['fell_at']}")
    pol = load_nominal(cfg, 0)
    r = rollout_mj(pol, cfg, g, z_ref, video_path=VIDEOS / f"e3d5_nominal_policy{SUF}.mp4")
    res["nominal_policy"] = {k: v for k, v in r.items() if k != "vxs"}
    res["nominal_vxs"] = r["vxs"]
    print(f"  标称策略: vx {r['vx_mean']:.3f} RMSE {r['vx_rmse']*1e3:.0f}mm/s "
          f"z {r['z_mean']:.3f} tilt {r['tilt_end']:.1f}° fell={r['fell_at']}")
    (RESULTS / f"e3d5_align{SUF}.json").write_text(json.dumps(res, indent=2))
    walked = (res["nominal_policy"]["fell_at"] is None
              and res["nominal_policy"]["vx_mean"] > 0.1)
    print(f"  → {'✅ 能走，差距即失配，进入 fit' if walked else '⚠ 走不动/摔——先调 kp/kd/ext0 或降速'}")


def stage_fit(cfg, g, z_ref):
    """MuJoCo 闭环数据 → 残差拟合。目标 = 实测加速度（Δv/dt_mj 测量，非梯度路径）。"""
    from residual_gait import GaitDualHead, StructuredDual, gait_accel
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    pol = load_nominal(cfg, 0)
    S, T, A, V2, W2 = [], [], [], [], []
    for seed in range(4):
        r = rollout_mj(pol, cfg, g, z_ref, seconds=6.0, noise=0.05, seed=seed,
                       record_data=True)
        d = r["_data"]
        S += d["S"]; T += d["T"]; A += d["A"]; V2 += d["V2"]; W2 += d["W2"]
        print(f"  采集 seed{seed}: +{len(d['S'])} 样本 (vx {r['vx_mean']:.3f}, "
              f"fell={r['fell_at']})")
    st = FloatingBaseState(*[torch.cat([getattr(x, k) for x in S], 0).double()
                             for k in "pqvw"])
    tt = torch.tensor(T, dtype=torch.float64)
    aa = torch.cat(A, 0).double()
    dt_mj = _make_mj(cfg, 300, 5).dt                         # oracle 步长（full=1e-3, menagerie=2e-3）
    a_lin = (torch.cat(V2, 0).double() - st.v) / dt_mj
    a_ang = (torch.cat(W2, 0).double() - st.w) / dt_mj
    aT = torch.cat([a_lin, a_ang], dim=-1)                  # (N,6) 实测
    N = st.p.shape[0]
    ntr = int(N * 0.8)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    tr, ho = perm[:ntr], perm[ntr:]
    fsl = lambda idx: (FloatingBaseState(st.p[idx], st.q[idx], st.v[idx], st.w[idx]),
                       tt[idx], aa[idx], aT[idx])
    dtr, ttr, atr, yT = fsl(tr)
    dho, tho, aho, yH = fsl(ho)
    with torch.no_grad():
        base_tr = ((gait_accel(dtr, ttr, atr, cfg64, g) - yT) ** 2).mean().item()
        base_ho = ((gait_accel(dho, tho, aho, cfg64, g) - yH) ** 2).mean().item()
    out = {"n": N, "base_train": base_tr, "base_holdout": base_ho}
    print(f"  N={N}  孪生 vs MuJoCo 实测加速度 MSE: train {base_tr:.1f} / ho {base_ho:.1f}")
    for tag, mk_model, iters, lr in [("mlp", GaitDualHead, 3000, 3e-3),
                                     ("structured", StructuredDual, 800, 1e-2)]:
        torch.manual_seed(0)
        head = mk_model().double()
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(iters):
            fe, dx = head.extras(dtr, ttr, atr, cfg64, g)
            fit = ((gait_accel(dtr, ttr, atr, cfg64, g, fe, dx) - yT) ** 2).mean()
            reg = (fe ** 2).mean() + ((1e4 * dx) ** 2).mean()
            opt.zero_grad(); (fit + 1e-4 * reg).backward(); opt.step()
        with torch.no_grad():
            fe, dx = head.extras(dtr, ttr, atr, cfg64, g)
            aP = gait_accel(dtr, ttr, atr, cfg64, g, fe, dx)
            ftr = ((aP - yT) ** 2).mean().item()
            feh, dxh = head.extras(dho, tho, aho, cfg64, g)
            fho = ((gait_accel(dho, tho, aho, cfg64, g, feh, dxh) - yH) ** 2).mean().item()
            C_f = (aP - gait_accel(dtr, ttr, atr, cfg64, g, None, dx)).norm(dim=-1).mean().item()
            C_k = (aP - gait_accel(dtr, ttr, atr, cfg64, g, fe, None)).norm(dim=-1).mean().item()
        out[tag] = dict(fit_train=ftr, fit_holdout=fho, C_f=C_f, C_k=C_k,
                        drop_pct=100 * (1 - fho / base_ho))
        extra = ""
        if tag == "structured":
            out[tag]["kappa"] = float(head.kappa.item())
            out[tag]["delta"] = [float(x) for x in head.delta]
            extra = (f"  κ̂={head.kappa.item():.4f} "
                     f"δ̂={np.round([float(x) for x in head.delta], 4).tolist()}")
        torch.save(head.state_dict(), MODELS / f"mj_residual_{tag}{SUF}.pt")
        print(f"  [{tag:10s}] fit {ftr:.1f}/{fho:.1f} (降 {out[tag]['drop_pct']:.0f}%)  "
              f"通道归因 C_f={C_f:.1f} C_k={C_k:.1f}{extra}")
    (RESULTS / f"e3d5_fit{SUF}.json").write_text(json.dumps(out, indent=2))


def stage_transfer(cfg, g, z_ref, device="cuda:0"):
    """corrected 孪生（MLP/structured）训策略 → 全部部署 MuJoCo（无 oracle 臂）。"""
    from residual_gait import GaitDualHead, StructuredDual
    from e3d4b_residual_gait import train
    cfgT = build_standing_config(device=device, dtype=torch.float32)
    heads = {}
    for tag, mk_model in [("mlp", GaitDualHead), ("structured", StructuredDual)]:
        h = mk_model()
        h.load_state_dict(torch.load(MODELS / f"mj_residual_{tag}{SUF}.pt",
                                     map_location=device, weights_only=True))
        heads[tag] = h.to(cfgT.device, cfgT.dtype).eval()
        for p in heads[tag].parameters():
            p.requires_grad_(False)
    t0 = time.time()
    summary = {}
    for arm in ["nominal", "corrected_mlp", "corrected_struct"]:
        rows = []
        for seed in range(3):
            if arm == "nominal":
                pol = load_nominal(cfgT, seed)
            else:
                tag = arm.split("_")[1]
                tagf = {"mlp": "mlp", "struct": "structured"}[tag]
                f = MODELS / f"mj_pol_{tagf}_s{seed}{SUF}.pt"
                if f.exists():
                    pol = GaitPolicy().to(cfgT.device, cfgT.dtype)
                    pol.load_state_dict(torch.load(f, map_location=cfgT.device,
                                                   weights_only=True))
                else:
                    h = heads[tagf]
                    pol, hist = train(cfgT, g, z_ref,
                                      lambda s, t, a, hh=h: hh.extras(s, t, a, cfgT, g),
                                      seed=seed)
                    torch.save(pol.state_dict(), f)
                    print(f"  trained {arm} s{seed} loss→{hist[-1]:.4f} "
                          f"[{time.time()-t0:.0f}s]")
            polc = GaitPolicy()
            polc.load_state_dict({k: v.cpu() for k, v in pol.state_dict().items()})
            video = (VIDEOS / f"e3d5_{arm}{SUF}.mp4") if seed == 0 else None
            r = rollout_mj(polc, cfg, g, z_ref, seconds=4.0, video_path=video)
            rows.append(r)
            print(f"  [MJ eval] {arm:16s} s{seed}: vx {r['vx_mean']:.3f}/{g.vx_cmd} "
                  f"RMSE {r['vx_rmse']*1e3:.0f}mm/s tilt {r['tilt_end']:.1f}° "
                  f"fell={r['fell_at']}")
        summary[arm] = dict(
            vx_mean=float(np.mean([r["vx_mean"] for r in rows])),
            vx_rmse_mm=float(np.mean([r["vx_rmse"] * 1e3 for r in rows])),
            vx_rmse_std=float(np.std([r["vx_rmse"] * 1e3 for r in rows])),
            fell=sum(1 for r in rows if r["fell_at"] is not None))
        m = summary[arm]
        print(f"  == {arm:16s}: vx {m['vx_mean']:.3f}  RMSE {m['vx_rmse_mm']:.0f}"
              f"±{m['vx_rmse_std']:.0f}mm/s  摔倒 {m['fell']}/3")
    (RESULTS / f"e3d5_transfer{SUF}.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["align", "fit", "transfer"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--oracle", default="menagerie", choices=["menagerie", "full"])
    args = ap.parse_args()
    global ORACLE, SUF
    ORACLE = args.oracle
    SUF = "" if args.oracle == "menagerie" else "_full"
    cfg = build_standing_config(device="cpu", dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-5 [{args.stage}]")
    if args.stage == "transfer":
        stage_transfer(cfg, g, z_ref, device=args.device)
    else:
        dict(align=stage_align, fit=stage_fit)[args.stage](cfg, g, z_ref)


if __name__ == "__main__":
    main()
