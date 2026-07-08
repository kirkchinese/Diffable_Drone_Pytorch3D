"""E3D-5a: SRBD 孪生的步态 → 全身 MuJoCo Go2（外部"真实系统"首联调 + 视频）。

管线：gait_3d.foot_plan（COM 系足端目标）→ +c_body 转 base 系 → 牛顿 IK（热启动）
→ MuJoCo 关节位置伺服。1s 站立 settle（含 0.3s 目标混合）+ 5s 开环 trot。

诚实定位：MuJoCo 全身模型对 SRBD 孪生是**真实未知失配**（腿质量/惯量、关节伺服动力学、
硬接触、伺服跟踪误差…），开环步态在这里的表现差距**就是 E3D-5b 要测量与修正的对象**
——走得不如孪生里好是预期结果，不是失败；摔倒才需要调底层（kp/速度档）。
产出：metrics JSON + 轨迹图 + **渲染视频**（非专家核验：机器狗走没走稳，眼睛直接判）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
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
sys.path.insert(0, str(HERE.parent / "models"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config  # noqa: E402
from gait_3d import GaitConfig, foot_plan  # noqa: E402
from go2_mjcf_full import build  # noqa: E402
from go2_leg_ik import LegKinematics  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
VIDEOS = HERE.parent / "videos"


def tilt_deg(quat_wxyz):
    w, x, y, z = quat_wxyz
    r22 = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(r22, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=60.0)
    ap.add_argument("--kv", type=float, default=2.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--hsw", type=float, default=0.04)
    ap.add_argument("--period", type=float, default=0.4)
    ap.add_argument("--T", type=float, default=6.0)
    ap.add_argument("--video", action="store_true", default=True)
    ap.add_argument("--tag", default="open_loop")
    args = ap.parse_args()

    cfgS = build_standing_config(device="cpu", dtype=torch.float32)
    g = GaitConfig(vx_cmd=args.vx, h_swing=args.hsw, period=args.period)
    c_body = cfgS.c_body.numpy()
    kin = LegKinematics()
    mj = build(kp=args.kp, kv=args.kv)
    d = mujoco.MjData(mj)

    # 站姿初始化：stance 中心目标 → IK → 设初始 qpos，base 高度放到运动学一致处
    stance_b = cfgS.foot_rel_com.numpy() + c_body
    stance_b[:, 0] = cfgS.foot_rel_com.numpy()[:, 0] + c_body[0]
    stance_b[:, 2] = cfgS.foot_rel_com.numpy()[:, 2] - g.ext0 + c_body[2]
    q_stand = kin.ik(stance_b)
    d.qpos[0:3] = [0, 0, -stance_b[:, 2].mean() + 0.022]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = q_stand.ravel()
    d.ctrl[:] = q_stand.ravel()
    mujoco.mj_forward(mj, d)

    renderer = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(mj, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.6, -15, 135

    n_settle = 1000
    n_total = int(args.T * 1000)
    qprev = q_stand
    log = dict(t=[], vx=[], z=[], tilt=[], ncon=[])
    for t in range(n_total):
        if t < n_settle:                                   # settle: 末 300ms 混合到步态相位0
            blend = max(0.0, (t - (n_settle - 300)) / 300.0)
            p_b, _, _, _ = foot_plan(0, torch.zeros(1, 8), cfgS, g)
            tgt = (1 - blend) * stance_b + blend * (p_b[0].numpy() + c_body)
        else:
            p_b, _, _, _ = foot_plan(t - n_settle, torch.zeros(1, 8), cfgS, g)
            tgt = p_b[0].numpy() + c_body
        qprev = kin.ik(tgt, q0=qprev, iters=4)
        d.ctrl[:] = qprev.ravel()
        mujoco.mj_step(mj, d)
        if t % 10 == 0:
            log["t"].append(t * 1e-3)
            log["vx"].append(float(d.qvel[0]))
            log["z"].append(float(d.qpos[2]))
            log["tilt"].append(tilt_deg(d.qpos[3:7]))
            log["ncon"].append(int(d.ncon))
        if renderer is not None and t % 40 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

    vx = np.array(log["vx"]); zz = np.array(log["z"]); tt = np.array(log["tilt"])
    i0 = (n_settle + 800) // 10                            # 步态稳态段（跳过起步）
    res = dict(kp=args.kp, kv=args.kv, vx_cmd=args.vx,
               vx_mean=float(vx[i0:].mean()), vx_std=float(vx[i0:].std()),
               z_mean=float(zz[i0:].mean()), tilt_mean=float(tt[i0:].mean()),
               tilt_max=float(tt[i0:].max()),
               fell=bool(tt[i0:].max() > 60 or zz[i0:].min() < 0.12),
               x_final=float(d.qpos[0]),
               ncon_mean=float(np.array(log["ncon"])[i0:].mean()))
    print(f"[e3d5a {args.tag}] vx={res['vx_mean']:.3f}±{res['vx_std']:.3f}/{args.vx} "
          f"z={res['z_mean']:.3f} tilt均/峰={res['tilt_mean']:.1f}/{res['tilt_max']:.1f}° "
          f"x末={res['x_final']:.2f}m {'❌摔倒' if res['fell'] else '✅走完'}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e3d5_mujoco_gait.json"
    all_res = json.loads(out.read_text()) if out.exists() else {}
    all_res[args.tag] = res
    out.write_text(json.dumps(all_res, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(log["t"], vx); ax[0].axhline(args.vx, color="k", ls=":")
    ax[0].axvline(1.0, color="gray", ls="--", lw=0.8)
    ax[0].set_title("MuJoCo 全身 Go2: vx(t)"); ax[0].set_xlabel("t (s)")
    ax[1].plot(log["t"], zz); ax[1].set_title("base z(t)"); ax[1].set_xlabel("t (s)")
    ax[2].plot(log["t"], tt); ax[2].set_title("tilt (deg)"); ax[2].set_xlabel("t (s)")
    fig.suptitle(f"E3D-5a 开环 trot → MuJoCo ({args.tag}): SRBD 步态在真实未知失配下的表现",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"e3d5_mujoco_gait_{args.tag}.png", dpi=110, bbox_inches="tight")

    if frames:
        VIDEOS.mkdir(exist_ok=True)
        vp = VIDEOS / f"e3d5_{args.tag}.mp4"
        imageio.mimsave(vp, frames, fps=25, codec="libx264", quality=7)
        print(f"video: {vp} ({len(frames)} 帧)")


if __name__ == "__main__":
    main()
