"""E3D-5c: 广义力通道残差——攻 E3D-5 边界（表达类对了能不能修正真实失配）。

E3D-5 结论：足端力/足位残差对真实失配（腿惯量/多体耦合）**出表达类**，自由 MLP 拟合越好
迁移越差（168mm/s）。本实验开**广义力通道**（state→base 净 wrench 修正 Δf/Δτ），它是
base 6-DoF 加速度失配的最小 in-class 表示。同一份 MuJoCo 数据、同一训练/部署管线，唯一
变量是残差通道。预测的三种诚实结局：
  (a) gen-force ≥ nominal → 表达类对了就能修正，E3D-5 边界破；
  (b) gen-force ≈ nominal（但≪足力168）→ 出类有害、入类无害，收益仍被反馈掩盖(E3D-7一致)；
  (c) gen-force 仍 <nominal → 即便入类，一步拟合的梯度也不可信，更深的问题。
判据**只看 MuJoCo 迁移**（一步拟合必~99%，正是 E3D-5 教训：拟合分会骗人）。

stage：fit（拟广义力残差 + 报拟合%）/ transfer（gen-force 孪生训策略 → 部署 MuJoCo）。
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config  # noqa: E402
from floating_base_srbd import FloatingBaseState  # noqa: E402
from gait_3d import GaitConfig, gait_step  # noqa: E402
from residual_gait import GenForceHead, gait_accel  # noqa: E402
from e3d4_gait_train import GaitPolicy, observe, loss_step, sample_init  # noqa: E402
from e3d5_mujoco_check import rollout_mj, load_nominal  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
VIDEOS = HERE.parent / "videos"


def collect_mj_data(cfg, g, z_ref, pol, n_roll=4):
    S, T, A, V2, W2 = [], [], [], [], []
    for seed in range(n_roll):
        r = rollout_mj(pol, cfg, g, z_ref, seconds=6.0, noise=0.05, seed=seed,
                       record_data=True)
        d = r["_data"]
        S += d["S"]; T += d["T"]; A += d["A"]; V2 += d["V2"]; W2 += d["W2"]
        print(f"  采集 seed{seed}: +{len(d['S'])} (vx {r['vx_mean']:.3f} fell={r['fell_at']})")
    st = FloatingBaseState(*[torch.cat([getattr(x, k) for x in S], 0).double() for k in "pqvw"])
    tt = torch.tensor(T, dtype=torch.float64)
    aa = torch.cat(A, 0).double()
    a_lin = (torch.cat(V2, 0).double() - st.v) / 0.002
    a_ang = (torch.cat(W2, 0).double() - st.w) / 0.002
    return st, tt, aa, torch.cat([a_lin, a_ang], dim=-1)


def stage_fit(cfg, g, z_ref):
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    pol = load_nominal(cfg, 0)
    st, tt, aa, yT = collect_mj_data(cfg, g, z_ref, pol)
    N = st.p.shape[0]; ntr = int(N * 0.8)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    idx = lambda s: (FloatingBaseState(st.p[s], st.q[s], st.v[s], st.w[s]), tt[s], aa[s], yT[s])
    dtr, ttr, atr, yTr = idx(perm[:ntr])
    dho, tho, aho, yHo = idx(perm[ntr:])
    with torch.no_grad():
        base = ((gait_accel(dho, tho, aho, cfg64, g) - yHo) ** 2).mean().item()
    torch.manual_seed(0)
    head = GenForceHead().double()
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(3000):
        gf = head.gen(dtr, ttr, atr, cfg64, g)
        fit = ((gait_accel(dtr, ttr, atr, cfg64, g, gen_force=gf) - yTr) ** 2).mean()
        reg = (gf[0] ** 2).mean() + (gf[1] ** 2).mean()     # 输出正则：抑制 OOD 饱和
        loss = fit + 1e-3 * reg
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ftr = ((gait_accel(dtr, ttr, atr, cfg64, g, gen_force=head.gen(dtr, ttr, atr, cfg64, g)) - yTr) ** 2).mean().item()
        fho = ((gait_accel(dho, tho, aho, cfg64, g, gen_force=head.gen(dho, tho, aho, cfg64, g)) - yHo) ** 2).mean().item()
    torch.save(head.state_dict(), MODELS / "mj_genforce.pt")
    out = dict(n=N, base_holdout=base, fit_train=ftr, fit_holdout=fho,
               drop_pct=100 * (1 - fho / base))
    print(f"  N={N}  base(未修正) ho MSE={base:.0f}  广义力残差 ho={fho:.1f} (降{out['drop_pct']:.0f}%)")
    print("  → 对照 E3D-5: 足力MLP降97% / 结构降83%（但迁移 168/101mm/s）")
    (RESULTS / "e3d5c_fit.json").write_text(json.dumps(out, indent=2))


def train_gf(cfg, g, z_ref, head, seed, iters=120, B=48, H=800, tbptt=150, lr=3e-3, clip=1.0):
    torch.manual_seed(seed)
    pol = GaitPolicy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = []
    for _ in range(iters):
        s = sample_init(cfg, g, z_ref, B, gen)
        loss = s.p.new_zeros(())
        for t in range(H):
            if tbptt and t > 0 and t % tbptt == 0:
                s = s.detach()
            phi = (t * cfg.dt / g.period) % 1.0
            a = pol(observe(s, phi, g, z_ref))
            df, dtau = head.gen(s, t, a, cfg, g)
            s, _ = gait_step(s, t, a, cfg, g, gen_force=(df, dtau))
            loss = loss + loss_step(s, a, g, z_ref)
        loss = loss / H
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), clip)
        opt.step(); hist.append(loss.item())
    return pol, hist


def stage_transfer(cfg, g, z_ref, device="cuda:0"):
    cfgT = build_standing_config(device=device, dtype=torch.float32)
    head = GenForceHead()
    head.load_state_dict(torch.load(MODELS / "mj_genforce.pt", map_location=device,
                                    weights_only=True))
    head = head.to(cfgT.device, cfgT.dtype).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    t0, rows = time.time(), []
    for seed in range(3):
        f = MODELS / f"mj_pol_genforce_s{seed}.pt"
        if f.exists():
            pol = GaitPolicy().to(cfgT.device, cfgT.dtype)
            pol.load_state_dict(torch.load(f, map_location=cfgT.device, weights_only=True))
        else:
            pol, hist = train_gf(cfgT, g, z_ref, head, seed)
            torch.save(pol.state_dict(), f)
            print(f"  trained genforce s{seed} loss→{hist[-1]:.4f} [{time.time()-t0:.0f}s]")
        polc = GaitPolicy()
        polc.load_state_dict({k: v.cpu() for k, v in pol.state_dict().items()})
        video = (VIDEOS / "e3d5_corrected_genforce.mp4") if seed == 0 else None
        r = rollout_mj(polc, cfg, g, z_ref, seconds=4.0, video_path=video)
        rows.append(r)
        print(f"  [MJ eval] genforce s{seed}: vx {r['vx_mean']:.3f}/{g.vx_cmd} "
              f"RMSE {r['vx_rmse']*1e3:.0f}mm/s tilt {r['tilt_end']:.1f}° fell={r['fell_at']}")
    gf = dict(vx_mean=float(np.mean([r["vx_mean"] for r in rows])),
              vx_rmse_mm=float(np.mean([r["vx_rmse"] * 1e3 for r in rows])),
              vx_rmse_std=float(np.std([r["vx_rmse"] * 1e3 for r in rows])),
              fell=sum(1 for r in rows if r["fell_at"] is not None))
    e5 = json.load(open(RESULTS / "e3d5_transfer.json"))
    allres = {"nominal": e5["nominal"]["vx_rmse_mm"],
              "foot_mlp(出类)": e5["corrected_mlp"]["vx_rmse_mm"],
              "结构(足)": e5["corrected_struct"]["vx_rmse_mm"],
              "genforce(入类)": gf["vx_rmse_mm"]}
    print("\n=== E3D-5c 主对照（MuJoCo 真机 vx RMSE mm/s）===")
    for k, v in allres.items():
        print(f"  {k:16s}: {v:.0f}")
    verdict = ("(a) 入类残差≥nominal: 表达类对了能修正, E3D-5 边界破"
               if gf["vx_rmse_mm"] <= e5["nominal"]["vx_rmse_mm"] * 1.1 else
               "(b) 入类≈nominal但≪足力出类: 出类有害/入类无害, 收益被反馈掩盖"
               if gf["vx_rmse_mm"] < e5["corrected_mlp"]["vx_rmse_mm"] * 0.85 else
               "(c) 入类仍劣: 一步拟合梯度即便入类也不可信")
    print(f"  → {verdict}")
    gf["verdict"] = verdict
    (RESULTS / "e3d5c_transfer.json").write_text(json.dumps(gf, indent=2))
    # 图
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    names = list(allres); vals = list(allres.values())
    cols = ["tab:gray", "tab:red", "tab:olive", "tab:green"]
    ax.bar(names, vals, color=cols)
    ax.axhline(e5["nominal"]["vx_rmse_mm"], color="k", ls=":", lw=1, label="nominal 基准")
    ax.set_ylabel("MuJoCo 真机 vx RMSE (mm/s)")
    ax.set_title("E3D-5c: 开广义力通道(入类残差) vs 足力通道(出类)\n" + verdict)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e3d5c_genforce.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d5c_genforce.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "transfer"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = build_standing_config(device="cpu", dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-5c [{args.stage}]")
    if args.stage == "fit":
        stage_fit(cfg, g, z_ref)
    else:
        stage_transfer(cfg, g, z_ref, device=args.device)


if __name__ == "__main__":
    main()
