"""E3D-10: 感知地形运动——heightmap → 自适应足高，跨过盲走会绊的隆起〔实验〕

Stage 2 第一块砖（接感知）：给策略一个**机体前向 heightmap**（便宜的可微外感知，proto-depth），
让它在崎岖地形上抬足跨障。核心对照=**盲 vs 感知**（同架构，盲的 heightmap 置零）：地形上
感知应摔得更少/走得更远——这就是"感知让你提前应对盲走会绊的地形"的最小可兑现。

贯穿全弧的梯度保真：地形=平滑高斯隆起（C∞，避免尖锐台阶的接触梯度爆炸），动作含**逐腿
摆动高度** dh（看到前方隆起→抬高对应足）。地形接入接触不改 contact_3d（足端 z 减地形高度算 gap）。

stage: train（盲/感知各训）/ 此版只做孪生侧（MuJoCo 地形闸门 = E3D-10b 后续）。
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
from srbd_standing import build_standing_config, level_error  # noqa: E402
from floating_base_srbd import srbd_step  # noqa: E402
from contact_3d import foot_contact_force_world  # noqa: E402
from gait_3d import GaitConfig, PHASE_OFF  # noqa: E402
from terrain import random_terrain, flat_terrain, make_grid, heightmap  # noqa: E402
from e3d4_gait_train import sample_init  # noqa: E402
from e3d3_standing_train import level_to_angle  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
DH_MAX = 0.08            # 逐腿摆动高度增量上限 [m]（h_swing=0.04 → 可达 0.12）
NG = 18                  # heightmap 网格点数 (6×3，覆盖前后足落点)


def perc_foot_plan(t_step, dLx, dext, dh, cfg, g):
    """逐腿摆动高度可调的 trot 足端规划（gait_3d.foot_plan 的感知版）。dLx/dext/dh:(B,4) 已限幅。"""
    B = dLx.shape[0]
    Lx = g.lx0 + dLx
    if not torch.is_tensor(t_step):
        t_step = dLx.new_full((B,), float(t_step))
    phi_g = (t_step.to(dLx.dtype) * cfg.dt / g.period) % 1.0
    phi = (phi_g[:, None] + dLx.new_tensor(PHASE_OFF)) % 1.0
    st = phi < g.duty
    fr = cfg.foot_rel_com
    s_st = phi / g.duty
    x_st = (0.5 - s_st) * Lx
    xd_st = -Lx / g.t_stance
    s_sw = ((phi - g.duty) / (1.0 - g.duty)).clamp(0.0, 1.0)
    k = (1.0 - g.duty) / g.duty
    a3, a2 = -2.0 * (1.0 + k), 3.0 * (1.0 + k)
    f = a3 * s_sw ** 3 + a2 * s_sw ** 2 - k * s_sw
    fp = 3 * a3 * s_sw ** 2 + 2 * a2 * s_sw - k
    x_sw = (-0.5 + f) * Lx
    xd_sw = (fp / ((1 - g.duty) * g.period)) * Lx
    h_sw = g.h_swing + dh                                       # (B,4) 逐腿摆动高度
    z_lift = h_sw * torch.sin(np.pi * s_sw)
    zd = h_sw * np.pi * torch.cos(np.pi * s_sw) / ((1 - g.duty) * g.period)
    x = torch.where(st, x_st, x_sw)
    xdot = torch.where(st, xd_st, xd_sw)
    z0 = fr[None, :, 2] - g.ext0 - dext
    z = torch.where(st, z0, z0 + z_lift)
    zdot = torch.where(st, torch.zeros_like(zd), zd)
    p_b = torch.stack([fr[None, :, 0] + x, fr[None, :, 1].expand(B, 4), z], dim=-1)
    pdot_b = torch.stack([xdot, torch.zeros_like(xdot), zdot], dim=-1)
    return p_b, pdot_b, st


def perc_step(state, t, dLx, dext, dh, cfg, g, terrain):
    """一步感知地形运动：足规划→世界足位/速→**减地形高度算 gap**→接触→wrench→srbd。"""
    p_b, pdot_b, stance = perc_foot_plan(t, dLx, dext, dh, cfg, g)
    R = quaternion_to_matrix(state.q)
    foot_w = state.p[:, None, :] + torch.einsum("bij,bkj->bki", R, p_b)
    w_world = torch.einsum("bij,bj->bi", R, state.w)
    r = foot_w - state.p[:, None, :]
    foot_v = (state.v[:, None, :]
              + torch.cross(w_world[:, None, :].expand_as(r), r, dim=-1)
              + torch.einsum("bij,bkj->bki", R, pdot_b))
    th = terrain.height(foot_w[:, :, :2])                       # (B,4) 足下地形高度
    foot_wc = torch.cat([foot_w[:, :, :2], (foot_w[:, :, 2] - th)[:, :, None]], dim=-1)
    out = foot_contact_force_world(foot_wc, foot_v, cfg.contact)
    f_each = out["f_world"]
    tau_body = torch.einsum("bji,bj->bi", R, torch.cross(r, f_each, dim=-1).sum(1))
    nxt = srbd_step(state, cfg.mass, cfg.I_body, cfg.I_body_inv, cfg.dt,
                    f_world=f_each.sum(1), tau_body=tau_body)
    return nxt, dict(stance=stance, foot_w=foot_w)


class PercPolicy(nn.Module):
    """obs = proprio(11) + heightmap(NG) → a∈R¹²=[dLx(4),dext(4),dh(4)]。盲版 heightmap 置零。"""

    def __init__(self, ng=NG, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(11 + ng, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 12))

    def forward(self, o):
        return self.net(o)


def observe_perc(state, phi, g, z_ref, hmap):
    R = quaternion_to_matrix(state.q)
    v_b = torch.einsum("bji,bj->bi", R, state.v)
    tilt = R[:, 2, :2]
    B = state.p.shape[0]
    ph = state.p.new_tensor([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)]).expand(B, 2)
    proprio = torch.cat([v_b[:, 0:1] - g.vx_cmd, v_b[:, 1:2], v_b[:, 2:3], tilt,
                         state.w, state.p[:, 2:3] - z_ref, ph], dim=-1)
    return torch.cat([proprio, hmap], dim=-1)


def split_act(a, g):
    dLx = g.dLx_max * torch.tanh(a[:, 0:4])
    dext = g.dext_max * torch.tanh(a[:, 4:8])
    dh = DH_MAX * torch.tanh(a[:, 8:12])
    return dLx, dext, dh


def loss_perc(state, a, g, z_ref, terrain):
    th = terrain.height(state.p[:, None, :2])[:, 0]                  # 机体下地形高度
    vx = state.v[:, 0]
    return (8.0 * (vx - g.vx_cmd) ** 2
            + 4.0 * (state.p[:, 2] - (th + z_ref)) ** 2
            + 2.0 * level_error(state.q)
            + 0.3 * (state.v[:, 1] ** 2 + state.v[:, 2] ** 2)
            + 0.05 * (state.w ** 2).sum(-1)
            + 1e-3 * (a ** 2).sum(-1)).mean()


def rollout(pol, cfg, g, z_ref, state, terrain, grid, H, tbptt, perceive):
    loss = state.p.new_zeros(())
    for t in range(H):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach()
        phi = (t * cfg.dt / g.period) % 1.0
        hmap = heightmap(state, terrain, grid) if perceive \
            else state.p.new_zeros(state.p.shape[0], grid.shape[0])
        a = pol(observe_perc(state, phi, g, z_ref, hmap))
        dLx, dext, dh = split_act(a, g)
        state, _ = perc_step(state, t, dLx, dext, dh, cfg, g, terrain)
        loss = loss + loss_perc(state, a, g, z_ref, terrain)
    return loss / H


def train_perc(cfg, g, z_ref, grid, perceive, iters, B=48, H=600, tbptt=150,
               lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = PercPolicy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist, gnorms = [], []
    for _ in range(iters):
        s = sample_init(cfg, g, z_ref, B, gen)
        terr = random_terrain(B, gen, cfg.device, cfg.dtype)
        loss = rollout(pol, cfg, g, z_ref, s, terr, grid, H, tbptt, perceive)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), clip).item()
        opt.step(); hist.append(loss.item()); gnorms.append(gn)
    return pol, hist, gnorms


@torch.no_grad()
def evaluate(pol, cfg, g, z_ref, grid, perceive, seed=7, B=64, H=6000):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, g, z_ref, B, gen)
    terr = random_terrain(B, gen, cfg.device, cfg.dtype)
    fell = torch.zeros(B, dtype=torch.bool, device=cfg.device)
    vxs = []
    for t in range(H):
        phi = (t * cfg.dt / g.period) % 1.0
        hmap = heightmap(s, terr, grid) if perceive else s.p.new_zeros(B, grid.shape[0])
        a = pol(observe_perc(s, phi, g, z_ref, hmap))
        dLx, dext, dh = split_act(a, g)
        s, _ = perc_step(s, t, dLx, dext, dh, cfg, g, terr)
        tilt = level_error(s.q)
        fell = fell | (tilt > 0.3)                                 # ~45° 倒
        vxs.append(s.v[:, 0])
    x_final = s.p[:, 0]
    vx = torch.stack(vxs, 0)                                       # (H,B)
    alive = ~fell
    return dict(fell_frac=float(fell.float().mean()),
                dist_mean=float(x_final.mean()),
                dist_alive=float(x_final[alive].mean()) if alive.any() else 0.0,
                vx_alive=float(vx[H // 2:, alive].mean()) if alive.any() else 0.0,
                x_final=x_final.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    grid = make_grid(device=cfg.device, dtype=cfg.dtype)
    print(f"E3D-10 感知地形运动 ({args.device}) NG={grid.shape[0]} dh_max={DH_MAX}")
    t0 = time.time()
    res = {}
    for perceive, name in [(False, "blind"), (True, "perceptive")]:
        evs = []
        gn_med = []
        for seed in range(args.seeds):
            pol, hist, gnorms = train_perc(cfg, g, z_ref, grid, perceive, args.iters, seed=seed)
            torch.save(pol.state_dict(), MODELS / f"e3d10_{name}_s{seed}.pt")
            ev = evaluate(pol, cfg, g, z_ref, grid, perceive, seed=99)
            evs.append(ev); gn_med.append(float(np.median(gnorms)))
            print(f"  {name:10s} s{seed} loss→{hist[-1]:.3f} |g|中位{np.median(gnorms):.1f} | "
                  f"摔{ev['fell_frac']*100:.0f}% 距离{ev['dist_mean']:.2f}m(存活{ev['dist_alive']:.2f}) "
                  f"vx{ev['vx_alive']:.2f} [{time.time()-t0:.0f}s]")
        res[name] = dict(
            fell_frac=float(np.mean([e["fell_frac"] for e in evs])),
            dist_mean=float(np.mean([e["dist_mean"] for e in evs])),
            dist_alive=float(np.mean([e["dist_alive"] for e in evs])),
            vx_alive=float(np.mean([e["vx_alive"] for e in evs])),
            gnorm_med=float(np.median(gn_med)))
    # 平地对照（感知策略在平地不应更差）——略，聚焦地形对照
    b, p = res["blind"], res["perceptive"]
    better = p["fell_frac"] <= b["fell_frac"] and p["dist_mean"] >= b["dist_mean"]
    verdict = (f"{'✅' if better else '⚠'} 地形上 感知 vs 盲: 摔 {p['fell_frac']*100:.0f}% vs "
               f"{b['fell_frac']*100:.0f}%、前进 {p['dist_mean']:.2f} vs {b['dist_mean']:.2f}m. "
               f"梯度有界(|g|中位 盲{b['gnorm_med']:.1f}/感知{p['gnorm_med']:.1f}=平滑地形+heightmap可微). "
               + ("感知让机器人提前抬足跨障——感知→控制回路在孪生侧兑现,下一步MuJoCo地形闸门(E3D-10b)。"
                  if better else "感知未显著优于盲——查地形难度/heightmap信息量/swing权限,或加trip罚。"))
    res["verdict"] = verdict
    print(f"  → {verdict}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3d10_perceptive.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    arms = ["blind", "perceptive"]; cols = ["tab:gray", "tab:green"]
    a = ax[0]
    a.bar(arms, [res[k]["fell_frac"] * 100 for k in arms], color=cols)
    a.set_ylabel("摔倒率 (%)"); a.set_title("崎岖地形摔倒率：感知 vs 盲")
    a = ax[1]
    x = np.arange(2); w = 0.35
    a.bar(x - w/2, [res[k]["dist_mean"] for k in arms], w, color=cols, label="全体均")
    a.bar(x + w/2, [res[k]["dist_alive"] for k in arms], w, alpha=0.5, color=cols, label="存活均")
    a.set_xticks(x); a.set_xticklabels(arms); a.set_ylabel("前进距离 (m)")
    a.set_title("前进距离"); a.legend(fontsize=8)
    fig.suptitle("E3D-10 感知地形运动：heightmap→逐腿足高，跨过盲走会绊的隆起\n"
                 f"(陡密隆起 0.05–0.11m) 摔倒 感知{p['fell_frac']*100:.0f}% vs 盲{b['fell_frac']*100:.0f}%、"
                 f"前进 {p['dist_mean']:.2f} vs {b['dist_mean']:.2f}m", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "e3d10_perceptive.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d10_perceptive.png'}\nsaved {RESULTS / 'e3d10_perceptive.json'}")


if __name__ == "__main__":
    main()
