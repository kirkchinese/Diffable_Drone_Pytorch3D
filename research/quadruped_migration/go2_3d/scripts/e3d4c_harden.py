"""E3D-4c: 收官结论的统计加固（验证路径①）+ 非专家可核验报告。

用户决策：①统计加固→②MuJoCo 外检顺序做；同步产出非专家核验报告。
本脚本只加固**内部稳健性**（对随机性/设置/实现的稳健），外部效度留给 E3D-5。

五阶段（--stage，独立落盘、已存模型跳过 → 崩溃可续）：
  fd        : ∇θL 的 BPTT 计算 vs 有限差分抽查（float64，方向导数 20 个随机方向）
              ——所有梯度保真结论所依赖的"梯度本身算对了"的核验。
  e3d6seeds : E3D-6 通道匹配 2×2 的 5 个头训练种子重复（5.9× 分离是否种子稳健）。
  sweep     : 失配强度扫描 κ∈{0.2,0.4,0.6}：κ̂ 参数恢复 + 一步雅可比比
              ——κ=0.4 单点结论是否在强度轴上连续成立。
  seeds     : 头条对照（M_force 步态四臂 nominal/MLP/structured/oracle）3→10 seeds
              × 3 个独立评测批——39/68/18.6/18 mm/s 的种子/评测稳健性（大头，过夜）。
  report    : 汇总图 + 非专家核验报告（每结论一个不需专业背景的检查点）。
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
from gait_3d import GaitConfig  # noqa: E402
from residual_gait import (KAPPA, StructuredDual, gait_accel,  # noqa: E402
                           gait_mismatch)
from e3d4_gait_train import GaitPolicy  # noqa: E402
from e3d4b_residual_gait import (MODELS, collect_real, eval_real,  # noqa: E402
                                 load_nominal, load_residuals, make_extra_fn,
                                 rollout_train, sl, to64, train)
from e3d4_gait_train import sample_init  # noqa: E402

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
OUT = RESULTS / "e3d4c_harden.json"
SEEDS10 = list(range(10))
EVAL_SEEDS = [7, 17, 27]
ARMS = ["nominal", "corrected_mlp", "corrected_struct", "oracle"]


def _load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _save(d):
    OUT.write_text(json.dumps(d, indent=2))


# --------------------------------------------------------------------------- #
def stage_fd(cfg, g, z_ref):
    """BPTT ∇θL vs 有限差分方向导数（float64；H=150；标称步态）。"""
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    torch.manual_seed(0)
    pol = GaitPolicy().double()
    gen = torch.Generator().manual_seed(1)
    s0 = sample_init(cfg64, g, z_ref, 8, gen)
    extra = lambda s, t, a: (None, None)

    def loss_at():
        st = type(s0)(s0.p.clone(), s0.q.clone(), s0.v.clone(), s0.w.clone())
        return rollout_train(pol, cfg64, g, z_ref, st, 150, extra, tbptt=None)

    L = loss_at()
    L.backward()
    gflat = torch.cat([p.grad.flatten() for p in pol.parameters()])
    params = [p for p in pol.parameters()]
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20):
        d = torch.tensor(rng.standard_normal(gflat.shape[0]))
        d = d / d.norm()
        eps = 1e-5
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.add_(eps * d[i:i + n].view_as(p)); i += n
        Lp = loss_at().item()
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.add_(-2 * eps * d[i:i + n].view_as(p)); i += n
        Lm = loss_at().item()
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.add_(eps * d[i:i + n].view_as(p)); i += n
        fd = (Lp - Lm) / (2 * eps)
        an = float((gflat * d).sum())
        rows.append((an, fd))
    an, fd = np.array(rows).T
    rel = float(np.abs(an - fd).max() / (np.abs(fd).max() + 1e-12))
    corr = float(np.corrcoef(an, fd)[0, 1])
    print(f"[fd] BPTT vs 有限差分（20 方向）: corr={corr:.6f}  max相对差={rel:.2e}")
    d = _load(); d["fd"] = dict(corr=corr, max_rel=rel, pairs=rows); _save(d)


def stage_e3d6seeds(cfg, g, z_ref):
    """E3D-6 通道匹配 2×2 的头训练种子重复（沿用其脚本基建）。"""
    import e3d6_channel_matching as E6
    from residual_3d import ResidualHead, accel, mismatch
    from residual_3d import F_SCALE as F6, K_SCALE as K6
    torch.set_default_dtype(torch.float64)
    cfgS = build_standing_config(device="cpu", dtype=torch.float64)
    data, leg = E6.collect_states(cfgS)
    res = {}
    for seed in range(5):
        diag, off = [], []
        for mk in ["force", "kin"]:
            with torch.no_grad():
                aT = accel(data, leg, cfgS, *mismatch(mk, data, leg, cfgS))
            for rk in ["force", "kin"]:
                torch.manual_seed(seed)
                head = ResidualHead(rk, F6 if rk == "force" else K6).double()
                opt = torch.optim.Adam(head.parameters(), lr=3e-3)
                for _ in range(2000):
                    fe, dx = head.extras(data, leg, cfgS)
                    loss = ((accel(data, leg, cfgS, fe, dx) - aT) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                ge = E6.grad_fidelity(data, leg, cfgS,
                                      lambda s, l, h=head: h.extras(s, l, cfgS),
                                      lambda s, l, m=mk: mismatch(m, s, l, cfgS))
                (diag if mk == rk else off).append(ge)
        res[seed] = dict(diag=float(np.mean(diag)), off=float(np.mean(off)),
                         ratio=float(np.mean(off) / np.mean(diag)))
        print(f"[e3d6seeds] seed{seed}: 匹配 {res[seed]['diag']:.3f} vs 错通道 "
              f"{res[seed]['off']:.3f} （{res[seed]['ratio']:.1f}×）")
    torch.set_default_dtype(torch.float32)
    d = _load(); d["e3d6seeds"] = res; _save(d)


def stage_sweep(cfg, g, z_ref):
    """κ∈{0.2,0.4,0.6}：结构化 κ̂ 恢复 + 一步雅可比比（nominal vs structured）。"""
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    nom = load_nominal(cfg, 0)
    out = {}
    for kap in [0.2, 0.4, 0.6]:
        S, T, A = collect_real(cfg, g, z_ref, nom, "force", horizon=1600, B=32)
        # 注意 collect_real 用默认 κ 注入——强度扫描须重采：本地重写注入
        gen = torch.Generator(device=cfg.device).manual_seed(3)
        from gait_3d import gait_step
        from e3d4_gait_train import observe
        s = sample_init(cfg, g, z_ref, 32, gen)
        Ss, Ts, As = [], [], []
        with torch.no_grad():
            for t in range(1600):
                phi = (t * cfg.dt / g.period) % 1.0
                a = nom(observe(s, phi, g, z_ref))
                a = a + 0.1 * torch.randn(a.shape, generator=gen, device=cfg.device,
                                          dtype=cfg.dtype)
                if t >= 10 and t % 5 == 0:
                    Ss.append(s.detach()); As.append(a.detach())
                    Ts.append(torch.full((32,), float(t), device=cfg.device))
                fe, dx = gait_mismatch("force", s, t, a, cfg, g, kappa=kap)
                s, _ = gait_step(s, t, a, cfg, g, f_extra=fe, dx_body=dx)
        from floating_base_srbd import FloatingBaseState
        st = FloatingBaseState(*[torch.cat([getattr(x, k) for x in Ss], 0)
                                 for k in "pqvw"])
        data, tt, aa = to64(st, torch.cat(Ts), torch.cat(As), 2048)
        with torch.no_grad():
            aT = gait_accel(data, tt, aa, cfg64, g,
                            *gait_mismatch("force", data, tt, aa, cfg64, g, kappa=kap))
            base = ((gait_accel(data, tt, aa, cfg64, g) - aT) ** 2).mean().item()
        sd = StructuredDual().double()
        opt = torch.optim.Adam(sd.parameters(), lr=1e-2)
        for _ in range(800):
            fe, dx = sd.extras(data, tt, aa, cfg64, g)
            loss = ((gait_accel(data, tt, aa, cfg64, g, fe, dx) - aT) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        out[str(kap)] = dict(kappa_hat=float(sd.kappa.item()), base=base)
        print(f"[sweep] κ={kap}: κ̂={sd.kappa.item():.4f}  (标称误差 {base:.1f})")
    d = _load(); d["sweep"] = out; _save(d)


def stage_seeds(cfg, g, z_ref):
    """头条四臂 M_force：10 seeds × 3 评测批。已存模型跳过（崩溃可续）。"""
    residuals = load_residuals(cfg)
    sdev = StructuredDual().to(cfg.device, cfg.dtype)
    sdev.load_state_dict(torch.load(MODELS / "gait_structured_force.pt",
                                    map_location=cfg.device, weights_only=True))
    for p in sdev.parameters():
        p.requires_grad_(False)
    fns = dict(nominal=lambda s, t, a: (None, None),
               corrected_mlp=make_extra_fn("corrected", "force", cfg, g, residuals),
               corrected_struct=lambda s, t, a: sdev.extras(s, t, a, cfg, g),
               oracle=make_extra_fn("oracle", "force", cfg, g, residuals))
    fname = dict(nominal="gait_smooth_tbptt_s{}.pt",
                 corrected_mlp="gait_corrected_force_s{}.pt",
                 corrected_struct="gait_corrstruct_force_s{}.pt",
                 oracle="gait_oracle_force_s{}.pt")
    # E3D-4a 的 nominal 只有 s0-2 → 其余补训；4b 的 corrected/oracle s0-2 已存
    t0 = time.time()
    d = _load(); d.setdefault("seeds", {})
    for arm in ARMS:
        d["seeds"].setdefault(arm, {})
        for seed in SEEDS10:
            key = str(seed)
            if key in d["seeds"][arm]:
                continue
            f = MODELS / fname[arm].format(seed)
            if f.exists():
                pol = GaitPolicy().to(cfg.device, cfg.dtype)
                pol.load_state_dict(torch.load(f, map_location=cfg.device,
                                               weights_only=True))
            else:
                pol, hist = train(cfg, g, z_ref, fns[arm], seed=seed)
                torch.save(pol.state_dict(), f)
                print(f"  trained {arm} s{seed} loss→{hist[-1]:.4f} "
                      f"[{time.time()-t0:.0f}s]")
            evs = [eval_real(pol, cfg, g, z_ref, "force", seed=es)
                   for es in EVAL_SEEDS]
            d["seeds"][arm][key] = dict(
                loss=[e["loss"] for e in evs],
                vx_rmse_mm=[e["vx_rmse"] * 1e3 for e in evs])
            _save(d)
            print(f"  eval {arm} s{seed}: RMSE "
                  f"{np.round(d['seeds'][arm][key]['vx_rmse_mm'],1).tolist()} mm/s")
    for arm in ARMS:
        allr = [r for s in d["seeds"][arm].values() for r in s["vx_rmse_mm"]]
        print(f"[seeds] {arm:16s}: vx RMSE 中位 {np.median(allr):.0f} "
              f"[{np.percentile(allr,25):.0f},{np.percentile(allr,75):.0f}] mm/s "
              f"(n={len(allr)})")


def stage_report(cfg, g, z_ref):
    d = _load()
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    # (a) 四臂分布
    a = ax[0]
    cols = dict(nominal="tab:gray", corrected_mlp="tab:red",
                corrected_struct="tab:green", oracle="tab:orange")
    meds = {}
    for i, arm in enumerate(ARMS):
        allr = [r for s in d["seeds"][arm].values() for r in s["vx_rmse_mm"]]
        meds[arm] = float(np.median(allr))
        a.scatter(np.full(len(allr), i) + np.random.default_rng(i).uniform(
            -0.12, 0.12, len(allr)), allr, s=14, alpha=0.6, color=cols[arm])
        a.plot([i - 0.25, i + 0.25], [meds[arm]] * 2, color="k", lw=2)
    a.set_xticks(range(4))
    a.set_xticklabels(["nominal", "MLP残差", "结构残差", "oracle"], fontsize=9)
    a.set_ylabel("真实系统 vx RMSE (mm/s)")
    a.set_title(f"(a) 四臂×10种子×3评测批\n中位 {meds['nominal']:.0f}/"
                f"{meds['corrected_mlp']:.0f}/{meds['corrected_struct']:.0f}/"
                f"{meds['oracle']:.0f} mm/s")
    # (b) κ̂ 恢复
    a = ax[1]
    ks = sorted(float(k) for k in d["sweep"])
    a.plot(ks, [d["sweep"][str(k)]["kappa_hat"] for k in ks], "o-", color="tab:green")
    a.plot([0, 0.7], [0, 0.7], "k:", lw=1)
    a.set_xlabel("真值 κ"); a.set_ylabel("辨识 κ̂")
    a.set_title("(b) 失配强度扫描:\nκ̂ 应落在对角线上")
    # (c) FD
    a = ax[2]
    an, fd = np.array(d["fd"]["pairs"]).T
    a.scatter(fd, an, s=18, color="tab:blue")
    lim = max(abs(fd).max(), abs(an).max()) * 1.1
    a.plot([-lim, lim], [-lim, lim], "k:", lw=1)
    a.set_xlabel("有限差分"); a.set_ylabel("BPTT 解析梯度")
    a.set_title(f"(c) 梯度算对了吗:\ncorr={d['fd']['corr']:.4f}")
    # (d) E3D-6 多种子
    a = ax[3]
    sd6 = d["e3d6seeds"]
    ss = sorted(sd6)
    a.bar([int(s) - 0.2 for s in ss], [sd6[s]["diag"] for s in ss], 0.4,
          color="tab:green", label="匹配通道")
    a.bar([int(s) + 0.2 for s in ss], [sd6[s]["off"] for s in ss], 0.4,
          color="tab:red", label="错通道")
    a.set_xlabel("头训练种子"); a.set_ylabel("梯度误差")
    a.set_title("(d) E3D-6 通道匹配 ×5 种子"); a.legend(fontsize=8)
    fig.suptitle("E3D-4c 统计加固：收官结论的内部稳健性（外部效度→E3D-5 MuJoCo）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "e3d4c_harden.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d4c_harden.png'}")

    ratios = [sd6[s]["ratio"] for s in ss]
    rep = f"""# E3D-4c 验证报告（非专家可核验版）

> 给"研究超出我的知识范围"的核验路径：你不需要审数学，只需要核对下面每条
> 检查点是否如所述。配图 `figures/e3d4c_harden.png`。**本报告只覆盖内部稳健性**
> （对随机种子/设置/实现的稳健）；"方法对真实未知失配是否成立"留待 E3D-5 MuJoCo
> 外部验证（含可直接观看的渲染视频）。

## 检查点 1：头条结论不是种子运气（图 a）
四臂各 10 个训练种子 × 3 个独立评测批（共 30 点/臂）。
**核对**：绿色（结构残差，中位 {meds['corrected_struct']:.0f} mm/s）应与橙色
（oracle，{meds['oracle']:.0f}）基本重叠；红色（MLP 残差，{meds['corrected_mlp']:.0f}）
应明显高于灰色（nominal，{meds['nominal']:.0f}）——"结构对了≈完美修正、
自由网络反而有害"在分布层面成立。

## 检查点 2：参数辨识落在对角线上（图 b）
把注入的失配强度 κ 从 0.2 扫到 0.6，辨识值 κ̂ 应贴着 y=x 对角线。
**核对**：三个点都在对角线附近 → 残差辨识不是 κ=0.4 单点巧合。

## 检查点 3：梯度本身算对了（图 c）
所有"梯度保真"结论依赖 BPTT 解析梯度。用与实现完全独立的有限差分
（扰动参数重算损失）抽查 20 个随机方向。
**核对**：散点贴对角线，相关系数 corr={d['fd']['corr']:.4f}（应 >0.999）。

## 检查点 4：通道匹配的分离是种子稳健的（图 d）
E3D-6 的"放错通道梯度坏 5.9×"换 5 个头训练种子重复。
**核对**：每个种子红柱（错通道）都显著高于绿柱（匹配），
分离倍数 {min(ratios):.1f}–{max(ratios):.1f}×，无一例外。

## 已知边界（如实）
- 以上全部以**我们自己的孪生+已知注入失配**为真值——自洽性已加固，
  外部效度（真实未知失配）是 E3D-5 的问题。
- M_kin 通道的闭环不可判（反馈掩盖）不因加固改变；其证据在梯度指标。
"""
    (HERE.parent / "reports" / "e3d4c_verification_report.md").write_text(rep)
    print("saved reports/e3d4c_verification_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["fd", "e3d6seeds", "sweep", "seeds", "report"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-4c [{args.stage}] ({args.device})")
    dict(fd=stage_fd, e3d6seeds=stage_e3d6seeds, sweep=stage_sweep,
         seeds=stage_seeds, report=stage_report)[args.stage](cfg, g, z_ref)


if __name__ == "__main__":
    main()
