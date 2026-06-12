"""E3D-4b: 步态任务上的三师对照——E3D-7 边界的兑现实验。

E3D-7 机制预测：修正收益集中在"失配×任务可控子空间"交集。步态下纵向力失配（M_force）
直接打击速度跟踪、且 ΔLx 有对抗权限（可控）；M_kin 改触地几何同样直接进任务通道——
预测三师闭环对比在步态上**变可判**（站立上不可判，E3D-7 §11.4）。

四阶段（--stage，各自独立落盘防崩溃）：
  audit   : 训 oracle×2(seed0)，闸门 = L_real(标称策略)/L_real(oracle策略)。
  fit     : 标称基线+噪声在真实步态系统采闭环数据 → 拟 GaitDualHead（等效牛顿正则）。
  gradfid : 一步 ∂accel/∂a 全局 Frobenius + 策略梯度 ∇θL 全局 cos（H∈{150,800}）。
  transfer: 三师×3seeds×2失配（nominal 复用 E3D-4a 的 3 个 seeds），TBPTT-150，
            全部迁回真实系统，gap closure。
失配与 E3D-6/7 完全一致（κ=0.4 / δ=[0.02,0,−0.012]）。
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dynamics"))
sys.path.insert(0, str(HERE))
from srbd_standing import build_standing_config  # noqa: E402
from floating_base_srbd import FloatingBaseState  # noqa: E402
from gait_3d import GaitConfig, gait_step  # noqa: E402
from residual_gait import (KAPPA, KIN_OFF, GaitDualHead, StructuredDual,  # noqa: E402
                           gait_accel, gait_mismatch)
from e3d4_gait_train import (GaitPolicy, observe, loss_step, sample_init,  # noqa: E402
                             evaluate, MODELS as MODELS4A)

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
MODELS = RESULTS / "e3d4_models"
H_TRAIN, H_EVAL = 800, 3200
SEEDS = [0, 1, 2]
TEACHERS = ["nominal", "corrected", "oracle"]
K_N_EQ, LAM = 1.0e4, 1e-4
TCOL = dict(nominal="tab:gray", corrected="tab:green", oracle="tab:orange")


def make_extra_fn(teacher, mk, cfg, g, residuals):
    if teacher == "nominal":
        return lambda s, t, a: (None, None)
    if teacher == "oracle":
        return lambda s, t, a: gait_mismatch(mk, s, t, a, cfg, g)
    dual = residuals[mk]
    return lambda s, t, a: dual.extras(s, t, a, cfg, g)


def rollout_train(pol, cfg, g, z_ref, state, horizon, extra_fn, tbptt=150):
    loss = state.p.new_zeros(())
    for t in range(horizon):
        if tbptt and t > 0 and t % tbptt == 0:
            state = state.detach()
        phi = (t * cfg.dt / g.period) % 1.0
        a = pol(observe(state, phi, g, z_ref))
        fe, dx = extra_fn(state, t, a)
        state, _ = gait_step(state, t, a, cfg, g, f_extra=fe, dx_body=dx)
        loss = loss + loss_step(state, a, g, z_ref)
    return loss / horizon


def train(cfg, g, z_ref, extra_fn, iters=120, B=48, lr=3e-3, seed=0, clip=1.0):
    torch.manual_seed(seed)
    pol = GaitPolicy().to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 100)
    hist = []
    for _ in range(iters):
        s = sample_init(cfg, g, z_ref, B, gen)
        loss = rollout_train(pol, cfg, g, z_ref, s, H_TRAIN, extra_fn)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), clip)
        opt.step(); hist.append(loss.item())
    return pol, hist


@torch.no_grad()
def eval_real(pol, cfg, g, z_ref, mk, horizon=H_EVAL, seed=7, B=32):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, g, z_ref, B, gen)
    loss = 0.0; vxs = []
    for t in range(horizon):
        phi = (t * cfg.dt / g.period) % 1.0
        a = pol(observe(s, phi, g, z_ref))
        fe, dx = gait_mismatch(mk, s, t, a, cfg, g)
        s, _ = gait_step(s, t, a, cfg, g, f_extra=fe, dx_body=dx)
        loss += loss_step(s, a, g, z_ref).item()
        vxs.append(s.v[:, 0].mean().item())
    vxs = np.array(vxs); half = horizon // 2
    return dict(loss=loss / horizon, vxs=vxs,
                vx_rmse=float(np.sqrt(((vxs - g.vx_cmd)[half:] ** 2).mean())))


def load_nominal(cfg, seed):
    pol = GaitPolicy().to(cfg.device, cfg.dtype)
    pol.load_state_dict(torch.load(MODELS4A / f"gait_smooth_tbptt_s{seed}.pt",
                                   map_location=cfg.device, weights_only=True))
    return pol


def collect_real(cfg, g, z_ref, pol, mk, horizon=H_EVAL, B=64, every=5, noise=0.1, seed=3):
    """闭环采 (state, t, action)。noise 加在未限幅动作上（tanh 前），覆盖动作域。"""
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    s = sample_init(cfg, g, z_ref, B, gen)
    S, T, A = [], [], []
    with torch.no_grad():
        for t in range(horizon):
            phi = (t * cfg.dt / g.period) % 1.0
            a = pol(observe(s, phi, g, z_ref))
            a = a + noise * torch.randn(a.shape, generator=gen, device=cfg.device,
                                        dtype=cfg.dtype)
            if t >= 10 and t % every == 0:
                S.append(s.detach()); A.append(a.detach())
                T.append(torch.full((B,), float(t), device=cfg.device))
            fe, dx = gait_mismatch(mk, s, t, a, cfg, g)
            s, _ = gait_step(s, t, a, cfg, g, f_extra=fe, dx_body=dx)
    st = FloatingBaseState(*[torch.cat([getattr(x, k) for x in S], 0) for k in "pqvw"])
    return st, torch.cat(T, 0), torch.cat(A, 0)


def to64(st, tt, aa, n, seed=0):
    idx = torch.randperm(st.p.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    f = lambda x: x[idx].detach().double().cpu()
    return (FloatingBaseState(f(st.p), f(st.q), f(st.v), f(st.w)),
            tt[idx].detach().double().cpu(), aa[idx].detach().double().cpu())


def sl(st, tt, aa, a, b):
    f = lambda x: x[a:b]
    return FloatingBaseState(f(st.p), f(st.q), f(st.v), f(st.w)), tt[a:b], aa[a:b]


# ============================== stages ==============================
def stage_audit(cfg, g, z_ref):
    print("[audit] 闸门 = L_real(标称策略)/L_real(oracle策略)")
    nom = load_nominal(cfg, 0)
    out = {}
    for mk in ["force", "kin"]:
        t0 = time.time()
        ora, hist = train(cfg, g, z_ref, make_extra_fn("oracle", mk, cfg, g, {}), seed=0)
        torch.save(ora.state_dict(), MODELS / f"gait_oracle_{mk}_s0.pt")
        rn = eval_real(nom, cfg, g, z_ref, mk)
        ro = eval_real(ora, cfg, g, z_ref, mk)
        gate = rn["loss"] / ro["loss"]
        out[mk] = dict(oracle_train_final=hist[-1], L_nom=rn["loss"], L_ora=ro["loss"],
                       gate=gate, vx_rmse_nom=rn["vx_rmse"], vx_rmse_ora=ro["vx_rmse"])
        print(f"  M_{mk:5s}: oracle 训到 {hist[-1]:.4f}；真实系统 eval "
              f"nominal {rn['loss']:.4f}(RMSE {rn['vx_rmse']*1e3:.0f}mm/s) vs "
              f"oracle {ro['loss']:.4f}(RMSE {ro['vx_rmse']*1e3:.0f}mm/s) → "
              f"gate={gate:.2f}× [{time.time()-t0:.0f}s]")
    ok = all(v["gate"] > 1.3 for v in out.values())
    print(f"  → {'✅ 闭环可判，继续' if ok else '⚠ 仍受反馈掩盖'}")
    (RESULTS / "e3d4b_audit.json").write_text(json.dumps(out, indent=2))


def stage_fit(cfg, g, z_ref, n_train=4096, iters=2000):
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    nom = load_nominal(cfg, 0)
    out = {}
    for mk in ["force", "kin"]:
        t0 = time.time()
        S, T, A = collect_real(cfg, g, z_ref, nom, mk)
        data, tt, aa = to64(S, T, A, n_train + n_train // 4)
        dtr, ttr, atr = sl(data, tt, aa, 0, n_train)
        dho, tho, aho = sl(data, tt, aa, n_train, data.p.shape[0])
        with torch.no_grad():
            aT = gait_accel(dtr, ttr, atr, cfg64, g,
                            *gait_mismatch(mk, dtr, ttr, atr, cfg64, g))
            aTh = gait_accel(dho, tho, aho, cfg64, g,
                             *gait_mismatch(mk, dho, tho, aho, cfg64, g))
            base = ((gait_accel(dtr, ttr, atr, cfg64, g) - aT) ** 2).mean().item()
        torch.manual_seed(0)
        dual = GaitDualHead().double()
        opt = torch.optim.Adam(dual.parameters(), lr=3e-3)
        for _ in range(iters):
            fe, dx = dual.extras(dtr, ttr, atr, cfg64, g)
            fit = ((gait_accel(dtr, ttr, atr, cfg64, g, fe, dx) - aT) ** 2).mean()
            reg = (fe ** 2).mean() + ((K_N_EQ * dx) ** 2).mean()
            opt.zero_grad(); (fit + LAM * reg).backward(); opt.step()
        with torch.no_grad():
            fe, dx = dual.extras(dtr, ttr, atr, cfg64, g)
            aP = gait_accel(dtr, ttr, atr, cfg64, g, fe, dx)
            fit_tr = ((aP - aT) ** 2).mean().item()
            feh, dxh = dual.extras(dho, tho, aho, cfg64, g)
            fit_ho = ((gait_accel(dho, tho, aho, cfg64, g, feh, dxh) - aTh) ** 2).mean().item()
            C_f = (aP - gait_accel(dtr, ttr, atr, cfg64, g, None, dx)).norm(dim=-1).mean().item()
            C_k = (aP - gait_accel(dtr, ttr, atr, cfg64, g, fe, None)).norm(dim=-1).mean().item()
        rho = (C_f if mk == "force" else C_k) / (C_f + C_k + 1e-12)
        torch.save(dual.state_dict(), MODELS / f"gait_residual_{mk}.pt")
        out[mk] = dict(fit_train=fit_tr, fit_holdout=fit_ho, base=base,
                       C_f=C_f, C_k=C_k, rho=rho)
        print(f"  [fit M_{mk:5s}] fit {fit_tr:.3e}/{fit_ho:.3e} (标称 {base:.3e}) "
              f"ρ(正确头)={rho:.3f} [{time.time()-t0:.0f}s]")
    (RESULTS / "e3d4b_fit.json").write_text(json.dumps(out, indent=2))


def load_residuals(cfg):
    res = {}
    for mk in ["force", "kin"]:
        d = GaitDualHead()
        d.load_state_dict(torch.load(MODELS / f"gait_residual_{mk}.pt",
                                     map_location="cpu", weights_only=True))
        res[mk] = d.to(cfg.device, cfg.dtype).eval()
        for p in res[mk].parameters():
            p.requires_grad_(False)
    return res


def stage_gradfid(cfg, g, z_ref):
    residuals = load_residuals(cfg)
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    res64 = {}
    for mk in ["force", "kin"]:
        d = GaitDualHead()
        d.load_state_dict(torch.load(MODELS / f"gait_residual_{mk}.pt",
                                     map_location="cpu", weights_only=True))
        res64[mk] = d.double().eval()
    out = {}
    # —— 一步雅可比 ∂accel/∂a(8) 全局 Frobenius（部署态）——
    nom = load_nominal(cfg, 0)
    for mk in ["force", "kin"]:
        S, T, A = collect_real(cfg, g, z_ref, nom, mk, horizon=800, B=16)
        data, tt, aa = to64(S, T, A, 64)
        def jac(extra_fn):
            J = []
            for i in range(48):
                d1 = FloatingBaseState(data.p[i:i+1], data.q[i:i+1],
                                       data.v[i:i+1], data.w[i:i+1])
                a1 = aa[i:i+1].clone().requires_grad_(True)
                fe, dx = extra_fn(d1, tt[i:i+1], a1)
                acc = gait_accel(d1, tt[i:i+1], a1, cfg64, g, fe, dx)[0]
                J.append(torch.stack([torch.autograd.grad(acc[j], a1,
                         retain_graph=True)[0][0] for j in range(6)]))
            return torch.stack(J)
        JT = jac(lambda s, t, a, m=mk: gait_mismatch(m, s, t, a, cfg64, g))
        JN = jac(lambda s, t, a: (None, None))
        JC = jac(lambda s, t, a, d=res64[mk]: d.extras(s, t, a, cfg64, g))
        jn = ((JN - JT).norm() / JT.norm()).item()
        jc = ((JC - JT).norm() / JT.norm()).item()
        out[mk] = dict(jac_nom=jn, jac_cor=jc)
        print(f"  [jac M_{mk:5s}] 全局 ∂accel/∂a 比: nominal {jn:.3f} → corrected {jc:.3f}"
              f" ({(1-jc/jn)*100:.0f}%↓)")
    # —— 策略梯度 ∇θL 全局 cos × 视野 ——
    def pgrad(pol, extra_fn, H):
        for p in pol.parameters():
            p.grad = None
        gen = torch.Generator(device=cfg.device).manual_seed(11)
        s = sample_init(cfg, g, z_ref, 24, gen)
        rollout_train(pol, cfg, g, z_ref, s, H, extra_fn, tbptt=None).backward()
        return torch.cat([p.grad.flatten() for p in pol.parameters()]).detach()
    pts = []
    for sd in [0, 1]:
        torch.manual_seed(sd)
        pts.append((f"rand{sd}", GaitPolicy().to(cfg.device, cfg.dtype)))
    pts.append(("nominal-trained", nom))
    for mk in ["force", "kin"]:
        fns = {t: make_extra_fn(t, mk, cfg, g, residuals) for t in TEACHERS}
        out[mk + "_pg"] = {}
        for H in [150, 800]:
            G = {t: torch.cat([pgrad(p, fns[t], H) for _, p in pts]) for t in TEACHERS}
            rec = {}
            for t in ["nominal", "corrected"]:
                rec[t] = dict(
                    cos=float(F.cosine_similarity(G[t], G["oracle"], dim=0).item()),
                    rel=float(((G[t] - G["oracle"]).norm() / G["oracle"].norm()).item()))
            out[mk + "_pg"][H] = rec
            print(f"  [∇θL M_{mk:5s} H={H}] nominal cos={rec['nominal']['cos']:+.3f} "
                  f"rel={rec['nominal']['rel']:.2f} | corrected "
                  f"cos={rec['corrected']['cos']:+.3f} rel={rec['corrected']['rel']:.2f}")
    (RESULTS / "e3d4b_gradfid.json").write_text(json.dumps(out, indent=2))


def stage_transfer(cfg, g, z_ref):
    residuals = load_residuals(cfg)
    runs, t0 = {}, time.time()
    for seed in SEEDS:                               # nominal 复用 E3D-4a
        runs[("-", "nominal", seed)] = load_nominal(cfg, seed)
    for mk in ["force", "kin"]:
        for teacher in ["corrected", "oracle"]:
            for seed in SEEDS:
                f = MODELS / f"gait_{teacher}_{mk}_s{seed}.pt"
                if f.exists():
                    pol = GaitPolicy().to(cfg.device, cfg.dtype)
                    pol.load_state_dict(torch.load(f, map_location=cfg.device,
                                                   weights_only=True))
                else:
                    pol, hist = train(cfg, g, z_ref,
                                      make_extra_fn(teacher, mk, cfg, g, residuals),
                                      seed=seed)
                    torch.save(pol.state_dict(), f)
                    print(f"  trained {teacher} M_{mk} s{seed} loss→{hist[-1]:.4f} "
                          f"[{time.time()-t0:.0f}s]")
                runs[(mk, teacher, seed)] = pol
    print("\n[eval in REAL gait system]")
    summary, evals = {}, {}
    for mk in ["force", "kin"]:
        summary[mk] = {}
        for teacher in TEACHERS:
            es = []
            for seed in SEEDS:
                pol = runs[("-" if teacher == "nominal" else mk, teacher, seed)]
                e = eval_real(pol, cfg, g, z_ref, mk)
                evals[(mk, teacher, seed)] = e
                es.append(e)
            summary[mk][teacher] = dict(
                loss_mean=float(np.mean([e["loss"] for e in es])),
                loss_std=float(np.std([e["loss"] for e in es])),
                vx_rmse_mm=float(np.mean([e["vx_rmse"] * 1e3 for e in es])))
            m = summary[mk][teacher]
            print(f"  M_{mk:5s} {teacher:9s}: loss {m['loss_mean']:.4f}±{m['loss_std']:.4f}"
                  f"  vx RMSE {m['vx_rmse_mm']:.0f}mm/s")
        Ln, Lc, Lo = (summary[mk][t]["loss_mean"] for t in TEACHERS)
        summary[mk]["gap_closure"] = gc = float((Ln - Lc) / (Ln - Lo + 1e-12))
        print(f"  M_{mk:5s} gap closure = ({Ln:.4f}−{Lc:.4f})/({Ln:.4f}−{Lo:.4f}) = {gc:.2f}")
    (RESULTS / "e3d4b_transfer.json").write_text(json.dumps(summary, indent=2))
    # —— 图 ——
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    t = np.arange(H_EVAL) * 1e-3
    for row, mk in enumerate(["force", "kin"]):
        a = ax[row, 0]
        for teacher in TEACHERS:
            a.plot(t, evals[(mk, teacher, 0)]["vxs"], color=TCOL[teacher], label=teacher,
                   lw=1.2)
        a.axhline(g.vx_cmd, color="k", ls=":", lw=1, label="vx*")
        a.set_title(f"M_{mk}: 真实系统 vx(t) (seed0)"); a.set_xlabel("t (s)")
        if row == 0:
            a.legend(fontsize=8)
        a = ax[row, 1]
        for i, teacher in enumerate(TEACHERS):
            ls = [evals[(mk, teacher, s)]["loss"] for s in SEEDS]
            a.bar(i, np.mean(ls), 0.6, yerr=np.std(ls), color=TCOL[teacher], capsize=4)
        a.set_xticks(range(3)); a.set_xticklabels(TEACHERS)
        rms = [summary[mk][t]["vx_rmse_mm"] for t in TEACHERS]
        a.set_title(f"M_{mk}: 真实系统 eval loss (3 seeds); vx RMSE "
                    f"{rms[0]:.0f}/{rms[1]:.0f}/{rms[2]:.0f} mm/s")
    fig.suptitle("E3D-4b: 三师×步态 — M_force 可判(oracle≪nominal) 但 MLP 残差头未接住"
                 "(corrected 更差)；M_kin 仍被反馈掩盖", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "e3d4b_transfer.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG / 'e3d4b_transfer.png'}")


def stage_structured(cfg, g, z_ref):
    """收官判别：结构化参数残差（κ̂ + δ̂，4 参数）。若它兑现收益而自由 MLP 头不能
    （M_force 闭环 68 vs nominal 39 vs oracle 18 mm/s），败因=头质量而非修正概念。"""
    cfg64 = build_standing_config(device="cpu", dtype=torch.float64)
    nom = load_nominal(cfg, 0)
    fitted = {}
    for mk in ["force", "kin"]:
        S, T, A = collect_real(cfg, g, z_ref, nom, mk)
        dtr, ttr, atr = to64(S, T, A, 4096)
        with torch.no_grad():
            aT = gait_accel(dtr, ttr, atr, cfg64, g,
                            *gait_mismatch(mk, dtr, ttr, atr, cfg64, g))
        sd = StructuredDual().double()
        opt = torch.optim.Adam(sd.parameters(), lr=1e-2)
        for _ in range(800):
            fe, dx = sd.extras(dtr, ttr, atr, cfg64, g)
            loss = ((gait_accel(dtr, ttr, atr, cfg64, g, fe, dx) - aT) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fe, dx = sd.extras(dtr, ttr, atr, cfg64, g)
            fit = ((gait_accel(dtr, ttr, atr, cfg64, g, fe, dx) - aT) ** 2).mean().item()
        torch.save(sd.state_dict(), MODELS / f"gait_structured_{mk}.pt")
        fitted[mk] = dict(fit=fit, kappa=float(sd.kappa.item()),
                          delta=[float(x) for x in sd.delta])
        print(f"  [structured M_{mk:5s}] fit={fit:.3e}  κ̂={sd.kappa.item():.4f}"
              f"(真值 {KAPPA}) δ̂={np.round([float(x) for x in sd.delta],4).tolist()}"
              f"(真值 {list(KIN_OFF)})")
    # M_force（可判通道）上 corrected-structured ×3 seeds 训练+迁移
    mk = "force"
    sdev = StructuredDual().to(cfg.device, cfg.dtype)
    sdev.load_state_dict(torch.load(MODELS / f"gait_structured_{mk}.pt",
                                    map_location=cfg.device, weights_only=True))
    for p in sdev.parameters():
        p.requires_grad_(False)
    extra = lambda s, t, a: sdev.extras(s, t, a, cfg, g)
    es, t0 = [], time.time()
    for seed in SEEDS:
        pol, hist = train(cfg, g, z_ref, extra, seed=seed)
        torch.save(pol.state_dict(), MODELS / f"gait_corrstruct_{mk}_s{seed}.pt")
        e = eval_real(pol, cfg, g, z_ref, mk)
        es.append(e)
        print(f"  corrected-structured M_{mk} s{seed}: 训到 {hist[-1]:.4f} → "
              f"真实 loss={e['loss']:.4f} vx RMSE={e['vx_rmse']*1e3:.0f}mm/s "
              f"[{time.time()-t0:.0f}s]")
    out = dict(fitted=fitted, transfer_force=dict(
        loss_mean=float(np.mean([e["loss"] for e in es])),
        loss_std=float(np.std([e["loss"] for e in es])),
        vx_rmse_mm=float(np.mean([e["vx_rmse"] * 1e3 for e in es]))))
    print(f"  M_force corrected-structured: loss {out['transfer_force']['loss_mean']:.4f}"
          f"±{out['transfer_force']['loss_std']:.4f}  vx RMSE "
          f"{out['transfer_force']['vx_rmse_mm']:.0f}mm/s "
          f"(对照: nominal 39 / MLP-corrected 68 / oracle 18)")
    (RESULTS / "e3d4b_structured.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "fit", "gradfid", "transfer", "structured"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = build_standing_config(device=args.device, dtype=torch.float32)
    g = GaitConfig()
    z_ref = cfg.rest_height + g.ext0 - 0.004
    print(f"E3D-4b [{args.stage}] ({args.device})")
    dict(audit=stage_audit, fit=stage_fit, gradfid=stage_gradfid,
         transfer=stage_transfer, structured=stage_structured)[args.stage](cfg, g, z_ref)


if __name__ == "__main__":
    main()
