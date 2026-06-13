"""E3D-5 汇总图：合成(已知失配) vs 真实(MuJoCo 未知失配) 的对照——方法边界的全景。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _plotstyle
_plotstyle.use_cjk()
R = HERE.parent / "results"

t5 = json.load(open(R / "e3d5_transfer.json"))
f5 = json.load(open(R / "e3d5_fit.json"))
h4 = json.load(open(R / "e3d4c_harden.json"))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) 合成 vs 真实：结构残差的命运反转
a = ax[0]
synth = {arm: np.median([r for s in h4["seeds"][arm].values() for r in s["vx_rmse_mm"]])
         for arm in ["nominal", "corrected_struct", "oracle"]}
real = {arm: t5[arm]["vx_rmse_mm"] for arm in ["nominal", "corrected_struct"]}
labels = ["nominal", "结构残差", "oracle"]
x = np.arange(3); w = 0.36
a.bar(x - w/2, [synth["nominal"], synth["corrected_struct"], synth["oracle"]], w,
      color="tab:blue", label="合成(已知失配)")
a.bar(x[:2] + w/2, [real["nominal"], real["corrected_struct"]], w,
      color="tab:orange", label="真实 MuJoCo(未知失配)")
a.set_xticks(x); a.set_xticklabels(labels)
a.set_ylabel("vx RMSE (mm/s)")
a.set_title("(a) 结构残差的命运反转\n合成: 11≈oracle13 (赢) | 真实: 101≈nominal92 (无收益)")
a.legend(fontsize=8)

# (b) 真实失配的拟合与通道归因
a = ax[1]
a.bar([0, 1], [f5["mlp"]["fit_holdout"], f5["structured"]["fit_holdout"]],
      color=["tab:red", "tab:green"], width=0.5)
a.axhline(f5["base_holdout"], color="k", ls="--", lw=1,
          label=f"未修正 {f5['base_holdout']:.0f}")
a.set_xticks([0, 1]); a.set_xticklabels(["MLP\n(降97%)", "结构\n(降83%)"])
a.set_ylabel("孪生vs MuJoCo 加速度 MSE")
a.set_title("(b) 残差能拟合真实失配\n(但真实失配 ≠ 严格参数化, 故结构降幅<MLP)")
a.legend(fontsize=8)
a2 = a.twinx()
cf = [f5["mlp"]["C_f"], f5["structured"]["C_f"]]
ck = [f5["mlp"]["C_k"], f5["structured"]["C_k"]]
a2.plot([0, 1], cf, "s-", color="tab:gray", label="C_force")
a2.plot([0, 1], ck, "^-", color="tab:purple", label="C_kin")
a2.set_ylabel("通道归因"); a2.legend(fontsize=7, loc="upper right")
a2.text(0.5, max(ck) * 0.9, "真实失配以运动学为主\n(C_kin≫C_force)",
        fontsize=8, ha="center", color="tab:purple")

# (c) MuJoCo 上三臂 vx(t)
a = ax[2]
align = json.load(open(R / "e3d5_align.json"))
a.axhline(0.3, color="k", ls=":", lw=1, label="vx*")
for arm, col, src in [("nominal", "tab:gray", None),
                      ("corrected_struct", "tab:green", None),
                      ("corrected_mlp", "tab:red", None)]:
    rmse = t5[arm]["vx_rmse_mm"]
    a.bar(0, 0, color=col, label=f"{arm} ({rmse:.0f}mm/s)")  # 仅图例
nv = align["nominal_vxs"]
t = np.arange(len(nv)) * 8 * 0.002
a.plot(t, nv, color="tab:gray")
a.set_xlabel("t (s)"); a.set_ylabel("vx (m/s)")
a.set_title("(c) MuJoCo 真实系统 vx(t)\nnominal 稳跟; 摔倒 0/0/0——都不摔, 差在跟踪")
a.legend(fontsize=8)

fig.suptitle("E3D-5 MuJoCo 外部验证：残差能诊断真实失配(以运动学为主)，"
             "但闭环 sim2real 收益不转移——方法的诚实边界", fontsize=12)
fig.tight_layout()
out = HERE.parent / "figures" / "e3d5_external.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
