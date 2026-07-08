"""Go2 单腿 FK + 牛顿 IK（numpy，4 腿批量；E3D-5 动作映射层）。

把 SRBD 孪生的足端目标（base 系）映射成 12 个关节角，喂 MuJoCo 位置伺服。
几何全部从 URDF 解析（不手填）；IK 用解析 FK + 数值雅可比的阻尼牛顿法
（5 迭代 + 热启动 1-2 迭代收敛到 <1e-9，免去手推符号的深夜陷阱——FK 即真值，
正确性由 ①FK vs MuJoCo site 逐位对账 ②IK 往返误差 双重验证）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_urdf import LEGS, load_go2


def _rx(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class LegKinematics:
    """逐腿链参数（URDF）：base→hip(Rx)→thigh(Ry)→calf(Ry)→foot(固定)。"""

    def __init__(self):
        m = load_go2()
        J = m.joints
        self.hip0 = np.stack([J[f"{l}_hip_joint"].origin_xyz for l in LEGS])
        self.thi0 = np.stack([J[f"{l}_thigh_joint"].origin_xyz for l in LEGS])
        self.cal0 = np.stack([J[f"{l}_calf_joint"].origin_xyz for l in LEGS])
        self.foo0 = np.stack([J[f"{l}_foot_joint"].origin_xyz for l in LEGS])
        lim = [J[f"{l}_{j}_joint"].limit for l in LEGS
               for j in ("hip", "thigh", "calf")]
        self.lo = np.array([x["lower"] for x in lim]).reshape(4, 3)
        self.hi = np.array([x["upper"] for x in lim]).reshape(4, 3)

    def fk(self, q):
        """q:(4,3) → 足端位置 (4,3)（base 系）。"""
        p = np.zeros((4, 3))
        for li in range(4):
            R1 = _rx(q[li, 0])
            R2 = _ry(q[li, 1])
            R3 = _ry(q[li, 2])
            p[li] = (self.hip0[li]
                     + R1 @ (self.thi0[li]
                             + R2 @ (self.cal0[li] + R3 @ self.foo0[li])))
        return p

    def ik(self, target, q0=None, iters=8, damp=1e-6):
        """target:(4,3) base 系足端目标 → q:(4,3)。阻尼牛顿 + 限位钳制。"""
        q = (np.tile([0.0, 0.8, -1.6], (4, 1)) if q0 is None else q0.copy())
        for _ in range(iters):
            p = self.fk(q)
            err = target - p                                   # (4,3)
            if np.abs(err).max() < 1e-10:
                break
            for li in range(4):                                # 数值雅可比 3×3
                Jn = np.zeros((3, 3))
                h = 1e-6
                for k in range(3):
                    qp = q[li].copy(); qp[k] += h
                    Jn[:, k] = (self._fk_one(li, qp) - p[li]) / h
                dq = np.linalg.solve(Jn.T @ Jn + damp * np.eye(3), Jn.T @ err[li])
                q[li] = np.clip(q[li] + dq, self.lo[li], self.hi[li])
        return q

    def _fk_one(self, li, qli):
        R1 = _rx(qli[0]); R2 = _ry(qli[1]); R3 = _ry(qli[2])
        return (self.hip0[li]
                + R1 @ (self.thi0[li] + R2 @ (self.cal0[li] + R3 @ self.foo0[li])))


if __name__ == "__main__":
    import mujoco
    from go2_mjcf_full import build
    kin = LegKinematics()
    mj = build()
    d = mujoco.MjData(mj)
    rng = np.random.default_rng(0)
    # ① FK vs MuJoCo site 对账（base 固定于原点单位姿态）
    emax = 0.0
    for _ in range(50):
        q = kin.lo + (kin.hi - kin.lo) * rng.uniform(0.1, 0.9, (4, 3))
        d.qpos[:] = 0; d.qpos[3] = 1.0
        d.qpos[7:] = q.ravel()
        mujoco.mj_kinematics(mj, d)
        for li, leg in enumerate(LEGS):
            sid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
            emax = max(emax, np.abs(d.site_xpos[sid] - kin.fk(q)[li]).max())
    print(f"① FK vs MuJoCo: 50 随机姿态最大误差 = {emax:.2e} m")
    # ② IK 往返——分两层：全限位域（存在多解/奇异，仅报告）与步态工作空间（须近零）
    for tag, lo, hi in [("全限位域(多解,仅参考)", 0.15, 0.85), ("步态工作空间", None, None)]:
        emax = 0.0
        qprev = None
        for _ in range(200):
            if lo is not None:
                q = kin.lo + (kin.hi - kin.lo) * rng.uniform(lo, hi, (4, 3))
            else:   # 步态范围: hip±0.3, thigh 0.4..1.3, calf −2.2..−1.0
                q = np.stack([rng.uniform(-0.3, 0.3, 4), rng.uniform(0.4, 1.3, 4),
                              rng.uniform(-2.2, -1.0, 4)], axis=1)
            p = kin.fk(q)
            q2 = kin.ik(p)        # 默认初值：跨不连续随机目标时热启动反而入错分支
            emax = max(emax, np.abs(kin.fk(q2) - p).max())
        print(f"② IK 往返[{tag}]: 200 目标最大足端误差 = {emax:.2e} m")
    # 步态实际使用方式 = 连续轨迹 + 热启动：用一段真实 foot_plan 轨迹验证
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynamics"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import torch
    from srbd_standing import build_standing_config
    from gait_3d import GaitConfig, foot_plan
    cfgS = build_standing_config(device="cpu", dtype=torch.float32)
    g = GaitConfig()
    c_body = cfgS.c_body.numpy()
    emax, qprev = 0.0, None
    for t in range(0, 1200, 2):
        p_b, _, _, _ = foot_plan(t, torch.zeros(1, 8), cfgS, g)
        tgt = p_b[0].numpy() + c_body              # COM 系 → base 系
        q2 = kin.ik(tgt, q0=qprev, iters=4)
        emax = max(emax, np.abs(kin.fk(q2) - tgt).max())
        qprev = q2
    print(f"③ 步态轨迹(连续+热启动, 4 迭代/步): 最大足端误差 = {emax:.2e} m")
