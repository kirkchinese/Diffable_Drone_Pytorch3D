"""MuJoCo 全身 Go2 桥接层（E3D-5 外部验证）：真实未知失配的"真机"。

设计原则：**孪生与 MuJoCo 共享同一套控制器栈**（gait_3d 足端规划 → 解析腿 IK →
关节 PD），唯一不同的是动力学本体（SRBD 运动学足 vs 全身 18-DoF + 真接触 + 腿惯量
+ PD 跟踪滞后 + dt 2ms）。这个差就是 E3D-5 要让残差去解释的"真实未知失配"。

预注册坑：
  ①IK 符号/分支错 → 对 MuJoCo FK 做往返验证（<1mm 才放行）；
  ②COM 速度须用 mj_subtreeVel（全身质心，与孪生 p=COM 约定对齐）；
  ③mujoco free joint 角速度在体系（与孪生 w 约定一致，已在 E3D-1 钉死同款约定）；
  ④ext0 穿透深度在真接触下变成"蹬地力指令"，高度偏置是预期失配，不调参掩盖。
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from floating_base_srbd import FloatingBaseState

_HERE = Path(__file__).resolve().parent
SCENE = _HERE.parent / "assets" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

# Go2 腿几何（与 parameters/go2_model_summary 一致；菜单模型同官方 URDF）
HIP_X, HIP_Y = 0.1934, 0.0465          # 髋安装点（躯干系）
D_LAT = 0.0955                          # hip→thigh 侧向偏置（左+右−）
L_THIGH = 0.213
# 菜单模型足心在 calf 系为 (−0.002, 0, −0.213)：等效 calf 连杆 + 固定角偏移 γ
L_CALF = float(np.hypot(0.002, 0.213))
GAMMA = float(np.arctan2(0.002, 0.213))
LEGS = ["FL", "FR", "RL", "RR"]
SIGN_X = dict(FL=1, FR=1, RL=-1, RR=-1)
SIGN_Y = dict(FL=1, FR=-1, RL=1, RR=-1)


def leg_ik(p_trunk: np.ndarray, leg: str) -> np.ndarray:
    """躯干系足端位置 → [hip, thigh, calf] 关节角（解析 3-DoF IK，Go2 膝负弯）。"""
    sx, sy = SIGN_X[leg], SIGN_Y[leg]
    px = p_trunk[0] - sx * HIP_X
    py = p_trunk[1] - sy * HIP_Y
    pz = p_trunk[2]
    d = sy * D_LAT
    # 髋滚转：把足端转进腿矢状面
    L2 = py * py + pz * pz
    Lp = np.sqrt(max(L2 - d * d, 1e-12))
    q1 = np.arctan2(py, -pz) - np.arctan2(d, Lp)
    # 矢状面 2 连杆：余弦定理求膝内角 θ（直腿 θ=π → q3=0；Go2 膝负弯 q3=θ−π≤0）
    r2 = px * px + Lp * Lp
    c_int = (L_THIGH ** 2 + L_CALF ** 2 - r2) / (2 * L_THIGH * L_CALF)
    q3_eff = np.arccos(np.clip(c_int, -1.0, 1.0)) - np.pi
    q2 = np.arctan2(-px, Lp) - np.arctan2(
        L_CALF * np.sin(q3_eff), L_THIGH + L_CALF * np.cos(q3_eff))
    return np.array([q1, q2, q3_eff - GAMMA])               # 真实膝角 = 等效角 − γ


class MJGo2:
    """MuJoCo Go2 + 关节 PD + 与孪生同构的状态读出。"""

    def __init__(self, kp: float = 60.0, kd: float = 2.0, c_body=None):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.kp, self.kd = kp, kd
        self.dt = self.m.opt.timestep                       # 2 ms
        self.c_body = np.zeros(3) if c_body is None else np.asarray(c_body)
        order = [f"{leg}_{j}" for leg in LEGS for j in ("hip", "thigh", "calf")]
        self.act_ids = np.array([mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                                 for i in range(self.m.nu)])
        self.perm = np.array([list(self.act_ids).index(n) for n in order])
        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")
                for n in order]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jids])
        self.tlim = self.m.actuator_ctrlrange[:, 1][self.perm]

    def reset(self, keyframe: int = 0):
        mujoco.mj_resetDataKeyframe(self.m, self.d, keyframe)
        mujoco.mj_forward(self.m, self.d)

    def set_base(self, pos, quat_wxyz):
        self.d.qpos[0:3] = pos
        self.d.qpos[3:7] = quat_wxyz
        self.d.qvel[:6] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_joints_ik(self, p_trunk_targets: np.ndarray):
        """(4,3) 躯干系足端目标 → IK 直接写关节角（初始化用）。"""
        for li, leg in enumerate(LEGS):
            self.d.qpos[self.qadr[3 * li:3 * li + 3]] = leg_ik(p_trunk_targets[li], leg)
        self.d.qvel[6:] = 0
        mujoco.mj_forward(self.m, self.d)

    def pd_step(self, q_des: np.ndarray):
        """一步 PD（12 目标角，规范序）→ mujoco step。"""
        q = self.d.qpos[self.qadr]
        qd = self.d.qvel[self.vadr]
        tau = np.clip(self.kp * (q_des - q) - self.kd * qd, -self.tlim, self.tlim)
        self.d.ctrl[self.perm] = tau
        mujoco.mj_step(self.m, self.d)

    def state(self, device="cpu", dtype=torch.float32) -> FloatingBaseState:
        """读出与孪生同构的状态：p=全身 COM(世界)、q=基座四元数(wxyz)、
        v=COM 速度(世界)、w=基座角速度(体系)。"""
        mujoco.mj_subtreeVel(self.m, self.d)
        p = self.d.subtree_com[1].copy()
        v = self.d.subtree_linvel[1].copy()
        quat = self.d.qpos[3:7].copy()
        w = self.d.qvel[3:6].copy()                          # 体系（mujoco 约定）
        t = lambda x: torch.tensor(x, device=device, dtype=dtype).unsqueeze(0)
        return FloatingBaseState(t(p), t(quat), t(v), t(w))

    def foot_targets_to_qdes(self, p_b_com: np.ndarray) -> np.ndarray:
        """孪生的 COM 体系足端目标 (4,3) → 躯干系 → IK → 12 关节目标角。"""
        q_des = np.zeros(12)
        for li, leg in enumerate(LEGS):
            p_trunk = p_b_com[li] + self.c_body
            q_des[3 * li:3 * li + 3] = leg_ik(p_trunk, leg)
        return q_des


class MJGo2Full:
    """**匹配参数**外部 oracle：由我们自己的 URDF 参数构建全身 MJCF（go2_mjcf_full）+
    位置伺服。与 MJGo2(menagerie) 同接口，供 E3D-5 双 oracle 鲁棒性交叉验证——
    它**隔离 SRBD 结构简化**（质量/惯量/接触半径/μ/dt 全与孪生匹配，唯一差距=单刚体
    假设 vs 全身 18-DoF + 真接触 + 位置伺服滞后）。menagerie 版则叠加了参数失配。"""

    def __init__(self, kp: float = 100.0, kv: float = 2.0, c_body=None):
        import sys as _sys
        _sys.path.insert(0, str(_HERE.parent / "models"))
        from go2_mjcf_full import build
        from go2_leg_ik import LegKinematics
        self.m = build(kp=kp, kv=kv, timestep=1e-3)
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.ik = LegKinematics()
        self.c_body = np.zeros(3) if c_body is None else np.asarray(c_body)
        order = [f"{leg}_{j}" for leg in LEGS for j in ("hip", "thigh", "calf")]
        self.perm = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                              for n in order])
        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")
                for n in order]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jids])
        self._q_last = np.tile([0.0, 0.8, -1.6], (4, 1))

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def set_base(self, pos, quat_wxyz):
        self.d.qpos[0:3] = pos
        self.d.qpos[3:7] = quat_wxyz
        self.d.qvel[:6] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_joints_ik(self, p_trunk_targets):
        q = self.ik.ik(np.asarray(p_trunk_targets), q0=self._q_last)
        self._q_last = q
        for li in range(4):
            self.d.qpos[self.qadr[3 * li:3 * li + 3]] = q[li]
        self.d.qvel[6:] = 0
        mujoco.mj_forward(self.m, self.d)

    def foot_targets_to_qdes(self, p_b_com):
        q = self.ik.ik(np.asarray(p_b_com) + self.c_body, q0=self._q_last)
        self._q_last = q
        return q.reshape(-1)

    def pd_step(self, q_des):
        self.d.ctrl[self.perm] = q_des                       # 位置伺服（ctrl=目标角）
        mujoco.mj_step(self.m, self.d)

    def state(self, device="cpu", dtype=torch.float32) -> FloatingBaseState:
        mujoco.mj_subtreeVel(self.m, self.d)
        p = self.d.subtree_com[1].copy()
        v = self.d.subtree_linvel[1].copy()
        quat = self.d.qpos[3:7].copy()
        w = self.d.qvel[3:6].copy()
        t = lambda x: torch.tensor(x, device=device, dtype=dtype).unsqueeze(0)
        return FloatingBaseState(t(p), t(quat), t(v), t(w))


if __name__ == "__main__":
    # 往返验证：随机关节角 → mujoco FK 足端位置 → IK → 应回到同一关节角
    mj = MJGo2()
    mj.reset()
    rng = np.random.default_rng(0)
    err_q, err_p = 0.0, 0.0
    foot_geoms = [mujoco.mj_name2id(mj.m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}") for leg in LEGS]
    # menagerie 足端 geom 名为 FL/FR/RL/RR
    for _ in range(50):
        q = np.zeros(12)
        for li in range(4):
            q[3 * li + 0] = rng.uniform(-0.6, 0.6)
            q[3 * li + 1] = rng.uniform(0.2, 1.4)
            q[3 * li + 2] = rng.uniform(-2.4, -0.9)
        mj.d.qpos[:3] = [0, 0, 1.0]
        mj.d.qpos[3:7] = [1, 0, 0, 0]
        mj.d.qpos[mj.qadr] = q
        mujoco.mj_forward(mj.m, mj.d)
        base = mj.d.qpos[:3].copy()
        for li, leg in enumerate(LEGS):
            gp = mj.d.geom_xpos[foot_geoms[li]] - base       # 躯干系（基座姿态单位阵）
            qik = leg_ik(gp, leg)
            err_q = max(err_q, np.abs(qik - q[3 * li:3 * li + 3]).max())
    print(f"IK 往返验证（50 随机姿态×4 腿）：最大关节角误差 = {err_q:.2e} rad "
          f"{'✅' if err_q < 1e-6 else '❌ 排查符号/分支'}")
