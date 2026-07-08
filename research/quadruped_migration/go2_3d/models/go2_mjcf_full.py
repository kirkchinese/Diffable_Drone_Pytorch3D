"""官方 URDF → 全身 MuJoCo MJCF（E3D-5 外部验证用的"真实系统"）。

与（已废弃平行线的）dynamics-only oracle 不同，这是**带接触、可走路、可渲染**的全身
Go2：base freejoint + 12 hinge（惯量/限位/力矩上限逐项取自 URDF），足端球碰撞
（r=0.022，μ=0.8 与孪生 ContactConfig 同量级）+ 地面 + 躯干碰撞盒（摔倒可见），
关节位置伺服（kp/kv 可调，Go2 常用量级）。视觉用几何原语（胶囊/盒），不依赖网格
——E3D-0 已证 .dae 链路烦人，看步态用原语足够。

诚实定位：MuJoCo 全身模型对 SRBD 孪生而言是**真实未知失配**（质量分布、腿惯量、
关节伺服、硬接触 LCP 全都不同）——这正是 E3D-5 要的外部裁判。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_urdf import LEGS, JOINTS_PER_LEG, load_go2


def _inertial(link, ind):
    i = link.inertial
    I = i.inertia
    fi = (f"{I[0,0]:.8g} {I[1,1]:.8g} {I[2,2]:.8g} "
          f"{I[0,1]:.8g} {I[0,2]:.8g} {I[1,2]:.8g}")
    return (f'{ind}<inertial pos="{i.com[0]:.8g} {i.com[1]:.8g} {i.com[2]:.8g}" '
            f'mass="{i.mass:.8g}" fullinertia="{fi}"/>')


def build_xml(kp: float = 60.0, kv: float = 2.0, timestep: float = 1e-3,
              mu: float = 0.8) -> str:
    m = load_go2()
    L, J = m.links, m.joints
    s = [f'<mujoco model="go2_full">',
         '  <compiler angle="radian" autolimits="true"/>',
         f'  <option gravity="0 0 -9.81" timestep="{timestep}"/>',
         '  <default>',
         f'    <geom friction="{mu} 0.005 0.0001" contype="0" conaffinity="0"/>',
         '    <joint damping="0.1"/>',
         '  </default>',
         '  <worldbody>',
         '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>',
         '    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.85 0.85 0.85 1"'
         '      contype="1" conaffinity="1"/>',
         '    <body name="base" pos="0 0 0.45">',
         '      <freejoint name="root"/>',
         _inertial(L["base_link"], "      "),
         '      <geom name="trunk" type="box" size="0.23 0.095 0.055" rgba="0.35 0.35 0.4 1"'
         '        contype="1" conaffinity="1" mass="0"/>',
         '      <site name="imu" pos="0 0 0" size="0.01"/>']
    for leg in LEGS:
        hip = J[f"{leg}_hip_joint"]; thi = J[f"{leg}_thigh_joint"]
        cal = J[f"{leg}_calf_joint"]; foo = J[f"{leg}_foot_joint"]
        lim = lambda j: f'{j.limit["lower"]:.6g} {j.limit["upper"]:.6g}'
        s += [f'      <body name="{leg}_hip" pos="{hip.origin_xyz[0]:.8g} '
              f'{hip.origin_xyz[1]:.8g} {hip.origin_xyz[2]:.8g}">',
              f'        <joint name="{leg}_hip_joint" type="hinge" axis="1 0 0" '
              f'range="{lim(hip)}"/>',
              _inertial(L[f"{leg}_hip"], "        "),
              f'        <geom type="sphere" size="0.045" rgba="0.2 0.4 0.8 1" mass="0"/>',
              f'        <body name="{leg}_thigh" pos="{thi.origin_xyz[0]:.8g} '
              f'{thi.origin_xyz[1]:.8g} {thi.origin_xyz[2]:.8g}">',
              f'          <joint name="{leg}_thigh_joint" type="hinge" axis="0 1 0" '
              f'range="{lim(thi)}"/>',
              _inertial(L[f"{leg}_thigh"], "          "),
              f'          <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.022" '
              f'rgba="0.9 0.55 0.15 1" mass="0"/>',
              f'          <body name="{leg}_calf" pos="{cal.origin_xyz[0]:.8g} '
              f'{cal.origin_xyz[1]:.8g} {cal.origin_xyz[2]:.8g}">',
              f'            <joint name="{leg}_calf_joint" type="hinge" axis="0 1 0" '
              f'range="{lim(cal)}"/>',
              _inertial(L[f"{leg}_calf"], "            "),
              f'            <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.016" '
              f'rgba="0.25 0.6 0.3 1" mass="0"/>',
              f'            <body name="{leg}_foot" pos="{foo.origin_xyz[0]:.8g} '
              f'{foo.origin_xyz[1]:.8g} {foo.origin_xyz[2]:.8g}">',
              _inertial(L[f"{leg}_foot"], "              "),
              f'              <geom name="{leg}_foot" type="sphere" size="0.022" '
              f'rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1" mass="0"/>',
              f'              <site name="{leg}_foot" pos="0 0 0" size="0.005"/>',
              '            </body>', '          </body>', '        </body>', '      </body>']
    s += ['    </body>', '  </worldbody>', '  <actuator>']
    for leg in LEGS:
        for jn in JOINTS_PER_LEG:
            j = J[f"{leg}_{jn}_joint"]
            s.append(f'    <position name="{leg}_{jn}" joint="{leg}_{jn}_joint" '
                     f'kp="{kp}" kv="{kv}" ctrlrange="{j.limit["lower"]:.6g} '
                     f'{j.limit["upper"]:.6g}" forcerange="-{j.limit["effort"]:.6g} '
                     f'{j.limit["effort"]:.6g}"/>')
    s += ['  </actuator>', '</mujoco>']
    return "\n".join(s)


def build(kp: float = 60.0, kv: float = 2.0, timestep: float = 1e-3):
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_xml(kp, kv, timestep))
    return model


if __name__ == "__main__":
    import mujoco
    m = build()
    print(f"go2_full: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
          f"mass={m.body_mass.sum():.4f} kg (URDF 15.019, 略去 Head 0.002)")
