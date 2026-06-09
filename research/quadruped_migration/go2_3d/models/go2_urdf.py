"""Lightweight, dependency-free parser for the official Unitree Go2 URDF.

Reads ``go2_description.urdf`` (SolidWorks-exported, flat -- no xacro) and exposes
the kinematic tree, link inertials, joint definitions and visual-mesh references as
plain dataclasses. Pure standard-library XML; no urdfpy/yourdfpy needed.

This is the *authoritative* source of Go2 physics parameters (mass, inertia, joint
limits, link offsets) for the differentiable digital twin. Meshes are referenced only
for rendering; the converted .obj files live in ``../assets/obj``.

Canonical orderings used throughout the go2_3d work
---------------------------------------------------
LEGS  = ("FL", "FR", "RL", "RR")          # URDF declaration order
JOINTS_PER_LEG = ("hip", "thigh", "calf") # abduction, hip-flexion, knee
Actuated joint vector q (12,) is therefore ordered:
    FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf

NOTE (sim alignment caveat): Unitree's low-level SDK indexes motors as
FR, FL, RR, RL -- a *different* order. Any later comparison against Unitree
tooling must remap. We keep the URDF order here and document the difference.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
JOINTS_PER_LEG = ("hip", "thigh", "calf")

_HERE = Path(__file__).resolve().parent
DEFAULT_URDF = _HERE.parent / "assets" / "go2_description" / "urdf" / "go2_description.urdf"
OBJ_DIR = _HERE.parent / "assets" / "obj"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Inertial:
    mass: float
    com: np.ndarray            # (3,) inertial-frame origin (xyz) in the link frame
    inertia: np.ndarray        # (3,3) symmetric tensor about the COM frame

    @property
    def inertia_dict(self) -> dict:
        I = self.inertia
        return dict(ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2],
                    ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2])


@dataclass(frozen=True)
class Visual:
    mesh: Optional[str]        # mesh basename without extension, e.g. "thigh_mirror"
    origin_xyz: np.ndarray     # (3,)
    origin_rpy: np.ndarray     # (3,) roll-pitch-yaw applied to the visual mesh


@dataclass(frozen=True)
class Link:
    name: str
    inertial: Optional[Inertial]
    visual: Optional[Visual]


@dataclass(frozen=True)
class Joint:
    name: str
    jtype: str                 # "revolute" | "fixed" | ...
    parent: str
    child: str
    origin_xyz: np.ndarray     # (3,) translation of child frame in parent frame
    origin_rpy: np.ndarray     # (3,)
    axis: np.ndarray           # (3,) unit axis in the child frame (zeros for fixed)
    limit: Optional[dict]      # {lower, upper, effort, velocity} for revolute


@dataclass
class Go2Model:
    name: str
    links: dict                # name -> Link
    joints: dict               # name -> Joint
    source_urdf: Path

    # ---- derived convenience views ----
    @property
    def actuated_joints(self) -> list:
        """The 12 revolute leg joints in canonical (LEGS x JOINTS_PER_LEG) order."""
        order = [f"{leg}_{j}_joint" for leg in LEGS for j in JOINTS_PER_LEG]
        return [self.joints[n] for n in order]

    @property
    def base_link(self) -> str:
        children = {j.child for j in self.joints.values()}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"expected a single root link, found {roots}")
        return roots[0]

    @property
    def foot_links(self) -> list:
        return [f"{leg}_foot" for leg in LEGS]

    @property
    def total_mass(self) -> float:
        return sum(l.inertial.mass for l in self.links.values() if l.inertial)

    def child_joint(self, parent_link: str) -> list:
        return [j for j in self.joints.values() if j.parent == parent_link]

    def chain_to(self, link_name: str) -> list:
        """Ordered list of joints from base to ``link_name`` (inclusive)."""
        parent_of_joint = {j.child: j for j in self.joints.values()}
        chain = []
        cur = link_name
        while cur in parent_of_joint:
            j = parent_of_joint[cur]
            chain.append(j)
            cur = j.parent
        return list(reversed(chain))


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _vec(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def _parse_origin(el) -> tuple:
    o = None if el is None else el.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    return _vec(o.get("xyz")), _vec(o.get("rpy"))


def _parse_inertial(link_el) -> Optional[Inertial]:
    el = link_el.find("inertial")
    if el is None:
        return None
    mass = float(el.find("mass").get("value"))
    com, _ = _parse_origin(el)
    i = el.find("inertia")
    ixx, iyy, izz = float(i.get("ixx")), float(i.get("iyy")), float(i.get("izz"))
    ixy, ixz, iyz = float(i.get("ixy")), float(i.get("ixz")), float(i.get("iyz"))
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
    return Inertial(mass=mass, com=com, inertia=I)


def _parse_visual(link_el) -> Optional[Visual]:
    el = link_el.find("visual")
    if el is None:
        return None
    xyz, rpy = _parse_origin(el)
    mesh_el = el.find("geometry/mesh")
    mesh = None
    if mesh_el is not None:
        fname = mesh_el.get("filename", "")
        mesh = Path(fname).stem  # "package://.../thigh_mirror.dae" -> "thigh_mirror"
    return Visual(mesh=mesh, origin_xyz=xyz, origin_rpy=rpy)


def _parse_joint(j_el) -> Joint:
    xyz, rpy = _parse_origin(j_el)
    axis_el = j_el.find("axis")
    axis = _vec(None if axis_el is None else axis_el.get("xyz"))
    lim_el = j_el.find("limit")
    limit = None
    if lim_el is not None:
        limit = dict(
            lower=float(lim_el.get("lower", "nan")),
            upper=float(lim_el.get("upper", "nan")),
            effort=float(lim_el.get("effort", "nan")),
            velocity=float(lim_el.get("velocity", "nan")),
        )
    return Joint(
        name=j_el.get("name"),
        jtype=j_el.get("type"),
        parent=j_el.find("parent").get("link"),
        child=j_el.find("child").get("link"),
        origin_xyz=xyz, origin_rpy=rpy, axis=axis, limit=limit,
    )


def load_go2(urdf_path: Path = DEFAULT_URDF) -> Go2Model:
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    links = {}
    for le in root.findall("link"):
        name = le.get("name")
        links[name] = Link(name=name, inertial=_parse_inertial(le), visual=_parse_visual(le))
    joints = {je.get("name"): _parse_joint(je) for je in root.findall("joint")}
    return Go2Model(name=root.get("name"), links=links, joints=joints, source_urdf=urdf_path)


if __name__ == "__main__":
    m = load_go2()
    print(f"robot: {m.name}   root/base link: {m.base_link}")
    print(f"links: {len(m.links)}   joints: {len(m.joints)}   total mass: {m.total_mass:.3f} kg")
    print(f"actuated joints ({len(m.actuated_joints)}):")
    for j in m.actuated_joints:
        lim = j.limit
        print(f"  {j.name:16s} axis={j.axis.astype(int).tolist()} "
              f"lim=[{lim['lower']:+.4f},{lim['upper']:+.4f}] "
              f"eff={lim['effort']:.1f} vel={lim['velocity']:.1f}")
    print("foot links:", m.foot_links)
