"""Emit a traceable Go2 parameter summary (JSON + Markdown) from the official URDF.

Every number here is read from the vendored official URDF; provenance (repo commit)
is copied from assets/go2_description/PROVENANCE.md. Outputs:
    ../parameters/go2_model_summary.json
    ../parameters/go2_model_summary.md
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from go2_urdf import load_go2, LEGS, JOINTS_PER_LEG, DEFAULT_URDF  # noqa: E402

HERE = Path(__file__).resolve().parent
PARAM_DIR = HERE.parent / "parameters"
PROV = HERE.parent / "assets" / "go2_description" / "PROVENANCE.md"

# Typical Go2 standing posture (ASSUMPTION -- legged_gym-style nominal, to be refined
# once we fit a real standing height). hip abduction 0, thigh flexion ~0.9, knee ~-1.8.
NOMINAL_STANDING_Q = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}


def _commit() -> str:
    if PROV.exists():
        m = re.search(r"Pinned commit:\s*([0-9a-f]{7,40})", PROV.read_text())
        if m:
            return m.group(1)
    return "unknown"


def zero_config_foot_offsets(m) -> dict:
    """Foot position relative to base at q=0 (legs straight down).
    Leg-chain joint origins all have rpy=0, so this is a pure translation sum."""
    out = {}
    for leg in LEGS:
        p = np.zeros(3)
        for j in m.chain_to(f"{leg}_foot"):
            p = p + j.origin_xyz  # all rpy=0 along the leg chain
        out[leg] = np.round(p, 5).tolist()
    return out


def build_summary(m) -> dict:
    leg_chain_geometry = {
        "hip_mount_xyz": {leg: m.joints[f"{leg}_hip_joint"].origin_xyz.round(5).tolist() for leg in LEGS},
        "hip_to_thigh_xyz": m.joints["FL_thigh_joint"].origin_xyz.round(5).tolist(),
        "thigh_length_m": float(-m.joints["FL_calf_joint"].origin_xyz[2]),
        "calf_length_m": float(-m.joints["FL_foot_joint"].origin_xyz[2]),
    }

    links_out = {}
    for name, l in m.links.items():
        entry = {"has_inertial": l.inertial is not None}
        if l.inertial:
            entry.update(mass=l.inertial.mass,
                         com=l.inertial.com.round(6).tolist(),
                         inertia=l.inertial.inertia_dict)
        if l.visual and l.visual.mesh:
            entry["visual_mesh"] = l.visual.mesh
            entry["visual_origin_rpy"] = l.visual.origin_rpy.round(5).tolist()
        links_out[name] = entry

    joints_out = {}
    for name, j in m.joints.items():
        joints_out[name] = {
            "type": j.jtype, "parent": j.parent, "child": j.child,
            "origin_xyz": j.origin_xyz.round(6).tolist(),
            "origin_rpy": j.origin_rpy.round(6).tolist(),
            "axis": j.axis.round(3).tolist(),
            "limit": j.limit,
        }

    return {
        "provenance": {
            "source_repo": "https://github.com/Unitree-Go2-Robot/go2_description",
            "branch": "humble", "commit": _commit(),
            "urdf": str(DEFAULT_URDF.relative_to(HERE.parent.parent)),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "robot_name": m.name,
        "base_link": m.base_link,
        "n_links": len(m.links),
        "n_joints": len(m.joints),
        "n_actuated_joints": len(m.actuated_joints),
        "total_mass_kg": round(m.total_mass, 4),
        "canonical_leg_order": list(LEGS),
        "canonical_joint_order": list(JOINTS_PER_LEG),
        "sdk_motor_order_caveat": "Unitree low-level SDK indexes motors FR,FL,RR,RL (differs from URDF FL,FR,RL,RR).",
        "leg_chain_geometry": leg_chain_geometry,
        "zero_config_foot_offset_from_base": zero_config_foot_offsets(m),
        "nominal_standing_q_assumed": NOMINAL_STANDING_Q,
        "foot_links": m.foot_links,
        "frame_convention": "URDF/ROS body frame: +X forward, +Y left, +Z up; gravity -Z.",
        "links": links_out,
        "joints": joints_out,
    }


def write_markdown(s: dict, path: Path) -> None:
    L = []
    p = s["provenance"]
    L.append(f"# Unitree Go2 model summary\n")
    L.append(f"> Source: {p['source_repo']} @ `{p['branch']}` commit `{p['commit']}` · generated {p['generated_utc']}")
    L.append(f"> URDF: `{p['urdf']}` · frame: {s['frame_convention']}\n")
    L.append(f"- **Robot**: `{s['robot_name']}` · base link `{s['base_link']}`")
    L.append(f"- **Links**: {s['n_links']} · **Joints**: {s['n_joints']} · **Actuated**: {s['n_actuated_joints']}")
    L.append(f"- **Total mass**: {s['total_mass_kg']} kg")
    g = s["leg_chain_geometry"]
    L.append(f"- **Thigh length**: {g['thigh_length_m']} m · **Calf length**: {g['calf_length_m']} m")
    L.append(f"- **Canonical order**: legs {s['canonical_leg_order']} × joints {s['canonical_joint_order']}")
    L.append(f"- ⚠ {s['sdk_motor_order_caveat']}\n")

    L.append("## Actuated joints (12)\n")
    L.append("| joint | axis | lower | upper | effort (N·m) | velocity (rad/s) |")
    L.append("|---|---|---|---|---|---|")
    for leg in s["canonical_leg_order"]:
        for jn in s["canonical_joint_order"]:
            j = s["joints"][f"{leg}_{jn}_joint"]
            lim = j["limit"]
            L.append(f"| {leg}_{jn} | {j['axis']} | {lim['lower']:+.4f} | {lim['upper']:+.4f} "
                     f"| {lim['effort']:.2f} | {lim['velocity']:.2f} |")

    L.append("\n## Link inertials (mass > 0)\n")
    L.append("| link | mass (kg) | com (m) | ixx | iyy | izz |")
    L.append("|---|---|---|---|---|---|")
    for name, e in s["links"].items():
        if not e.get("has_inertial") or e.get("mass", 0) <= 0:
            continue
        I = e["inertia"]
        L.append(f"| {name} | {e['mass']:.4f} | {e['com']} | {I['ixx']:.5g} | {I['iyy']:.5g} | {I['izz']:.5g} |")

    L.append("\n## Hip mount points & zero-config foot offsets (from base, m)\n")
    L.append("| leg | hip mount xyz | foot offset @ q=0 |")
    L.append("|---|---|---|")
    for leg in s["canonical_leg_order"]:
        L.append(f"| {leg} | {g['hip_mount_xyz'][leg]} | {s['zero_config_foot_offset_from_base'][leg]} |")
    L.append(f"\n_Nominal standing q (assumed, typical): {s['nominal_standing_q_assumed']}_\n")
    path.write_text("\n".join(L))


def main() -> None:
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    m = load_go2()
    s = build_summary(m)
    (PARAM_DIR / "go2_model_summary.json").write_text(json.dumps(s, indent=2))
    write_markdown(s, PARAM_DIR / "go2_model_summary.md")
    print(f"wrote {PARAM_DIR/'go2_model_summary.json'}")
    print(f"wrote {PARAM_DIR/'go2_model_summary.md'}")
    print(f"  total mass {s['total_mass_kg']} kg, "
          f"thigh/calf {s['leg_chain_geometry']['thigh_length_m']}/{s['leg_chain_geometry']['calf_length_m']} m")
    print(f"  FL foot @ q=0: {s['zero_config_foot_offset_from_base']['FL']}")


if __name__ == "__main__":
    main()
