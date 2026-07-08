"""External no-contact acceleration gate against MuJoCo Menagerie Go2."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.go2.aba import forward_dynamics  # noqa: E402
from diffsim.go2.dynamics import initial_state  # noqa: E402
from diffsim.go2.model import JOINT_ORDER, compile_go2_model  # noqa: E402
from research.quadruped_migration.go2_3d.models.go2_urdf import DEFAULT_URDF  # noqa: E402


MJCF = (
    ROOT
    / "research/quadruped_migration/go2_3d/assets/mujoco_menagerie"
    / "unitree_go2/go2.xml"
)


def _load_matching_urdf(*, with_floor: bool = False) -> mujoco.MjModel:
    """Let MuJoCo independently compile the authoritative inertia/kinematic tree."""

    tree = ET.parse(DEFAULT_URDF)
    root = tree.getroot()
    collision_index = 0
    for link in root.findall("link"):
        for element in link.findall("visual"):
            link.remove(element)
        for element in link.findall("collision"):
            element.set("name", f"collision_{collision_index}")
            collision_index += 1
        if not with_floor:
            for element in link.findall("collision"):
                link.remove(element)
    ET.SubElement(root, "link", name="world")
    floating = ET.SubElement(root, "joint", name="floating_base", type="floating")
    ET.SubElement(floating, "parent", link="world")
    ET.SubElement(floating, "child", link="base_link")
    if with_floor:
        floor = ET.SubElement(root, "link", name="floor")
        collision = ET.SubElement(floor, "collision", name="floor_collision")
        ET.SubElement(collision, "origin", xyz="0 0 -0.05", rpy="0 0 0")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="20 20 0.1")
        floor_joint = ET.SubElement(root, "joint", name="floor_joint", type="fixed")
        ET.SubElement(floor_joint, "parent", link="world")
        ET.SubElement(floor_joint, "child", link="floor")
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        discardvisual="true",
        fusestatic="true",
        balanceinertia="false",
    )
    with tempfile.NamedTemporaryFile(suffix=".urdf") as handle:
        tree.write(handle.name, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(handle.name)


def _quaternion(rng: np.random.Generator) -> np.ndarray:
    rotation = rng.normal(0.0, 0.18, 3)
    angle = np.linalg.norm(rotation)
    if angle < 1e-12:
        return np.array((1.0, 0.0, 0.0, 0.0))
    axis = rotation / angle
    return np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def acceleration_errors(
    samples: int = 64,
    seed: int = 20260707,
    *,
    rotate_base: bool = True,
    moving: bool = True,
    oracle: str = "urdf",
):
    mj_model = _load_matching_urdf() if oracle == "urdf" else mujoco.MjModel.from_xml_path(str(MJCF))
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model.dof_damping[:] = 0.0
    mj_model.dof_armature[:] = 0.0
    mj_model.dof_frictionloss[:] = 0.0
    mj_data = mujoco.MjData(mj_model)
    names = list(JOINT_ORDER)
    joint_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in names
    ]
    qpos_address = np.array([mj_model.jnt_qposadr[index] for index in joint_ids])
    dof_address = np.array([mj_model.jnt_dofadr[index] for index in joint_ids])

    model = compile_go2_model(dtype=torch.float64)
    rng = np.random.default_rng(seed)
    predicted, reference = [], []
    for _ in range(samples):
        state = initial_state(model, 1, base_height=1.5)
        quaternion = _quaternion(rng) if rotate_base else np.array((1.0, 0.0, 0.0, 0.0))
        joint_position = model.default_joint_pos.numpy() + rng.uniform(-0.35, 0.35, 12)
        base_velocity = rng.uniform(-0.5, 0.5, 3) if moving else np.zeros(3)
        base_omega = rng.uniform(-0.6, 0.6, 3) if moving else np.zeros(3)
        joint_velocity = rng.uniform(-1.0, 1.0, 12) if moving else np.zeros(12)
        torque = rng.uniform(-12.0, 12.0, 12)
        state.base_quat[:] = torch.from_numpy(quaternion)
        state.base_vel[:] = torch.from_numpy(base_velocity)
        state.base_omega[:] = torch.from_numpy(base_omega)
        state.joint_pos[:] = torch.from_numpy(joint_position)
        state.joint_vel[:] = torch.from_numpy(joint_velocity)
        acceleration = forward_dynamics(model, state, torch.from_numpy(torque)[None])
        predicted.append(
            torch.cat(
                (acceleration.base_linear_world, acceleration.base_angular_body, acceleration.joint),
                dim=-1,
            )[0].numpy()
        )

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:3] = (0.0, 0.0, 1.5)
        mj_data.qpos[3:7] = quaternion
        mj_data.qpos[qpos_address] = joint_position
        mj_data.qvel[:3] = base_velocity
        mj_data.qvel[3:6] = base_omega
        mj_data.qvel[dof_address] = joint_velocity
        mj_data.qfrc_applied[dof_address] = torque
        mujoco.mj_forward(mj_model, mj_data)
        reference.append(np.r_[mj_data.qacc[:6], mj_data.qacc[dof_address]])

    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    scale = np.linalg.norm(reference, axis=1).clip(1.0)
    relative = np.linalg.norm(predicted - reference, axis=1) / scale
    component_scale = np.maximum(np.abs(reference), 1.0)
    component_relative = np.abs(predicted - reference) / component_scale
    return relative, component_relative, predicted, reference


def main():
    relative, component_relative, predicted, reference = acceleration_errors()
    median = float(np.median(relative))
    percentile95 = float(np.percentile(relative, 95))
    component_sign = float(np.mean(np.sign(predicted) == np.sign(reference)))
    print(f"sample norm relative error: median={100 * median:.3f}% p95={100 * percentile95:.3f}%")
    print(
        f"component relative: median={100 * np.median(component_relative):.3f}% "
        f"p95={100 * np.percentile(component_relative, 95):.3f}% sign={100 * component_sign:.2f}%"
    )
    for label, selection in (("base linear", slice(0, 3)), ("base angular", slice(3, 6)), ("joints", slice(6, 18))):
        scale = np.linalg.norm(reference[:, selection], axis=1).clip(1.0)
        error = np.linalg.norm(predicted[:, selection] - reference[:, selection], axis=1) / scale
        print(f"{label:12s}: median={100 * np.median(error):.3f}% p95={100 * np.percentile(error, 95):.3f}%")
    for label, rotate, moving in (
        ("identity/zero velocity", False, False),
        ("rotated/zero velocity", True, False),
        ("identity/moving", False, True),
    ):
        case, _, _, _ = acceleration_errors(24, rotate_base=rotate, moving=moving)
        print(f"{label:22s}: median={100 * np.median(case):.3f}% p95={100 * np.percentile(case, 95):.3f}%")
    mismatch, _, _, _ = acceleration_errors(24, oracle="menagerie")
    print(
        "Menagerie parameter-mismatch audit: "
        f"median={100 * np.median(mismatch):.3f}% p95={100 * np.percentile(mismatch, 95):.3f}%"
    )
    assert median <= 0.02
    assert percentile95 <= 0.05
    print("[PASS] MuJoCo random no-contact acceleration gate")


if __name__ == "__main__":
    main()
