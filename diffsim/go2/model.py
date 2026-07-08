"""Compile the official Go2 URDF into a fixed-topology GPU dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from research.quadruped_migration.go2_3d.models.go2_urdf import (
    DEFAULT_URDF,
    JOINTS_PER_LEG,
    LEGS,
    load_go2,
)

JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in JOINTS_PER_LEG)
SDK_TO_CANONICAL = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

COLLISION_SPHERE = 0
COLLISION_CAPSULE = 1
COLLISION_BOX = 2


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = xyz
    return out


def _parse_vec(value: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array(default if value is None else [float(v) for v in value.split()], dtype=float)


@dataclass(frozen=True)
class CollisionGeometry:
    owner: torch.Tensor
    kind: torch.Tensor
    local_pos: torch.Tensor
    local_rot: torch.Tensor
    size: torch.Tensor
    link_names: tuple[str, ...]
    sample_owner: torch.Tensor
    sample_pos: torch.Tensor
    sample_radius: torch.Tensor
    sample_is_foot: torch.Tensor
    sample_collision: torch.Tensor
    bounding_radius: torch.Tensor
    self_sample: torch.Tensor
    self_collision: torch.Tensor


@dataclass(frozen=True)
class Go2Model:
    parent: torch.Tensor
    parent_indices: tuple[int, ...]
    joint_axis: torch.Tensor
    origin_rotation: torch.Tensor
    origin_translation: torch.Tensor
    spatial_inertia: torch.Tensor
    body_mass: torch.Tensor
    body_com: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    joint_effort: torch.Tensor
    joint_velocity: torch.Tensor
    default_joint_pos: torch.Tensor
    collisions: CollisionGeometry
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_bodies(self) -> int:
        return len(self.body_names)


def compile_go2_model(
    urdf_path: Path | str = DEFAULT_URDF,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Go2Model:
    """Collapse fixed joints while preserving their inertia and collision geometry."""

    urdf_path = Path(urdf_path)
    parsed = load_go2(urdf_path)
    actuated = {name: i + 1 for i, name in enumerate(JOINT_ORDER)}
    body_names = (parsed.base_link,) + tuple(parsed.joints[name].child for name in JOINT_ORDER)
    children: dict[str, list] = {}
    child_joint = {}
    for joint in parsed.joints.values():
        children.setdefault(joint.parent, []).append(joint)
        child_joint[joint.child] = joint

    absolute = {parsed.base_link: np.eye(4)}
    queue = [parsed.base_link]
    while queue:
        parent_link = queue.pop(0)
        for joint in children.get(parent_link, []):
            absolute[joint.child] = absolute[parent_link] @ _transform(joint.origin_xyz, joint.origin_rpy)
            queue.append(joint.child)

    def owner_for(link_name: str) -> int:
        current = link_name
        while current != parsed.base_link:
            joint = child_joint[current]
            if joint.name in actuated:
                return actuated[joint.name]
            current = joint.parent
        return 0

    spatial = np.zeros((13, 6, 6), dtype=float)
    mass_sum = np.zeros(13, dtype=float)
    first_moment = np.zeros((13, 3), dtype=float)
    for link_name, link in parsed.links.items():
        if link.inertial is None or link.inertial.mass == 0:
            continue
        owner = owner_for(link_name)
        t_owner_link = np.linalg.inv(absolute[body_names[owner]]) @ absolute[link_name]
        rotation = t_owner_link[:3, :3]
        com = t_owner_link[:3, 3] + rotation @ link.inertial.com
        inertia_com = rotation @ link.inertial.inertia @ rotation.T
        m = link.inertial.mass
        cx = np.array(((0, -com[2], com[1]), (com[2], 0, -com[0]), (-com[1], com[0], 0)))
        item = np.block(
            [[inertia_com - m * cx @ cx, m * cx], [-m * cx, m * np.eye(3)]]
        )
        spatial[owner] += item
        mass_sum[owner] += m
        first_moment[owner] += m * com
    body_com = first_moment / np.maximum(mass_sum[:, None], 1e-12)

    parents, axes, rotations, translations = [], [], [], []
    lower, upper, effort, velocity = [], [], [], []
    for name in JOINT_ORDER:
        joint = parsed.joints[name]
        body_id = actuated[name]
        parent_id = owner_for(joint.parent)
        relative = np.linalg.inv(absolute[body_names[parent_id]]) @ absolute[joint.child]
        parents.append(parent_id)
        axes.append(joint.axis)
        rotations.append(relative[:3, :3])
        translations.append(relative[:3, 3])
        lower.append(joint.limit["lower"])
        upper.append(joint.limit["upper"])
        effort.append(joint.limit["effort"])
        velocity.append(joint.limit["velocity"])
        if body_names[body_id] != joint.child:
            raise AssertionError("dynamic body ordering mismatch")

    root = ET.parse(urdf_path).getroot()
    collision_owner, collision_kind = [], []
    collision_pos, collision_rot, collision_size, collision_links = [], [], [], []
    for link_el in root.findall("link"):
        link_name = link_el.get("name")
        owner = owner_for(link_name)
        t_owner_link = np.linalg.inv(absolute[body_names[owner]]) @ absolute[link_name]
        for collision in link_el.findall("collision"):
            origin = collision.find("origin")
            xyz = _parse_vec(None if origin is None else origin.get("xyz"))
            rpy = _parse_vec(None if origin is None else origin.get("rpy"))
            t_owner_collision = t_owner_link @ _transform(xyz, rpy)
            geometry = collision.find("geometry")
            sphere = geometry.find("sphere")
            cylinder = geometry.find("cylinder")
            box = geometry.find("box")
            if sphere is not None:
                kind = COLLISION_SPHERE
                size = (float(sphere.get("radius")), 0.0, 0.0)
            elif cylinder is not None:
                kind = COLLISION_CAPSULE
                size = (float(cylinder.get("radius")), 0.5 * float(cylinder.get("length")), 0.0)
            elif box is not None:
                kind = COLLISION_BOX
                size = tuple(0.5 * _parse_vec(box.get("size")))
            else:
                continue
            collision_owner.append(owner)
            collision_kind.append(kind)
            collision_pos.append(t_owner_collision[:3, 3])
            collision_rot.append(t_owner_collision[:3, :3])
            collision_size.append(size)
            collision_links.append(link_name)

    def tensor(value, *, long=False):
        return torch.as_tensor(value, device=device, dtype=torch.long if long else dtype)

    collisions = CollisionGeometry(
        owner=tensor(collision_owner, long=True),
        kind=tensor(collision_kind, long=True),
        local_pos=tensor(np.asarray(collision_pos)),
        local_rot=tensor(np.asarray(collision_rot)),
        size=tensor(np.asarray(collision_size)),
        link_names=tuple(collision_links),
        sample_owner=torch.empty(0, device=device, dtype=torch.long),
        sample_pos=torch.empty(0, 3, device=device, dtype=dtype),
        sample_radius=torch.empty(0, device=device, dtype=dtype),
        sample_is_foot=torch.empty(0, device=device, dtype=torch.bool),
        sample_collision=torch.empty(0, device=device, dtype=torch.long),
        bounding_radius=torch.empty(0, device=device, dtype=dtype),
        self_sample=torch.empty(0, device=device, dtype=torch.long),
        self_collision=torch.empty(0, device=device, dtype=torch.long),
    )
    sample_owner, sample_pos, sample_radius, sample_is_foot, sample_collision = [], [], [], [], []
    bounding_radius = []
    for collision_index, (owner, kind, pos, rotation, size, link_name) in enumerate(
        zip(collision_owner, collision_kind, collision_pos, collision_rot, collision_size, collision_links)
    ):
        size_arr = np.asarray(size)
        if kind == COLLISION_SPHERE:
            offsets = [np.zeros(3)]
            radii = [size_arr[0]]
            bound = size_arr[0]
        elif kind == COLLISION_CAPSULE:
            offsets = [np.array((0.0, 0.0, z)) for z in (-size_arr[1], 0.0, size_arr[1])]
            radii = [size_arr[0]] * len(offsets)
            bound = size_arr[0] + size_arr[1]
        else:
            signs = [np.array((x, y, z)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            offsets = [sign * size_arr for sign in signs]
            offsets += [np.array((axis == 0, axis == 1, axis == 2), dtype=float) * sign * size_arr
                        for axis in range(3) for sign in (-1, 1)]
            radii = [0.0] * len(offsets)
            bound = float(np.linalg.norm(size_arr))
        for offset, radius in zip(offsets, radii):
            sample_owner.append(owner)
            sample_pos.append(np.asarray(pos) + np.asarray(rotation) @ offset)
            sample_radius.append(radius)
            sample_is_foot.append(link_name.endswith("_foot"))
            sample_collision.append(collision_index)
        bounding_radius.append(bound)
    # Directed sample-to-primitive tests cover every non-adjacent body pair.
    # Testing both directions catches box corners against capsules without a
    # coarse whole-link bounding sphere.
    pair_a, pair_b = [], []
    samples_by_collision = [[] for _ in collision_owner]
    for sample_index, collision_index in enumerate(sample_collision):
        samples_by_collision[collision_index].append(sample_index)
    for first in range(len(collision_owner)):
        for second in range(first + 1, len(collision_owner)):
            owner_a, owner_b = collision_owner[first], collision_owner[second]
            if owner_a == owner_b:
                continue
            adjacent = (
                (owner_a > 0 and parents[owner_a - 1] == owner_b)
                or (owner_b > 0 and parents[owner_b - 1] == owner_a)
            )
            if adjacent:
                continue
            pair_a.extend(samples_by_collision[first])
            pair_b.extend([second] * len(samples_by_collision[first]))
            pair_a.extend(samples_by_collision[second])
            pair_b.extend([first] * len(samples_by_collision[second]))
    collisions = CollisionGeometry(
        owner=collisions.owner,
        kind=collisions.kind,
        local_pos=collisions.local_pos,
        local_rot=collisions.local_rot,
        size=collisions.size,
        link_names=collisions.link_names,
        sample_owner=tensor(sample_owner, long=True),
        sample_pos=tensor(np.asarray(sample_pos)),
        sample_radius=tensor(sample_radius),
        sample_is_foot=torch.as_tensor(sample_is_foot, device=device, dtype=torch.bool),
        sample_collision=tensor(sample_collision, long=True),
        bounding_radius=tensor(bounding_radius),
        self_sample=tensor(pair_a, long=True),
        self_collision=tensor(pair_b, long=True),
    )
    default = tensor(np.tile((0.0, 0.9, -1.8), 4))
    return Go2Model(
        parent=tensor(parents, long=True),
        parent_indices=tuple(parents),
        joint_axis=tensor(np.asarray(axes)),
        origin_rotation=tensor(np.asarray(rotations)),
        origin_translation=tensor(np.asarray(translations)),
        spatial_inertia=tensor(spatial),
        body_mass=tensor(mass_sum),
        body_com=tensor(body_com),
        joint_lower=tensor(lower),
        joint_upper=tensor(upper),
        joint_effort=tensor(effort),
        joint_velocity=tensor(velocity),
        default_joint_pos=default,
        collisions=collisions,
        body_names=body_names,
        joint_names=JOINT_ORDER,
    )
