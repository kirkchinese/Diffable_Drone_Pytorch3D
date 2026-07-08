"""Smooth full-body contact against analytic scenes and selected self pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .aba import Go2Kinematics, forward_kinematics, spatial_velocities
from .model import COLLISION_CAPSULE, COLLISION_SPHERE, Go2Model
from .scene import Go2Scene
from .types import Go2EnvConfig, Go2State


@dataclass
class ContactOutput:
    wrench_body: torch.Tensor
    gap: torch.Tensor
    penetration: torch.Tensor
    probability: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    normal: torch.Tensor
    sample_world: torch.Tensor
    sample_velocity_world: torch.Tensor
    sample_is_foot: torch.Tensor
    source: torch.Tensor
    self_gap: torch.Tensor


def _smooth_relu(value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    return 0.5 * (value + torch.sqrt(value.square() + epsilon * epsilon))


def _scatter_wrench(
    batch: int,
    n_bodies: int,
    owner: torch.Tensor,
    local_point: torch.Tensor,
    rotation: torch.Tensor,
    force_world: torch.Tensor,
) -> torch.Tensor:
    force_body = torch.einsum("bsji,bsj->bsi", rotation, force_world)
    moment_body = torch.cross(local_point, force_body, dim=-1)
    spatial = torch.cat((moment_body, force_body), dim=-1)
    wrench = spatial.new_zeros(batch, n_bodies, 6)
    index = owner.view(1, -1, 1).expand(batch, -1, 6)
    return wrench.scatter_add(1, index, spatial)


def _collision_centers(model: Go2Model, kin: Go2Kinematics, velocity: torch.Tensor):
    geometry = model.collisions
    owner = geometry.owner
    rotation = kin.body_rotation[:, owner]
    local = geometry.local_pos[None, :, :].expand(kin.body_rotation.shape[0], -1, -1)
    position = kin.body_position[:, owner] + torch.einsum("bsij,bsj->bsi", rotation, local)
    body_velocity = velocity[:, owner]
    point_velocity_body = body_velocity[..., 3:] + torch.cross(
        body_velocity[..., :3], local, dim=-1
    )
    point_velocity = torch.einsum("bsij,bsj->bsi", rotation, point_velocity_body)
    return position, point_velocity, rotation, local


def _primitive_sdf(
    point_world: torch.Tensor,
    center_world: torch.Tensor,
    rotation: torch.Tensor,
    kind: torch.Tensor,
    size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = torch.einsum("bpji,bpj->bpi", rotation, point_world - center_world)
    norm = torch.linalg.norm(relative, dim=-1).clamp_min(1e-9)
    sphere_distance = norm - size[..., 0]
    sphere_normal = relative / norm[..., None]

    capsule_z = relative[..., 2].clamp(-size[..., 1], size[..., 1])
    capsule_nearest = torch.stack(
        (torch.zeros_like(capsule_z), torch.zeros_like(capsule_z), capsule_z), -1
    )
    capsule_delta = relative - capsule_nearest
    capsule_norm = torch.linalg.norm(capsule_delta, dim=-1).clamp_min(1e-9)
    capsule_distance = capsule_norm - size[..., 0]
    capsule_normal = capsule_delta / capsule_norm[..., None]

    box_q = relative.abs() - size
    box_out = box_q.clamp_min(0.0)
    box_out_norm = torch.linalg.norm(box_out, dim=-1)
    box_distance = box_out_norm + box_q.amax(-1).clamp_max(0.0)
    outside_normal = relative.sign() * box_out / box_out_norm[..., None].clamp_min(1e-9)
    inside_axis = torch.nn.functional.one_hot(box_q.argmax(-1), 3).to(relative)
    inside_normal = relative.sign() * inside_axis
    box_normal = torch.where((box_out_norm > 1e-9)[..., None], outside_normal, inside_normal)

    distance = torch.where(
        kind == COLLISION_SPHERE,
        sphere_distance,
        torch.where(kind == COLLISION_CAPSULE, capsule_distance, box_distance),
    )
    normal_local = torch.where(
        (kind == COLLISION_SPHERE)[..., None],
        sphere_normal,
        torch.where((kind == COLLISION_CAPSULE)[..., None], capsule_normal, box_normal),
    )
    normal_world = torch.einsum("bpij,bpj->bpi", rotation, normal_local)
    return distance, normal_world


def contact_wrenches(
    model: Go2Model,
    state: Go2State,
    scene: Go2Scene,
    config: Go2EnvConfig,
    *,
    kinematics: Go2Kinematics | None = None,
) -> ContactOutput:
    """Compute world and self-contact forces as body-frame spatial wrenches."""

    kin = forward_kinematics(model, state) if kinematics is None else kinematics
    velocity, _, _ = spatial_velocities(model, state, kin)
    geometry = model.collisions
    owner = geometry.sample_owner
    rotation = kin.body_rotation[:, owner]
    local_center = geometry.sample_pos[None, :, :].expand(state.batch_size, -1, -1)
    center_world = kin.body_position[:, owner] + torch.einsum(
        "bsij,bsj->bsi", rotation, local_center
    )
    body_velocity = velocity[:, owner]
    center_velocity_body = body_velocity[..., 3:] + torch.cross(
        body_velocity[..., :3], local_center, dim=-1
    )
    center_velocity = torch.einsum("bsij,bsj->bsi", rotation, center_velocity_body)

    distance, normal, source = scene.signed_distance(center_world)
    radius = geometry.sample_radius[None, :]
    gap = distance - radius
    contact_point = center_world - normal * radius[..., None]
    local_point = local_center - torch.einsum(
        "bsji,bsj->bsi", rotation, normal * radius[..., None]
    )
    normal_velocity = (center_velocity * normal).sum(-1)
    epsilon = config.contact_smoothing
    penetration = epsilon * torch.nn.functional.softplus(-gap / epsilon)
    probability = torch.sigmoid(-gap / epsilon)
    approach = _smooth_relu(-normal_velocity)
    normal_force = config.contact_stiffness * penetration + config.contact_damping * probability * approach
    tangent_velocity = center_velocity - normal_velocity[..., None] * normal
    tangent_speed = torch.linalg.norm(tangent_velocity, dim=-1, keepdim=True)
    tangent_direction = tangent_velocity / tangent_speed.clamp_min(1e-9)
    friction = scene.friction[:, None, :]
    tangent_magnitude = friction * normal_force[..., None] * torch.tanh(
        tangent_speed / config.friction_velocity
    )
    tangential_force = -tangent_magnitude * tangent_direction
    force_world = normal_force[..., None] * normal + tangential_force
    wrench = _scatter_wrench(
        state.batch_size, model.n_bodies, owner, local_point, rotation, force_world
    )

    center, _, primitive_rotation, _ = _collision_centers(
        model, kin, velocity
    )
    sample_index, target_collision = geometry.self_sample, geometry.self_collision
    if sample_index.numel():
        source_owner = geometry.sample_owner[sample_index]
        target_owner = geometry.owner[target_collision]
        source_rotation = kin.body_rotation[:, source_owner]
        target_rotation = primitive_rotation[:, target_collision]
        source_center = center_world[:, sample_index]
        target_center = center[:, target_collision]
        distance_self, normal_self = _primitive_sdf(
            source_center,
            target_center,
            target_rotation,
            geometry.kind[target_collision][None],
            geometry.size[target_collision][None],
        )
        source_radius = geometry.sample_radius[sample_index][None]
        self_gap = distance_self - source_radius
        source_local = geometry.sample_pos[sample_index][None].expand(state.batch_size, -1, -1)
        source_point_local = source_local - torch.einsum(
            "bpji,bpj->bpi", source_rotation, normal_self * source_radius[..., None]
        )
        target_surface_world = source_center - normal_self * distance_self[..., None]
        target_point_local = torch.einsum(
            "bpji,bpj->bpi",
            target_rotation,
            target_surface_world - kin.body_position[:, target_owner],
        )
        source_spatial_velocity = velocity[:, source_owner]
        target_spatial_velocity = velocity[:, target_owner]
        source_velocity_body = source_spatial_velocity[..., 3:] + torch.cross(
            source_spatial_velocity[..., :3], source_point_local, dim=-1
        )
        target_velocity_body = target_spatial_velocity[..., 3:] + torch.cross(
            target_spatial_velocity[..., :3], target_point_local, dim=-1
        )
        source_velocity_world = torch.einsum(
            "bpij,bpj->bpi", source_rotation, source_velocity_body
        )
        target_velocity_world = torch.einsum(
            "bpij,bpj->bpi", target_rotation, target_velocity_body
        )
        relative_velocity = source_velocity_world - target_velocity_world
        normal_velocity_self = (relative_velocity * normal_self).sum(-1)
        closing = _smooth_relu(-normal_velocity_self)
        self_pen = epsilon * torch.nn.functional.softplus(-self_gap / epsilon)
        self_gate = torch.sigmoid(-self_gap / epsilon)
        magnitude = 0.5 * (
            config.contact_stiffness * self_pen
            + config.contact_damping * self_gate * closing
        )
        tangent_velocity_self = relative_velocity - normal_velocity_self[..., None] * normal_self
        tangent_speed_self = torch.linalg.norm(tangent_velocity_self, dim=-1, keepdim=True)
        tangent_direction_self = tangent_velocity_self / tangent_speed_self.clamp_min(1e-9)
        friction_self = config.friction * magnitude[..., None] * torch.tanh(
            tangent_speed_self / config.friction_velocity
        )
        force_a = magnitude[..., None] * normal_self - friction_self * tangent_direction_self
        force_b = -force_a
        wrench = wrench + _scatter_wrench(
            state.batch_size,
            model.n_bodies,
            source_owner,
            source_point_local,
            source_rotation,
            force_a,
        )
        wrench = wrench + _scatter_wrench(
            state.batch_size,
            model.n_bodies,
            target_owner,
            target_point_local,
            target_rotation,
            force_b,
        )
    else:
        self_gap = gap.new_empty(state.batch_size, 0)
    return ContactOutput(
        wrench_body=wrench,
        gap=gap,
        penetration=penetration,
        probability=probability,
        normal_force=normal_force,
        tangential_force=tangential_force,
        normal=normal,
        sample_world=contact_point,
        sample_velocity_world=center_velocity,
        sample_is_foot=geometry.sample_is_foot,
        source=source,
        self_gap=self_gap,
    )
