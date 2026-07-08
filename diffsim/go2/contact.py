"""Smooth full-body contact against analytic scenes and selected self pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .aba import Go2Kinematics, forward_kinematics, spatial_velocities
from .model import Go2Model
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

    center, center_velocity_primitive, primitive_rotation, primitive_local = _collision_centers(
        model, kin, velocity
    )
    pair_a, pair_b = geometry.self_pair_a, geometry.self_pair_b
    if pair_a.numel():
        delta = center[:, pair_a] - center[:, pair_b]
        distance_self = torch.linalg.norm(delta, dim=-1).clamp_min(1e-9)
        normal_self = delta / distance_self[..., None]
        self_gap = distance_self - geometry.bounding_radius[pair_a] - geometry.bounding_radius[pair_b]
        relative_velocity = center_velocity_primitive[:, pair_a] - center_velocity_primitive[:, pair_b]
        closing = _smooth_relu(-(relative_velocity * normal_self).sum(-1))
        self_pen = epsilon * torch.nn.functional.softplus(-self_gap / epsilon)
        self_gate = torch.sigmoid(-self_gap / epsilon)
        magnitude = config.contact_stiffness * self_pen + config.contact_damping * self_gate * closing
        force_a = magnitude[..., None] * normal_self
        force_b = -force_a
        wrench = wrench + _scatter_wrench(
            state.batch_size,
            model.n_bodies,
            geometry.owner[pair_a],
            primitive_local[:, pair_a],
            primitive_rotation[:, pair_a],
            force_a,
        )
        wrench = wrench + _scatter_wrench(
            state.batch_size,
            model.n_bodies,
            geometry.owner[pair_b],
            primitive_local[:, pair_b],
            primitive_rotation[:, pair_b],
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
        sample_world=contact_point,
        sample_velocity_world=center_velocity,
        sample_is_foot=geometry.sample_is_foot,
        source=source,
        self_gap=self_gap,
    )
