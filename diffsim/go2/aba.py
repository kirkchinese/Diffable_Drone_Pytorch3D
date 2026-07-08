"""GPU-batched floating-base Featherstone articulated-body dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.transforms import quaternion_to_matrix

from .model import Go2Model
from .spatial import force_cross, motion_cross, skew, spatial_transform
from .types import Go2State


@dataclass
class Go2Kinematics:
    body_rotation: torch.Tensor  # (B,13,3,3), body -> world
    body_position: torch.Tensor  # (B,13,3), body origins in world
    xup: torch.Tensor            # (B,12,6,6), parent motion -> child


@dataclass
class Go2Acceleration:
    base_linear_world: torch.Tensor
    base_angular_body: torch.Tensor
    joint: torch.Tensor
    kinematics: Go2Kinematics
    spatial_velocity: torch.Tensor


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation; axis (J,3), angle (B,J)."""

    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ax = skew(axis).unsqueeze(0)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype).view(1, 1, 3, 3)
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return eye + sin * ax + (1.0 - cos) * (ax @ ax)


def forward_kinematics(model: Go2Model, state: Go2State) -> Go2Kinematics:
    """Compute moving-body poses and spatial parent-to-child transforms."""

    batch = state.batch_size
    base_rotation = quaternion_to_matrix(state.base_quat)
    rotations: list[torch.Tensor] = [base_rotation]
    positions: list[torch.Tensor] = [state.base_pos]
    xup: list[torch.Tensor] = []
    joint_rotation = _axis_rotation(model.joint_axis, state.joint_pos)
    for joint in range(model.n_joints):
        parent = model.parent_indices[joint]
        r_pc = model.origin_rotation[joint].expand(batch, 3, 3) @ joint_rotation[:, joint]
        t_pc = model.origin_translation[joint].expand(batch, 3)
        r_world = rotations[parent] @ r_pc
        p_world = positions[parent] + torch.einsum("bij,bj->bi", rotations[parent], t_pc)
        rotations.append(r_world)
        positions.append(p_world)
        xup.append(spatial_transform(r_pc, t_pc))
    return Go2Kinematics(
        body_rotation=torch.stack(rotations, dim=1),
        body_position=torch.stack(positions, dim=1),
        xup=torch.stack(xup, dim=1),
    )


def _gravity_wrenches(
    model: Go2Model,
    kin: Go2Kinematics,
    gravity_world: torch.Tensor,
    body_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    gravity_body = torch.einsum("bnji,j->bni", kin.body_rotation, gravity_world)
    mass = model.body_mass[None] if body_mass is None else body_mass
    force = mass[..., None] * gravity_body
    moment = torch.cross(model.body_com[None, :, :].expand_as(force), force, dim=-1)
    return torch.cat((moment, force), dim=-1)


def spatial_velocities(model: Go2Model, state: Go2State, kin: Go2Kinematics):
    base_v_body = torch.einsum("bji,bj->bi", kin.body_rotation[:, 0], state.base_vel)
    velocities: list[torch.Tensor] = [torch.cat((state.base_omega, base_v_body), dim=-1)]
    biases: list[torch.Tensor] = []
    subspaces: list[torch.Tensor] = []
    for joint in range(model.n_joints):
        parent = model.parent_indices[joint]
        linear_zero = torch.zeros_like(model.joint_axis[joint])
        subspace = torch.cat((model.joint_axis[joint], linear_zero), dim=-1)
        vj = subspace[None, :] * state.joint_vel[:, joint : joint + 1]
        velocity = torch.einsum("bij,bj->bi", kin.xup[:, joint], velocities[parent]) + vj
        bias = torch.einsum("bij,bj->bi", motion_cross(velocity), vj)
        velocities.append(velocity)
        biases.append(bias)
        subspaces.append(subspace)
    return torch.stack(velocities, dim=1), biases, subspaces


def forward_dynamics(
    model: Go2Model,
    state: Go2State,
    joint_torque: torch.Tensor,
    external_wrench_body: torch.Tensor | None = None,
    gravity_world: torch.Tensor | None = None,
    spatial_inertia: torch.Tensor | None = None,
    body_mass: torch.Tensor | None = None,
) -> Go2Acceleration:
    """Floating-base ABA with external body-frame spatial wrenches."""

    kin = forward_kinematics(model, state)
    velocity, bias, subspaces = spatial_velocities(model, state, kin)
    if gravity_world is None:
        gravity_world = state.base_pos.new_tensor((0.0, 0.0, -9.81))
    external = _gravity_wrenches(model, kin, gravity_world, body_mass)
    if external_wrench_body is not None:
        external = external + external_wrench_body

    inertia_batch = (
        model.spatial_inertia[None].expand(state.batch_size, -1, -1, -1)
        if spatial_inertia is None
        else spatial_inertia
    )
    inertias = [inertia_batch[:, i] for i in range(model.n_bodies)]
    articulated = list(inertias)
    bias_force = []
    for body in range(model.n_bodies):
        momentum = torch.einsum("bij,bj->bi", inertias[body], velocity[:, body])
        bias_force.append(
            torch.einsum("bij,bj->bi", force_cross(velocity[:, body]), momentum) - external[:, body]
        )

    u_store: list[torch.Tensor | None] = [None] * model.n_joints
    d_store: list[torch.Tensor | None] = [None] * model.n_joints
    scalar_store: list[torch.Tensor | None] = [None] * model.n_joints
    for joint in reversed(range(model.n_joints)):
        body = joint + 1
        parent = model.parent_indices[joint]
        s = subspaces[joint]
        u_vec = torch.einsum("bij,j->bi", articulated[body], s)
        d = torch.einsum("bi,i->b", u_vec, s).clamp_min(1e-9)
        scalar_u = joint_torque[:, joint] - torch.einsum("bi,i->b", bias_force[body], s)
        reduced_i = articulated[body] - torch.einsum("bi,bj->bij", u_vec, u_vec) / d[:, None, None]
        reduced_p = (
            bias_force[body]
            + torch.einsum("bij,bj->bi", reduced_i, bias[joint])
            + u_vec * (scalar_u / d)[:, None]
        )
        x = kin.xup[:, joint]
        articulated[parent] = articulated[parent] + x.transpose(-1, -2) @ reduced_i @ x
        bias_force[parent] = bias_force[parent] + torch.einsum(
            "bij,bj->bi", x.transpose(-1, -2), reduced_p
        )
        u_store[joint], d_store[joint], scalar_store[joint] = u_vec, d, scalar_u

    # solve() checks the CUDA ``info`` tensor on the host.  The articulated
    # root inertia is SPD by construction, so use the non-synchronizing form.
    root_acc = torch.linalg.solve_ex(
        articulated[0], -bias_force[0].unsqueeze(-1), check_errors=False
    )[0].squeeze(-1)
    accelerations: list[torch.Tensor] = [root_acc]
    joint_acc: list[torch.Tensor] = []
    for joint in range(model.n_joints):
        parent = model.parent_indices[joint]
        a = torch.einsum("bij,bj->bi", kin.xup[:, joint], accelerations[parent]) + bias[joint]
        qdd = (scalar_store[joint] - (u_store[joint] * a).sum(-1)) / d_store[joint]
        a = a + subspaces[joint][None, :] * qdd[:, None]
        accelerations.append(a)
        joint_acc.append(qdd)

    base_v_body = velocity[:, 0, 3:]
    base_a_world = torch.einsum(
        "bij,bj->bi",
        kin.body_rotation[:, 0],
        root_acc[:, 3:] + torch.cross(state.base_omega, base_v_body, dim=-1),
    )
    return Go2Acceleration(
        base_linear_world=base_a_world,
        base_angular_body=root_acc[:, :3],
        joint=torch.stack(joint_acc, dim=-1),
        kinematics=kin,
        spatial_velocity=velocity,
    )


def inverse_dynamics(
    model: Go2Model,
    state: Go2State,
    base_linear_acc_world: torch.Tensor,
    base_angular_acc_body: torch.Tensor,
    joint_acc: torch.Tensor,
    external_wrench_body: torch.Tensor | None = None,
    gravity_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RNEA oracle returning required root wrench and joint torques."""

    kin = forward_kinematics(model, state)
    velocity, bias, subspaces = spatial_velocities(model, state, kin)
    if gravity_world is None:
        gravity_world = state.base_pos.new_tensor((0.0, 0.0, -9.81))
    external = _gravity_wrenches(model, kin, gravity_world)
    if external_wrench_body is not None:
        external = external + external_wrench_body
    base_v_body = velocity[:, 0, 3:]
    base_linear_derivative = (
        torch.einsum("bji,bj->bi", kin.body_rotation[:, 0], base_linear_acc_world)
        - torch.cross(state.base_omega, base_v_body, dim=-1)
    )
    accelerations: list[torch.Tensor] = [torch.cat((base_angular_acc_body, base_linear_derivative), -1)]
    for joint in range(model.n_joints):
        parent = model.parent_indices[joint]
        a = torch.einsum("bij,bj->bi", kin.xup[:, joint], accelerations[parent]) + bias[joint]
        a = a + subspaces[joint][None, :] * joint_acc[:, joint : joint + 1]
        accelerations.append(a)

    body_force: list[torch.Tensor] = []
    for body in range(model.n_bodies):
        inertia = model.spatial_inertia[body].expand(state.batch_size, 6, 6)
        momentum = torch.einsum("bij,bj->bi", inertia, velocity[:, body])
        body_force.append(
            torch.einsum("bij,bj->bi", inertia, accelerations[body])
            + torch.einsum("bij,bj->bi", force_cross(velocity[:, body]), momentum)
            - external[:, body]
        )
    torques: list[torch.Tensor | None] = [None] * model.n_joints
    for joint in reversed(range(model.n_joints)):
        body = joint + 1
        parent = model.parent_indices[joint]
        torques[joint] = torch.einsum("bi,i->b", body_force[body], subspaces[joint])
        body_force[parent] = body_force[parent] + torch.einsum(
            "bij,bj->bi", kin.xup[:, joint].transpose(-1, -2), body_force[body]
        )
    return body_force[0], torch.stack(torques, dim=-1)
