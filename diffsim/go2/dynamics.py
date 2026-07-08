"""Actuator model and semi-implicit integration for the Go2 ABA core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.transforms import quaternion_multiply

from .aba import Go2Acceleration, forward_dynamics
from .model import Go2Model
from .types import Go2EnvConfig, Go2State


@dataclass
class Go2DynamicsOutput:
    state: Go2State
    acceleration: Go2Acceleration
    torque: torch.Tensor
    desired_joint_pos: torch.Tensor


def quaternion_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotation_vector, dim=-1, keepdim=True)
    half = 0.5 * theta
    scale = torch.where(
        theta < 1e-6,
        0.5 - theta.square() / 48.0,
        torch.sin(half) / theta.clamp_min(1e-12),
    )
    return torch.cat((torch.cos(half), rotation_vector * scale), dim=-1)


def initial_state(model: Go2Model, batch_size: int, *, base_height: float = 0.32) -> Go2State:
    ref = model.spatial_inertia
    base_pos = torch.zeros(batch_size, 3, device=ref.device, dtype=ref.dtype)
    base_pos[:, 2] = base_height
    base_quat = torch.zeros(batch_size, 4, device=ref.device, dtype=ref.dtype)
    base_quat[:, 0] = 1.0
    joint_pos = model.default_joint_pos.expand(batch_size, -1).clone()
    return Go2State(
        base_pos=base_pos,
        base_quat=base_quat,
        base_vel=torch.zeros_like(base_pos),
        base_omega=torch.zeros_like(base_pos),
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        actuator_target=joint_pos.clone(),
    )


def actuator_torque(
    model: Go2Model,
    state: Go2State,
    action: torch.Tensor,
    config: Go2EnvConfig,
    *,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
    motor_strength: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized joint targets to smooth torque-limited PD control."""

    desired = model.default_joint_pos + config.action_scale * torch.tanh(action)
    desired = torch.maximum(model.joint_lower + 0.02, torch.minimum(model.joint_upper - 0.02, desired))
    alpha = 1.0 - torch.exp(state.joint_pos.new_tensor(-config.physics_dt / config.actuator_time_constant))
    filtered = state.actuator_target + alpha * (desired - state.actuator_target)
    if kp is None:
        kp = state.joint_pos.new_full((state.batch_size, 1), config.kp)
    if kd is None:
        kd = state.joint_pos.new_full((state.batch_size, 1), config.kd)
    if motor_strength is None:
        motor_strength = state.joint_pos.new_ones((state.batch_size, 1))
    raw = kp * (filtered - state.joint_pos) - kd * state.joint_vel
    eps = config.joint_limit_smoothing
    lower_pen = eps * torch.nn.functional.softplus((model.joint_lower - state.joint_pos) / eps)
    upper_pen = eps * torch.nn.functional.softplus((state.joint_pos - model.joint_upper) / eps)
    raw = raw + config.joint_limit_stiffness * (lower_pen - upper_pen)
    limits = model.joint_effort * motor_strength
    torque = limits * torch.tanh(raw / limits.clamp_min(1e-6))
    return torque, desired, filtered


def dynamics_step(
    model: Go2Model,
    state: Go2State,
    action: torch.Tensor,
    config: Go2EnvConfig,
    *,
    external_wrench_body: torch.Tensor | None = None,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
    motor_strength: torch.Tensor | None = None,
) -> Go2DynamicsOutput:
    torque, desired, filtered = actuator_torque(
        model, state, action, config, kp=kp, kd=kd, motor_strength=motor_strength
    )
    acceleration = forward_dynamics(model, state, torque, external_wrench_body)
    dt = config.physics_dt
    base_vel = state.base_vel + dt * acceleration.base_linear_world
    base_omega = state.base_omega + dt * acceleration.base_angular_body
    joint_vel = state.joint_vel + dt * acceleration.joint
    base_pos = state.base_pos + dt * base_vel
    delta = quaternion_exp(base_omega * dt)
    base_quat = quaternion_multiply(state.base_quat, delta)
    base_quat = base_quat / torch.linalg.norm(base_quat, dim=-1, keepdim=True).clamp_min(1e-12)
    joint_pos = state.joint_pos + dt * joint_vel
    next_state = Go2State(
        base_pos=base_pos,
        base_quat=base_quat,
        base_vel=base_vel,
        base_omega=base_omega,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        actuator_target=filtered,
    )
    return Go2DynamicsOutput(next_state, acceleration, torque, desired)
