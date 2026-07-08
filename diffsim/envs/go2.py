"""Registered GPU-vectorized Go2 environment."""

from __future__ import annotations

from dataclasses import fields

import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix

from ..api import EnvMetadata, StepOutput
from ..factory import EnvBuildContext, register_env
from ..go2.aba import forward_kinematics
from ..go2.contact import ContactOutput, contact_wrenches
from ..go2.dynamics import Go2DynamicsOutput, dynamics_step, initial_state
from ..go2.model import Go2Model, compile_go2_model
from ..go2.scene import Go2Scene, flat_scene, random_scene
from ..go2.types import Go2EnvConfig, Go2Observation, Go2State


class Go2Env:
    """Differentiable floating-base articulated Go2 environment."""

    def __init__(
        self,
        config: Go2EnvConfig,
        device: torch.device,
        *,
        seed: int = 0,
        randomize: bool = True,
        sensor_mode: str = "heightmap",
    ):
        self.config = config
        self.device = torch.device(device)
        self.B = config.batch_size
        self.model: Go2Model = compile_go2_model(device=self.device, dtype=config.dtype)
        self.metadata = EnvMetadata(
            name="go2",
            batch_size=self.B,
            device=self.device,
            physics_dt=config.physics_dt,
            control_dt=config.control_dt,
            differentiable_dynamics=True,
            differentiable_observation=sensor_mode == "heightmap",
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.randomize = randomize
        self.sensor_mode = sensor_mode
        self.scene: Go2Scene = flat_scene(config, self.device, config.dtype)
        self.command = torch.zeros(self.B, 3, device=self.device, dtype=config.dtype)
        self.previous_action = torch.zeros(self.B, 12, device=self.device, dtype=config.dtype)
        self.kp = torch.full((self.B, 1), config.kp, device=self.device, dtype=config.dtype)
        self.kd = torch.full((self.B, 1), config.kd, device=self.device, dtype=config.dtype)
        self.motor_strength = torch.ones(self.B, 1, device=self.device, dtype=config.dtype)
        self.body_mass = self.model.body_mass[None].expand(self.B, -1).clone()
        self.spatial_inertia = self.model.spatial_inertia[None].expand(self.B, -1, -1, -1).clone()
        self.sensor_noise_std = torch.zeros(self.B, 1, device=self.device, dtype=config.dtype)
        self.state = initial_state(self.model, self.B, base_height=self._nominal_height())
        self.last_contact: ContactOutput | None = None
        self.last_dynamics: Go2DynamicsOutput | None = None
        self._renderer = None
        self.reset()

    def _nominal_height(self) -> float:
        state = initial_state(self.model, 1, base_height=0.0)
        kin = forward_kinematics(self.model, state)
        geometry = self.model.collisions
        owner = geometry.sample_owner
        rotation = kin.body_rotation[:, owner]
        samples = kin.body_position[:, owner] + torch.einsum(
            "bsij,sj->bsi", rotation, geometry.sample_pos
        )
        bottom = samples[..., 2] - geometry.sample_radius
        lowest_foot = bottom[:, geometry.sample_is_foot].amin()
        return float((-lowest_foot - 0.002).detach().cpu())

    def _sample_uniform(self, shape, low: float, high: float):
        return low + (high - low) * torch.rand(
            shape, generator=self.generator, device=self.device, dtype=self.config.dtype
        )

    @staticmethod
    def _select(mask: torch.Tensor, replacement: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        view = (mask.shape[0],) + (1,) * (current.ndim - 1)
        return torch.where(mask.reshape(view), replacement, current)

    def reset(self, mask: torch.Tensor | None = None) -> StepOutput[Go2State, Go2Observation]:
        reset_all = mask is None
        if mask is None:
            mask = torch.ones(self.B, device=self.device, dtype=torch.bool)
        elif mask.shape != (self.B,):
            raise ValueError(f"reset mask must have shape {(self.B,)}, got {tuple(mask.shape)}")

        if self.randomize:
            new_scene = random_scene(
                self.config, self.generator, self.device, self.config.dtype
            )
        else:
            new_scene = flat_scene(self.config, self.device, self.config.dtype)
        new_state = initial_state(self.model, self.B, base_height=self._nominal_height())
        if self.randomize:
            new_state.joint_pos = new_state.joint_pos + self._sample_uniform((self.B, 12), -0.04, 0.04)
            rpy = self._sample_uniform((self.B, 3), -0.04, 0.04)
            # Obstacles occupy the initial forward corridor; yaw commands still
            # provide turning while reset headings remain inside that corridor.
            rpy[:, 2] = self._sample_uniform((self.B,), -0.08, 0.08)
            new_state.base_quat = matrix_to_quaternion(euler_angles_to_matrix(rpy, "XYZ"))
            new_state.base_vel = self._sample_uniform((self.B, 3), -0.03, 0.03)
            new_state.base_omega = self._sample_uniform((self.B, 3), -0.03, 0.03)
        new_state.actuator_target = new_state.joint_pos.clone()
        new_command = torch.zeros_like(self.command)
        new_command[:, 0] = self._sample_uniform((self.B,), 0.3, 2.0) if self.randomize else 0.5
        if self.randomize:
            new_command[:, 1] = self._sample_uniform((self.B,), -0.3, 0.3)
            new_command[:, 2] = self._sample_uniform((self.B,), -0.8, 0.8)
            new_kp = self._sample_uniform((self.B, 1), 35.0, 60.0)
            new_kd = self._sample_uniform((self.B, 1), 0.8, 1.8)
            new_motor = self._sample_uniform((self.B, 1), 0.85, 1.15)
            mass_scale = self._sample_uniform((self.B, self.model.n_bodies), 0.90, 1.10)
            inertia_scale = self._sample_uniform((self.B, self.model.n_bodies), 0.90, 1.10)
            new_mass = self.model.body_mass[None] * mass_scale
            new_inertia = self.model.spatial_inertia[None] * mass_scale[..., None, None]
            new_inertia[..., :3, :3] = (
                new_inertia[..., :3, :3] * inertia_scale[..., None, None]
            )
            new_sensor_noise = self._sample_uniform((self.B, 1), 0.0, 0.015)
        else:
            new_kp = torch.full_like(self.kp, self.config.kp)
            new_kd = torch.full_like(self.kd, self.config.kd)
            new_motor = torch.ones_like(self.motor_strength)
            new_mass = self.model.body_mass[None].expand(self.B, -1)
            new_inertia = self.model.spatial_inertia[None].expand(self.B, -1, -1, -1)
            new_sensor_noise = torch.zeros_like(self.sensor_noise_std)

        if reset_all:
            self.scene = new_scene
            self.state = new_state
        else:
            for field in fields(Go2Scene):
                current = getattr(self.scene, field.name)
                replacement = getattr(new_scene, field.name)
                setattr(self.scene, field.name, self._select(mask, replacement, current))
            for field in fields(Go2State):
                current = getattr(self.state, field.name)
                replacement = getattr(new_state, field.name)
                setattr(self.state, field.name, self._select(mask, replacement, current))
        self.previous_action = self._select(mask, torch.zeros_like(self.previous_action), self.previous_action)
        self.command = self._select(mask, new_command, self.command)
        self.kp = self._select(mask, new_kp, self.kp)
        self.kd = self._select(mask, new_kd, self.kd)
        self.motor_strength = self._select(mask, new_motor, self.motor_strength)
        self.body_mass = self._select(mask, new_mass, self.body_mass)
        self.spatial_inertia = self._select(mask, new_inertia, self.spatial_inertia)
        self.sensor_noise_std = self._select(mask, new_sensor_noise, self.sensor_noise_std)
        self.last_contact = None
        self.last_dynamics = None
        observation = self.observe()
        return StepOutput(state=self.state, observation=observation, terminated=self.terminated())

    def observe(self, *, include_depth: bool | None = None) -> Go2Observation:
        rotation = quaternion_to_matrix(self.state.base_quat)
        body_velocity = torch.einsum("bji,bj->bi", rotation, self.state.base_vel)
        projected_gravity = torch.einsum(
            "bji,j->bi", rotation, self.state.base_pos.new_tensor((0.0, 0.0, -1.0))
        )
        proprio = torch.cat(
            (
                body_velocity,
                self.state.base_omega,
                projected_gravity,
                self.state.joint_pos - self.model.default_joint_pos,
                self.state.joint_vel,
                self.previous_action,
                self.command,
            ),
            dim=-1,
        )
        if proprio.shape[-1] != 48:
            raise AssertionError("Go2 proprioception contract must remain 48-D")
        heightmap = self.scene.heightmap(self.state, self.config)
        if self.randomize:
            proprio = proprio + self.sensor_noise_std * torch.randn(
                proprio.shape, generator=self.generator, device=self.device, dtype=self.config.dtype
            )
            heightmap = heightmap + self.sensor_noise_std[:, None] * torch.randn(
                heightmap.shape, generator=self.generator, device=self.device, dtype=self.config.dtype
            )
        wants_depth = self.sensor_mode == "depth" if include_depth is None else include_depth
        depth = self.render_depth() if wants_depth else None
        return Go2Observation(proprio=proprio, heightmap=heightmap, depth=depth)

    def terminated(self) -> torch.Tensor:
        rotation = quaternion_to_matrix(self.state.base_quat)
        terrain_height, _ = self.scene.terrain_height_and_normal(self.state.base_pos[:, None, :2])
        relative_height = self.state.base_pos[:, 2] - terrain_height[:, 0]
        return (relative_height < 0.12) | (rotation[:, 2, 2] < 0.3)

    def step(self, action: torch.Tensor, **_kwargs) -> StepOutput[Go2State, Go2Observation]:
        if action.shape != (self.B, 12):
            raise ValueError(f"Go2 action must have shape {(self.B, 12)}, got {tuple(action.shape)}")
        for _ in range(self.config.substeps):
            contact = contact_wrenches(self.model, self.state, self.scene, self.config)
            dynamics = dynamics_step(
                self.model,
                self.state,
                action,
                self.config,
                external_wrench_body=contact.wrench_body,
                kp=self.kp,
                kd=self.kd,
                motor_strength=self.motor_strength,
                spatial_inertia=self.spatial_inertia,
                body_mass=self.body_mass,
            )
            self.state = dynamics.state
        self.previous_action = action
        self.last_contact, self.last_dynamics = contact, dynamics
        actual_penetration = torch.relu(-contact.gap)
        diagnostics = {
            "max_penetration": actual_penetration.amax(dim=-1),
            "mean_contact_probability": contact.probability.mean(dim=-1),
            "joint_torque_rms": dynamics.torque.square().mean(dim=-1).sqrt(),
            "self_min_gap": contact.self_gap.amin(dim=-1)
            if contact.self_gap.shape[-1]
            else action.new_full((self.B,), float("inf")),
        }
        return StepOutput(
            state=self.state,
            observation=self.observe(),
            terminated=self.terminated(),
            diagnostics=diagnostics,
        )

    def render_depth(self) -> torch.Tensor:
        if self._renderer is None:
            from ..go2.render import Go2DepthRenderer

            self._renderer = Go2DepthRenderer(self.model, self.config, self.device)
        depth = self._renderer.render(self.state, self.scene)
        if self.randomize:
            depth = depth + self.sensor_noise_std[:, :, None, None] * torch.randn(
                depth.shape, generator=self.generator, device=self.device, dtype=self.config.dtype
            )
        return depth.clamp(0.05, 6.0)


def build_go2_env(context: EnvBuildContext) -> Go2Env:
    args = context.args
    batch_size = getattr(args, "go2_batch_size", None) or args.batch_size
    config = Go2EnvConfig(
        batch_size=batch_size,
        physics_dt=getattr(args, "physics_dt", 1.0e-3),
        control_dt=getattr(args, "go2_control_dt", context.control_dt),
        dtype=torch.float32,
        max_obstacles=getattr(args, "go2_max_obstacles", 24),
    )
    return Go2Env(
        config,
        context.device,
        seed=getattr(args, "seed", 0) or 0,
        randomize=getattr(args, "go2_randomize", True),
        sensor_mode=getattr(args, "go2_sensor_mode", "heightmap"),
    )


register_env("go2", build_go2_env)
