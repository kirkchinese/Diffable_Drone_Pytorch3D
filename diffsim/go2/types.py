"""Tensor-only public types for the Go2 environment."""

from __future__ import annotations

from dataclasses import dataclass, fields
import torch


Go2Action = torch.Tensor


@dataclass
class Go2State:
    """Batched generalized state.

    ``base_pos`` and ``base_vel`` are expressed in world coordinates.  The
    quaternion is scalar-first and maps body vectors to world.  ``base_omega``
    is expressed in the base frame.  Joint order is FL, FR, RL, RR with
    hip/thigh/calf inside each leg.
    """

    base_pos: torch.Tensor
    base_quat: torch.Tensor
    base_vel: torch.Tensor
    base_omega: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    actuator_target: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.base_pos.shape[0]

    def detach(self) -> "Go2State":
        return Go2State(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def clone(self) -> "Go2State":
        return Go2State(**{f.name: getattr(self, f.name).clone() for f in fields(self)})


@dataclass
class Go2Observation:
    proprio: torch.Tensor       # (B,48)
    heightmap: torch.Tensor     # (B,12,8)
    depth: torch.Tensor | None  # (B,1,48,64)


@dataclass(frozen=True)
class Go2EnvConfig:
    batch_size: int = 64
    physics_dt: float = 1.0e-3
    control_dt: float = 2.0e-2
    dtype: torch.dtype = torch.float32
    contact_stiffness: float = 1.2e4
    contact_damping: float = 350.0
    contact_smoothing: float = 2.0e-3
    friction_velocity: float = 0.05
    friction: float = 0.8
    actuator_time_constant: float = 0.02
    action_scale: float = 0.45
    kp: float = 45.0
    kd: float = 1.2
    joint_limit_stiffness: float = 30.0
    joint_limit_smoothing: float = 0.03
    terrain_rows: int = 12
    terrain_cols: int = 8
    terrain_x_min: float = -0.35
    terrain_x_max: float = 1.20
    terrain_y_min: float = -0.45
    terrain_y_max: float = 0.45
    max_obstacles: int = 24
    depth_height: int = 48
    depth_width: int = 64
    camera_pos_body: tuple[float, float, float] = (0.285, 0.0, 0.01)
    camera_pitch_deg: float = -10.0

    @property
    def substeps(self) -> int:
        ratio = self.control_dt / self.physics_dt
        rounded = round(ratio)
        if abs(ratio - rounded) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        return int(rounded)
