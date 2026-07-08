"""Recurrent perceptive Go2 policy with a shared output head for all legs."""

from __future__ import annotations

import torch
from torch import nn

from .types import Go2Observation


_MIRROR_LEGS = (1, 0, 3, 2)


def _mirror_leg_tensor(value: torch.Tensor) -> torch.Tensor:
    shaped = value.reshape(*value.shape[:-1], 4, 3)
    mirrored = shaped[..., _MIRROR_LEGS, :].clone()
    mirrored[..., 0] = -mirrored[..., 0]
    return mirrored.flatten(-2)


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """Reflect FL/FR and RL/RR targets across the sagittal plane."""

    return _mirror_leg_tensor(action)


def mirror_observation(observation: Go2Observation) -> Go2Observation:
    """Sagittal reflection respecting vector and axial-vector conventions."""

    proprio = observation.proprio.clone()
    proprio[..., 1] *= -1.0  # linear velocity y
    proprio[..., 3] *= -1.0  # axial omega x
    proprio[..., 5] *= -1.0  # axial omega z
    proprio[..., 7] *= -1.0  # projected gravity y
    proprio[..., 9:21] = _mirror_leg_tensor(proprio[..., 9:21])
    proprio[..., 21:33] = _mirror_leg_tensor(proprio[..., 21:33])
    proprio[..., 33:45] = _mirror_leg_tensor(proprio[..., 33:45])
    proprio[..., 46] *= -1.0  # lateral velocity command
    proprio[..., 47] *= -1.0  # yaw-rate command
    heightmap = observation.heightmap.flip(-1)
    depth = None if observation.depth is None else observation.depth.flip(-1)
    return Go2Observation(proprio=proprio, heightmap=heightmap, depth=depth)


class Go2Policy(nn.Module):
    """Shared proprio encoder, selectable perception encoder, GRU, and leg head."""

    def __init__(self, sensor_mode: str = "heightmap", hidden_size: int = 128):
        super().__init__()
        if sensor_mode not in {"blind", "heightmap", "depth"}:
            raise ValueError(f"unknown Go2 sensor mode {sensor_mode!r}")
        self.sensor_mode = sensor_mode
        self.hidden_size = hidden_size
        self.proprio_encoder = nn.Sequential(
            nn.Linear(48, 128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU()
        )
        self.height_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(12 * 8, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 8, 64),
            nn.SiLU(),
        )
        self.blind_embedding = nn.Parameter(torch.zeros(64))
        self.gru = nn.GRUCell(96 + 64, hidden_size)
        # The same MLP is evaluated for FL, FR, RL, RR.  Front/left signs tell
        # it which physical leg it controls without four unrelated heads.
        self.leg_head = nn.Sequential(
            nn.Linear(hidden_size + 9 + 2, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 3),
        )
        self.register_buffer(
            "leg_identity", torch.tensor(((1, 1), (1, -1), (-1, 1), (-1, -1)), dtype=torch.float32)
        )

    def initial_hidden(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        reference = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def _perception(self, observation: Go2Observation) -> torch.Tensor:
        if self.sensor_mode == "blind":
            return self.blind_embedding[None].expand(observation.proprio.shape[0], -1)
        if self.sensor_mode == "heightmap":
            return self.height_encoder(observation.heightmap)
        if observation.depth is None:
            raise ValueError("depth sensor mode requires observation.depth")
        return self.depth_encoder(observation.depth / 6.0)

    def forward(
        self, observation: Go2Observation, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = observation.proprio.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch, observation.proprio.device, observation.proprio.dtype)
        features = torch.cat((self.proprio_encoder(observation.proprio), self._perception(observation)), -1)
        next_hidden = self.gru(features, hidden)
        local = torch.cat(
            (
                observation.proprio[:, 9:21].reshape(batch, 4, 3),
                observation.proprio[:, 21:33].reshape(batch, 4, 3),
                observation.proprio[:, 33:45].reshape(batch, 4, 3),
            ),
            dim=-1,
        )
        recurrent = next_hidden[:, None, :].expand(-1, 4, -1)
        identity = self.leg_identity.to(local).expand(batch, -1, -1)
        action = self.leg_head(torch.cat((recurrent, local, identity), dim=-1)).flatten(1)
        return action, next_hidden
