"""GPU-native differentiable terrain and analytic obstacle SDFs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import Go2EnvConfig, Go2State


SHAPE_SPHERE = 0
SHAPE_CAPSULE = 1
SHAPE_BOX = 2


@dataclass
class Go2Scene:
    slope: torch.Tensor              # (B,2), dz/dx and dz/dy
    bump_centers: torch.Tensor       # (B,H,2)
    bump_heights: torch.Tensor       # (B,H)
    bump_widths: torch.Tensor        # (B,H)
    obstacle_kind: torch.Tensor      # (B,K)
    obstacle_position: torch.Tensor  # (B,K,3)
    obstacle_rotation: torch.Tensor  # (B,K,3,3), local -> world
    obstacle_size: torch.Tensor      # sphere radius | capsule (r,half) | box half extents
    obstacle_mask: torch.Tensor      # (B,K)
    friction: torch.Tensor           # (B,1)

    @property
    def batch_size(self) -> int:
        return self.slope.shape[0]

    def terrain_height_and_normal(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return height and upward normal for points shaped (B,N,2)."""

        delta = xy[:, :, None, :] - self.bump_centers[:, None, :, :]
        width2 = self.bump_widths[:, None, :].square().clamp_min(1e-8)
        gaussian = torch.exp(-0.5 * delta.square().sum(-1) / width2)
        height = (xy * self.slope[:, None, :]).sum(-1)
        height = height + (self.bump_heights[:, None, :] * gaussian).sum(-1)
        grad = self.slope[:, None, :] + (
            self.bump_heights[:, None, :, None]
            * gaussian[..., None]
            * (-delta / width2[..., None])
        ).sum(2)
        normal = torch.cat((-grad, torch.ones_like(height)[..., None]), dim=-1)
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp_min(1e-9)
        return height, normal

    def signed_distance(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scene union SDF for points (B,N,3).

        Returns signed distance, outward normal, and source index (-1 for terrain).
        The minimum selection is piecewise differentiable; gradients within a
        fixed contact mode remain exact and mode changes are exposed separately.
        """

        height, terrain_normal = self.terrain_height_and_normal(points[..., :2])
        terrain_distance = (points[..., 2] - height) * terrain_normal[..., 2]

        relative_world = points[:, :, None, :] - self.obstacle_position[:, None, :, :]
        relative = torch.einsum("bkji,bnkj->bnki", self.obstacle_rotation, relative_world)
        size = self.obstacle_size[:, None, :, :]

        sphere_delta = relative
        sphere_norm = torch.linalg.norm(sphere_delta, dim=-1).clamp_min(1e-9)
        sphere_distance = sphere_norm - size[..., 0]
        sphere_normal_local = sphere_delta / sphere_norm[..., None]

        capsule_z = relative[..., 2].clamp(-size[..., 1], size[..., 1])
        capsule_nearest = torch.stack(
            (torch.zeros_like(capsule_z), torch.zeros_like(capsule_z), capsule_z), dim=-1
        )
        capsule_delta = relative - capsule_nearest
        capsule_norm = torch.linalg.norm(capsule_delta, dim=-1).clamp_min(1e-9)
        capsule_distance = capsule_norm - size[..., 0]
        capsule_normal_local = capsule_delta / capsule_norm[..., None]

        box_q = relative.abs() - size
        box_out = box_q.clamp_min(0.0)
        box_out_norm = torch.linalg.norm(box_out, dim=-1)
        box_distance = box_out_norm + box_q.amax(dim=-1).clamp_max(0.0)
        outside_normal = relative.sign() * box_out / box_out_norm[..., None].clamp_min(1e-9)
        inside_axis = torch.nn.functional.one_hot(box_q.argmax(dim=-1), 3).to(relative.dtype)
        inside_normal = relative.sign() * inside_axis
        box_normal_local = torch.where((box_out_norm > 1e-9)[..., None], outside_normal, inside_normal)

        kinds = self.obstacle_kind[:, None, :]
        obstacle_distance = torch.where(
            kinds == SHAPE_SPHERE,
            sphere_distance,
            torch.where(kinds == SHAPE_CAPSULE, capsule_distance, box_distance),
        )
        normal_local = torch.where(
            (kinds == SHAPE_SPHERE)[..., None],
            sphere_normal_local,
            torch.where((kinds == SHAPE_CAPSULE)[..., None], capsule_normal_local, box_normal_local),
        )
        obstacle_normal = torch.einsum("bkij,bnkj->bnki", self.obstacle_rotation, normal_local)
        obstacle_distance = torch.where(
            self.obstacle_mask[:, None, :], obstacle_distance, torch.full_like(obstacle_distance, 1e6)
        )

        all_distance = torch.cat((terrain_distance[..., None], obstacle_distance), dim=-1)
        minimum, index = all_distance.min(dim=-1)
        all_normal = torch.cat((terrain_normal[:, :, None, :], obstacle_normal), dim=2)
        selected = torch.gather(all_normal, 2, index[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        return minimum, selected, index - 1

    def heightmap(self, state: Go2State, config: Go2EnvConfig) -> torch.Tensor:
        xs = torch.linspace(
            config.terrain_x_min,
            config.terrain_x_max,
            config.terrain_rows,
            device=state.base_pos.device,
            dtype=state.base_pos.dtype,
        )
        ys = torch.linspace(
            config.terrain_y_min,
            config.terrain_y_max,
            config.terrain_cols,
            device=state.base_pos.device,
            dtype=state.base_pos.dtype,
        )
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((gx.flatten(), gy.flatten()), dim=-1)
        from pytorch3d.transforms import quaternion_to_matrix

        rotation = quaternion_to_matrix(state.base_quat)
        world_xy = state.base_pos[:, None, :2] + torch.einsum(
            "bij,nj->bni", rotation[:, :2, :2], grid
        )
        height, _ = self.terrain_height_and_normal(world_xy)
        relative_world = world_xy[:, :, None, :] - self.obstacle_position[:, None, :, :2]
        relative_local = torch.einsum(
            "bkji,bnkj->bnki", self.obstacle_rotation[..., :2, :2], relative_world
        )
        size = self.obstacle_size[:, None, :, :]
        radial2 = relative_local.square().sum(-1)
        sphere_cap = torch.sqrt((size[..., 0].square() - radial2).clamp_min(0.0))
        sphere_top = self.obstacle_position[:, None, :, 2] + sphere_cap
        capsule_top = self.obstacle_position[:, None, :, 2] + size[..., 1] + sphere_cap
        box_inside = (relative_local.abs() - size[..., :2]).amax(-1) <= 0
        radial_inside = radial2 <= size[..., 0].square()
        kinds = self.obstacle_kind[:, None, :]
        top = torch.where(
            kinds == SHAPE_SPHERE,
            sphere_top,
            torch.where(kinds == SHAPE_CAPSULE, capsule_top, self.obstacle_position[:, None, :, 2] + size[..., 2]),
        )
        inside = torch.where(kinds == SHAPE_BOX, box_inside, radial_inside)
        base_kind = self.obstacle_kind
        vertical_extent = torch.where(
            base_kind == SHAPE_SPHERE,
            self.obstacle_size[..., 0],
            torch.where(
                base_kind == SHAPE_CAPSULE,
                self.obstacle_size[..., 0] + self.obstacle_size[..., 1],
                self.obstacle_size[..., 2],
            ),
        )
        grounded = self.obstacle_position[:, :, 2] - vertical_extent <= 0.04
        valid = inside & self.obstacle_mask[:, None, :] & grounded[:, None, :]
        top = torch.where(valid, top, torch.full_like(top, -1e6))
        obstacle_top = top.amax(-1)
        height = torch.maximum(height, obstacle_top)
        relative = height - state.base_pos[:, 2:3]
        return relative.reshape(state.batch_size, config.terrain_rows, config.terrain_cols)


def _yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(*yaw.shape, 3, 3)


def flat_scene(config: Go2EnvConfig, device: torch.device | str, dtype: torch.dtype) -> Go2Scene:
    batch, obstacles = config.batch_size, config.max_obstacles
    zeros2 = torch.zeros(batch, 2, device=device, dtype=dtype)
    return Go2Scene(
        slope=zeros2,
        bump_centers=torch.zeros(batch, 1, 2, device=device, dtype=dtype),
        bump_heights=torch.zeros(batch, 1, device=device, dtype=dtype),
        bump_widths=torch.ones(batch, 1, device=device, dtype=dtype),
        obstacle_kind=torch.zeros(batch, obstacles, device=device, dtype=torch.long),
        obstacle_position=torch.zeros(batch, obstacles, 3, device=device, dtype=dtype),
        obstacle_rotation=torch.eye(3, device=device, dtype=dtype).expand(batch, obstacles, 3, 3).clone(),
        obstacle_size=torch.zeros(batch, obstacles, 3, device=device, dtype=dtype),
        obstacle_mask=torch.zeros(batch, obstacles, device=device, dtype=torch.bool),
        friction=torch.full((batch, 1), config.friction, device=device, dtype=dtype),
    )


def random_scene(
    config: Go2EnvConfig,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Go2Scene:
    """Generate smooth terrain plus grounded and hanging analytic obstacles."""

    batch, count = config.batch_size, config.max_obstacles

    def uniform(shape, low, high):
        return low + (high - low) * torch.rand(shape, generator=generator, device=device, dtype=dtype)

    slope = uniform((batch, 2), -0.08, 0.08)
    bump_centers = torch.stack(
        (uniform((batch, 8), 0.2, 3.0), uniform((batch, 8), -0.6, 0.6)), dim=-1
    )
    bump_heights = uniform((batch, 8), -0.025, 0.06)
    bump_widths = uniform((batch, 8), 0.10, 0.30)
    kind = torch.randint(0, 3, (batch, count), generator=generator, device=device)
    position = torch.stack(
        (uniform((batch, count), 0.35, 4.0), uniform((batch, count), -0.7, 0.7), uniform((batch, count), 0.03, 0.45)),
        dim=-1,
    )
    size = uniform((batch, count, 3), 0.03, 0.12)
    # Boxes are sparse foothold obstacles; some are raised to become hanging bars.
    size[..., 0] = uniform((batch, count), 0.04, 0.16)
    size[..., 1] = uniform((batch, count), 0.04, 0.25)
    size[..., 2] = uniform((batch, count), 0.03, 0.08)
    grounded = torch.rand((batch, count), generator=generator, device=device) < 0.75
    vertical_extent = torch.where(
        kind == SHAPE_SPHERE,
        size[..., 0],
        torch.where(kind == SHAPE_CAPSULE, size[..., 0] + size[..., 1], size[..., 2]),
    )
    position[..., 2] = torch.where(grounded, vertical_extent, position[..., 2])
    yaw = uniform((batch, count), -torch.pi, torch.pi)
    active_count = torch.randint(4, count + 1, (batch, 1), generator=generator, device=device)
    mask = torch.arange(count, device=device)[None, :] < active_count
    return Go2Scene(
        slope=slope,
        bump_centers=bump_centers,
        bump_heights=bump_heights,
        bump_widths=bump_widths,
        obstacle_kind=kind,
        obstacle_position=position,
        obstacle_rotation=_yaw_rotation(yaw),
        obstacle_size=size,
        obstacle_mask=mask,
        friction=uniform((batch, 1), 0.45, 1.15),
    )
