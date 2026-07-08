"""Batched spatial-vector algebra using [angular, linear] convention."""

from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), dim=-1).reshape(*v.shape[:-1], 3, 3)


def motion_cross(v: torch.Tensor) -> torch.Tensor:
    """Spatial cross-product matrix crm(v)."""

    w, linear = v[..., :3], v[..., 3:]
    wx, vx = skew(w), skew(linear)
    z = torch.zeros_like(wx)
    top = torch.cat((wx, z), dim=-1)
    bottom = torch.cat((vx, wx), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def force_cross(v: torch.Tensor) -> torch.Tensor:
    """Spatial force cross-product matrix crf(v) = -crm(v)^T."""

    return -motion_cross(v).transpose(-1, -2)


def spatial_transform(rotation_parent_child: torch.Tensor, translation_parent: torch.Tensor) -> torch.Tensor:
    """Motion transform from parent coordinates to child coordinates.

    ``rotation_parent_child`` maps child vectors to parent coordinates and
    ``translation_parent`` is the child origin in parent coordinates.
    """

    rt = rotation_parent_child.transpose(-1, -2)
    z = torch.zeros_like(rt)
    top = torch.cat((rt, z), dim=-1)
    bottom = torch.cat((-rt @ skew(translation_parent), rt), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def spatial_inertia(mass: torch.Tensor, com: torch.Tensor, inertia_com: torch.Tensor) -> torch.Tensor:
    """Spatial inertia about a frame origin, expressed in that frame."""

    cx = skew(com)
    eye = torch.eye(3, device=com.device, dtype=com.dtype).expand(*com.shape[:-1], 3, 3)
    upper_left = inertia_com - mass[..., None, None] * (cx @ cx)
    upper_right = mass[..., None, None] * cx
    lower_left = -upper_right
    lower_right = mass[..., None, None] * eye
    return torch.cat(
        (torch.cat((upper_left, upper_right), dim=-1), torch.cat((lower_left, lower_right), dim=-1)),
        dim=-2,
    )


def transform_inertia(inertia: torch.Tensor, rotation_owner_link: torch.Tensor, translation_owner: torch.Tensor) -> torch.Tensor:
    """Move a link-frame spatial inertia into an owner frame."""

    x_link_owner = spatial_transform(rotation_owner_link, translation_owner)
    return x_link_owner.transpose(-1, -2) @ inertia @ x_link_owner
