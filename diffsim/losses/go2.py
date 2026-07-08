"""Independent differentiable locomotion objective for the Go2 environment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.transforms import quaternion_to_matrix
from torch import nn

from ..go2.contact import ContactOutput
from ..go2.model import Go2Model
from ..go2.scene import Go2Scene
from ..go2.types import Go2State
from .core import LossOutput
from .factory import LossBuildContext, register_loss


@dataclass
class Go2LossContext:
    state: Go2State
    action: torch.Tensor
    previous_action: torch.Tensor
    command: torch.Tensor
    torque: torch.Tensor
    contact: ContactOutput
    scene: Go2Scene
    model: Go2Model
    valid: torch.Tensor | None = None


@dataclass(frozen=True)
class Go2LossWeights:
    velocity: float = 3.0
    yaw: float = 1.0
    upright: float = 1.5
    body_height: float = 0.8
    joint_limit: float = 0.25
    energy: float = 2.0e-4
    action_rate: float = 0.08
    foot_slip: float = 0.8
    penetration: float = 8.0
    nonfoot_contact: float = 1.5
    foothold: float = 0.6


def _masked_mean(value: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    per_batch = value.reshape(value.shape[0], -1).mean(-1)
    if valid is None:
        return per_batch.mean()
    weight = valid.to(per_batch)
    return (per_batch * weight).sum() / weight.sum().clamp_min(1.0)


class Go2LocomotionLoss(nn.Module):
    def __init__(self, weights: Go2LossWeights | None = None):
        super().__init__()
        self.weights = Go2LossWeights() if weights is None else weights

    def _foothold_feasibility(self, context: Go2LossContext) -> torch.Tensor:
        contact = context.contact
        foot = contact.sample_is_foot
        points = contact.sample_world[:, foot]
        probability = contact.probability[:, foot]
        gap = contact.gap[:, foot]
        _, normal, _ = context.scene.signed_distance(points)
        inside = torch.nn.functional.softplus(-gap / 0.004) * 0.004
        steep = torch.relu(0.72 - normal[..., 2])
        reach = torch.relu(torch.linalg.norm(points[..., :2] - context.state.base_pos[:, None, :2], dim=-1) - 0.62)

        owners = context.model.collisions.sample_owner[foot]
        leg_owners = points.new_tensor((3, 6, 9, 12), dtype=torch.long)
        membership = (owners[:, None] == leg_owners[None, :]).to(points)
        leg_weight = torch.einsum("bs,sl->bsl", probability, membership)
        denominator = leg_weight.sum(1).clamp_min(1e-5)
        feet_xy = torch.einsum("bsl,bsi->bli", leg_weight, points[..., :2]) / denominator[..., None]
        support_weight = denominator.clamp(0.0, 1.0)
        centroid = (feet_xy * support_weight[..., None]).sum(1) / support_weight.sum(1, keepdim=True).clamp_min(1e-5)
        spread = torch.sqrt(
            ((feet_xy - centroid[:, None]).square().sum(-1) * support_weight).sum(1)
            / support_weight.sum(1).clamp_min(1e-5)
            + 1e-8
        )
        com_offset = torch.linalg.norm(context.state.base_pos[:, :2] - centroid, dim=-1)
        unstable = torch.relu(com_offset - 0.65 * spread - 0.03)
        local = (inside.square() + steep.square() + reach.square()) * probability
        return local.mean(-1) + unstable.square()

    def forward(self, context: Go2LossContext) -> LossOutput:
        state, contact, valid = context.state, context.contact, context.valid
        rotation = quaternion_to_matrix(state.base_quat)
        body_velocity = torch.einsum("bji,bj->bi", rotation, state.base_vel)
        projected_gravity = torch.einsum(
            "bji,j->bi", rotation, state.base_pos.new_tensor((0.0, 0.0, -1.0))
        )
        terrain_height, _ = context.scene.terrain_height_and_normal(state.base_pos[:, None, :2])
        relative_height = state.base_pos[:, 2] - terrain_height[:, 0]
        lower = torch.relu(context.model.joint_lower - state.joint_pos)
        upper = torch.relu(state.joint_pos - context.model.joint_upper)
        normal_speed = (contact.sample_velocity_world * contact.normal).sum(-1)
        tangent_velocity = contact.sample_velocity_world - normal_speed[..., None] * contact.normal
        tangent_speed2 = tangent_velocity.square().sum(-1)
        foot = contact.sample_is_foot
        nonfoot = ~foot
        terms = {
            "velocity": _masked_mean((body_velocity[:, :2] - context.command[:, :2]).square(), valid),
            "yaw": _masked_mean((state.base_omega[:, 2] - context.command[:, 2]).square()[:, None], valid),
            "upright": _masked_mean(projected_gravity[:, :2].square(), valid),
            "body_height": _masked_mean((relative_height - 0.30).square()[:, None], valid),
            "joint_limit": _masked_mean(lower.square() + upper.square(), valid),
            "energy": _masked_mean((context.torque * state.joint_vel).abs(), valid),
            "action_rate": _masked_mean((context.action - context.previous_action).square(), valid),
            "foot_slip": _masked_mean(contact.probability[:, foot] * tangent_speed2[:, foot], valid),
            "penetration": _masked_mean(torch.relu(-contact.gap).square(), valid),
            "nonfoot_contact": _masked_mean(contact.probability[:, nonfoot].square(), valid),
            "foothold": _masked_mean(self._foothold_feasibility(context)[:, None], valid),
        }
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        return LossOutput(loss=total, terms=terms)


def build_go2_locomotion_loss(context: LossBuildContext) -> Go2LocomotionLoss:
    args = context.args
    defaults = Go2LossWeights()
    weights = Go2LossWeights(
        **{
            name: getattr(args, f"go2_loss_{name}", getattr(defaults, name))
            for name in defaults.__dataclass_fields__
        }
    )
    return Go2LocomotionLoss(weights)


register_loss("go2_locomotion", build_go2_locomotion_loss)
