"""Contact, scene-SDF, penetration, and fixed-mode gradient gates."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.go2.aba import forward_dynamics, forward_kinematics  # noqa: E402
from diffsim.go2.contact import contact_wrenches  # noqa: E402
from diffsim.go2.dynamics import dynamics_step, initial_state  # noqa: E402
from diffsim.go2.model import compile_go2_model  # noqa: E402
from diffsim.go2.scene import flat_scene  # noqa: E402
from diffsim.go2.types import Go2EnvConfig  # noqa: E402


DTYPE = torch.float64


def _standing_setup():
    config = Go2EnvConfig(batch_size=1)
    model = compile_go2_model(dtype=DTYPE)
    state = initial_state(model, 1, base_height=0.0)
    kin = forward_kinematics(model, state)
    geometry = model.collisions
    owner = geometry.sample_owner
    sample = kin.body_position[:, owner] + torch.einsum(
        "bsij,sj->bsi", kin.body_rotation[:, owner], geometry.sample_pos
    )
    bottom = sample[..., 2] - geometry.sample_radius
    nominal_height = -bottom[:, geometry.sample_is_foot].amin() - 0.002
    state.base_pos[:, 2] = nominal_height
    return config, model, state, flat_scene(config, "cpu", DTYPE)


def test_flat_scene_sdf_and_normal():
    config = Go2EnvConfig(batch_size=1)
    scene = flat_scene(config, "cpu", DTYPE)
    points = torch.tensor([[[0.0, 0.0, 0.2], [1.0, -1.0, -0.03]]], dtype=DTYPE)
    distance, normal, source = scene.signed_distance(points)
    assert torch.allclose(distance, points[..., 2], atol=1e-12)
    assert torch.allclose(normal, torch.tensor((0.0, 0.0, 1.0), dtype=DTYPE).expand_as(normal))
    assert (source == -1).all()


def test_contact_cone_dissipation_and_self_clearance():
    config, model, state, scene = _standing_setup()
    state.base_vel[:, 0] = 0.3
    contact = contact_wrenches(model, state, scene, config)
    tangent_norm = torch.linalg.norm(contact.tangential_force, dim=-1)
    cone = scene.friction * contact.normal_force
    assert torch.all(tangent_norm <= cone + 1e-9)
    tangent_velocity = contact.sample_velocity_world.clone()
    tangent_velocity[..., 2] = 0.0
    assert (contact.tangential_force * tangent_velocity).sum(-1).amax() <= 1e-9
    assert contact.self_gap.amin() > 0.01


def test_standing_penetration_bound():
    config, model, state, scene = _standing_setup()
    action = torch.zeros(1, 12, dtype=DTYPE)
    contact = None
    for _ in range(800):
        contact = contact_wrenches(model, state, scene, config)
        state = dynamics_step(
            model, state, action, config, external_wrench_body=contact.wrench_body
        ).state
    actual_penetration = torch.relu(-contact.gap)
    foot_penetration = actual_penetration[:, model.collisions.sample_is_foot].amax()
    assert foot_penetration < 0.003
    assert torch.isfinite(state.base_pos).all()
    assert state.base_pos[0, 2] > 0.20


def test_fixed_contact_gradient_matches_finite_difference():
    config, model, state, scene = _standing_setup()
    base_z = state.base_pos[:, 2].detach().clone().requires_grad_(True)

    def objective(z):
        local = state.clone()
        local.base_pos = torch.cat((state.base_pos[:, :2], z[:, None]), dim=-1)
        contact = contact_wrenches(model, local, scene, config)
        acceleration = forward_dynamics(
            model, local, torch.zeros(1, 12, dtype=DTYPE), contact.wrench_body
        )
        return acceleration.base_linear_world[:, 2].sum()

    analytic = torch.autograd.grad(objective(base_z), base_z)[0]
    eps = 1e-6
    finite = (objective(base_z.detach() + eps) - objective(base_z.detach() - eps)) / (2 * eps)
    assert torch.allclose(analytic, finite, atol=2e-3, rtol=2e-4)
    # The floating base and articulated contacts couple the sign of this
    # derivative; the fixed-mode gate is agreement and a non-degenerate signal.
    assert analytic.abs().item() > 1e-6


def test_nonadjacent_self_contact_has_equal_opposite_force():
    config = Go2EnvConfig(batch_size=1)
    model = compile_go2_model(dtype=DTYPE)
    state = initial_state(model, 1, base_height=2.0)
    state.joint_pos[:] = torch.tensor(
        [[
            -0.19247669, 2.00703669, -0.94081855,
            0.85189545, 0.09826255, -2.11648297,
            -0.61497575, 4.51418161, -2.15443802,
            -0.75189614, 1.85362518, -0.87318873,
        ]],
        dtype=DTYPE,
    )
    contact = contact_wrenches(model, state, flat_scene(config, "cpu", DTYPE), config)
    assert contact.self_gap.amin() < -0.04
    kinematics = forward_kinematics(model, state)
    force_world = torch.einsum(
        "bnij,bnj->bni", kinematics.body_rotation, contact.wrench_body[..., 3:]
    )
    residual = torch.linalg.norm(force_world.sum(1))
    magnitude = torch.linalg.norm(force_world, dim=-1).sum(1).clamp_min(1.0)
    assert residual < 1e-9 * magnitude


def main():
    tests = [
        test_flat_scene_sdf_and_normal,
        test_contact_cone_dissipation_and_self_clearance,
        test_standing_penetration_bound,
        test_fixed_contact_gradient_matches_finite_difference,
        test_nonadjacent_self_contact_has_equal_opposite_force,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} Go2 contact checks passed.")


if __name__ == "__main__":
    main()
