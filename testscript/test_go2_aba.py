"""Deterministic gates for the batched floating-base Go2 ABA core."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.go2.aba import forward_dynamics, inverse_dynamics  # noqa: E402
from diffsim.go2.dynamics import initial_state  # noqa: E402
from diffsim.go2.model import JOINT_ORDER, SDK_TO_CANONICAL, compile_go2_model  # noqa: E402


def test_model_compilation():
    model = compile_go2_model(dtype=torch.float64)
    assert model.n_bodies == 13
    assert model.n_joints == 12
    assert len(model.collisions.link_names) == 27
    assert abs(float(model.body_mass.sum()) - 15.019) < 1e-10
    assert JOINT_ORDER[:3] == ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint")
    assert SDK_TO_CANONICAL == (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


def test_free_fall_is_rigid_and_uniform():
    model = compile_go2_model(dtype=torch.float64)
    state = initial_state(model, 2)
    acceleration = forward_dynamics(model, state, torch.zeros(2, 12, dtype=torch.float64))
    expected = torch.tensor((0.0, 0.0, -9.81), dtype=torch.float64).expand(2, 3)
    assert torch.allclose(acceleration.base_linear_world, expected, atol=1e-10)
    assert acceleration.base_angular_body.abs().max() < 1e-10
    assert acceleration.joint.abs().max() < 1e-10


def test_aba_and_rnea_are_inverse():
    torch.manual_seed(7)
    model = compile_go2_model(dtype=torch.float64)
    state = initial_state(model, 4)
    state.joint_pos = state.joint_pos + 0.08 * torch.randn_like(state.joint_pos)
    state.joint_vel = 0.2 * torch.randn_like(state.joint_vel)
    torque = 3.0 * torch.randn(4, 12, dtype=torch.float64)
    acceleration = forward_dynamics(model, state, torque)
    root, recovered = inverse_dynamics(
        model,
        state,
        acceleration.base_linear_world,
        acceleration.base_angular_body,
        acceleration.joint,
    )
    assert root.abs().max() < 1e-9
    assert torch.allclose(recovered, torque, atol=1e-9, rtol=1e-9)


def test_torque_gradient_matches_directional_difference():
    torch.manual_seed(11)
    model = compile_go2_model(dtype=torch.float64)
    state = initial_state(model, 1)
    state.joint_pos = state.joint_pos + 0.03 * torch.randn_like(state.joint_pos)
    torque = torch.randn(1, 12, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(torque)
    direction = direction / direction.norm()

    def objective(value):
        acc = forward_dynamics(model, state, value)
        return 0.4 * acc.joint.square().mean() + acc.base_linear_world.square().mean()

    loss = objective(torque)
    analytic = torch.autograd.grad(loss, torque)[0]
    eps = 1e-6
    finite = (objective(torque.detach() + eps * direction) - objective(torque.detach() - eps * direction)) / (2 * eps)
    projected = (analytic * direction).sum()
    assert torch.allclose(projected, finite, atol=1e-5, rtol=2e-4)


def main():
    tests = [
        test_model_compilation,
        test_free_fall_is_rigid_and_uniform,
        test_aba_and_rnea_are_inverse,
        test_torque_gradient_matches_directional_difference,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} Go2 ABA checks passed.")


if __name__ == "__main__":
    main()
