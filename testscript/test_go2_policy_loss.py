"""Go2 policy modality, symmetry transform, and locomotion-loss gates."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.envs.go2 import Go2Env  # noqa: E402
from diffsim.go2.policy import Go2Policy, mirror_action, mirror_observation  # noqa: E402
from diffsim.go2.types import Go2EnvConfig, Go2Observation  # noqa: E402
from diffsim.losses import LossBuildContext, available_losses, make_loss  # noqa: E402
from diffsim.losses.go2 import Go2LossContext  # noqa: E402


def test_symmetry_transforms_are_involutions():
    observation = Go2Observation(
        proprio=torch.randn(3, 48),
        heightmap=torch.randn(3, 12, 8),
        depth=torch.randn(3, 1, 48, 64),
    )
    twice = mirror_observation(mirror_observation(observation))
    assert torch.equal(twice.proprio, observation.proprio)
    assert torch.equal(twice.heightmap, observation.heightmap)
    assert torch.equal(twice.depth, observation.depth)
    action = torch.randn(3, 12)
    assert torch.equal(mirror_action(mirror_action(action)), action)


def test_policy_modalities_and_shared_head():
    observation = Go2Observation(
        proprio=torch.randn(2, 48),
        heightmap=torch.randn(2, 12, 8),
        depth=torch.rand(2, 1, 48, 64) * 6.0,
    )
    for mode in ("blind", "heightmap", "depth"):
        policy = Go2Policy(mode)
        action, hidden = policy(observation)
        assert action.shape == (2, 12)
        assert hidden.shape == (2, policy.hidden_size)
        action.square().mean().backward()
        assert all(parameter.grad is not None for parameter in policy.leg_head.parameters())


def test_locomotion_loss_is_registered_and_differentiable():
    assert "go2_locomotion" in available_losses()
    config = Go2EnvConfig(batch_size=2)
    env = Go2Env(config, torch.device("cpu"), seed=7, randomize=False)
    previous_action = env.previous_action
    action = torch.zeros(2, 12, requires_grad=True)
    output = env.step(action)
    loss_fn = make_loss(
        "go2_locomotion", LossBuildContext(args=Namespace(), control_dt=config.control_dt)
    )
    result = loss_fn(
        Go2LossContext(
            state=output.state,
            action=action,
            previous_action=previous_action,
            command=env.command,
            torque=env.last_dynamics.torque,
            contact=env.last_contact,
            scene=env.scene,
            model=env.model,
            valid=~output.terminated,
        )
    )
    assert torch.isfinite(result.loss)
    expected = {
        "velocity", "yaw", "upright", "body_height", "joint_limit", "energy",
        "action_rate", "foot_slip", "penetration", "nonfoot_contact", "foothold",
    }
    assert set(result.terms) == expected
    gradient = torch.autograd.grad(result.loss, action)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().amax() > 1e-9


def main():
    tests = [
        test_symmetry_transforms_are_involutions,
        test_policy_modalities_and_shared_head,
        test_locomotion_loss_is_registered_and_differentiable,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} Go2 policy/loss checks passed.")


if __name__ == "__main__":
    main()
