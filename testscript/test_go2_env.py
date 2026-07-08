"""Go2 environment registration, observation, rollout, and gradient smoke gates."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim import EnvBuildContext, available_envs, make_env  # noqa: E402


def _build(batch_size=2, device="cpu", sensor_mode="heightmap"):
    args = Namespace(
        batch_size=batch_size,
        physics_dt=1e-3,
        ctl_dt=2e-2,
        seed=13,
        random_scene=False,
        go2_max_obstacles=8,
        go2_sensor_mode=sensor_mode,
    )
    context = EnvBuildContext(
        args=args,
        device=torch.device(device),
        control_dt=args.ctl_dt,
        focal_length=32.0,
    )
    return make_env("go2", context)


def test_registration_and_contract():
    assert "go2" in available_envs()
    env = _build()
    reset = env.reset()
    assert reset.observation.proprio.shape == (2, 48)
    assert reset.observation.heightmap.shape == (2, 12, 8)
    assert reset.observation.depth is None
    assert reset.terminated.shape == (2,)
    assert env.metadata.physics_dt == 1e-3
    assert env.metadata.control_dt == 2e-2


def test_control_step_is_differentiable():
    env = _build()
    action = torch.zeros(2, 12, requires_grad=True)
    output = env.step(action)
    objective = output.state.base_pos[:, 2].sum() + 0.01 * output.state.joint_pos.square().sum()
    gradient = torch.autograd.grad(objective, action)[0]
    assert gradient.shape == action.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().amax() > 1e-9
    assert output.diagnostics["max_penetration"].shape == (2,)


def test_depth_contract_if_cuda_available():
    if not torch.cuda.is_available():
        return
    env = _build(batch_size=1, device="cuda:0", sensor_mode="depth")
    observation = env.observe()
    assert observation.depth.shape == (1, 1, 48, 64)
    assert torch.isfinite(observation.depth).all()
    assert observation.depth.amin() >= 0.05
    assert observation.depth.amax() <= 6.0


def main():
    tests = [
        test_registration_and_contract,
        test_control_step_is_differentiable,
        test_depth_contract_if_cuda_available,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} Go2 environment checks passed.")


if __name__ == "__main__":
    main()
