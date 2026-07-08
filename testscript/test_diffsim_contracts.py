"""Contract checks for the incremental differentiable-simulation boundary."""

from argparse import Namespace
from pathlib import Path
import sys
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffsim import EnvBuildContext, available_envs, make_env  # noqa: E402
from diffsim.losses import (  # noqa: E402
    LossBuildContext,
    WeightedLossComposer,
    make_loss,
)
from diffsim.registry import Registry  # noqa: E402
from loss import DroneLoss  # noqa: E402


def _drone_args():
    return Namespace(
        batch_size=8,
        mesh_path="scene.obj",
        image_height=48,
        image_width=64,
        grad_decay=0.8,
        num_samples=1234,
        subdivide_times=0,
        n_drones_per_group=None,
        min_spawn_inter_distance=0.0,
        arena_range=6.0,
        margin_max=0.8,
        coef_v=1.0,
        coef_speed=0.0,
        coef_v_pred=2.0,
        coef_collide=2.0,
        coef_obj_avoidance=1.5,
        coef_d_acc=0.01,
        coef_d_jerk=0.001,
        coef_d_snap=0.0,
        coef_ground_affinity=0.0,
        coef_bias=0.0,
    )


def test_registry_failures_are_explicit():
    registry = Registry("demo")
    registry.register("one", lambda: 1)
    assert registry.create("ONE") == 1
    try:
        registry.register("one", lambda: 2)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registrations must fail")
    try:
        registry.create("missing")
    except ValueError as exc:
        assert "Available: one" in str(exc)
    else:
        raise AssertionError("unknown registry keys must fail")


def test_drone_factory_preserves_constructor_mapping():
    args = _drone_args()
    captured = {}

    def fake_simulator(**kwargs):
        captured.update(kwargs)
        return object()

    context = EnvBuildContext(
        args=args,
        device=torch.device("cpu"),
        control_dt=1 / 15,
        focal_length=52.0,
        scene_generator="scene-generator",
        extras={"enable_airmode": False},
    )
    assert "drone" in available_envs()
    with patch("diffsim.envs.drone.DroneSimulator", side_effect=fake_simulator):
        env = make_env("drone", context)

    assert type(env) is object
    assert captured["batch_size"] == 8
    assert captured["dt"] == 1 / 15
    assert captured["device"] == torch.device("cpu")
    assert captured["image_size"] == (48, 64)
    assert captured["focal_length"] == 52.0
    assert captured["scene_generator"] == "scene-generator"
    assert captured["enable_airmode"] is False
    assert captured["n_drones_per_group"] == 8
    assert args.min_spawn_inter_distance == 2.1


def test_legacy_drone_loss_is_built_unchanged():
    args = _drone_args()
    loss = make_loss(
        "drone_navigation",
        LossBuildContext(args=args, control_dt=1 / 15),
    )
    assert isinstance(loss, DroneLoss)
    assert loss.coefs["v"] == 1.0
    assert loss.coefs["obj_avoidance"] == 1.5
    assert loss.ctl_dt == 1 / 15


def test_weighted_composer_keeps_gradient_on_device():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    composer = WeightedLossComposer(
        terms={
            "squared": lambda ctx: ctx.square().mean(),
            "linear": lambda ctx: ctx.mean(),
        },
        weights={"squared": 2.0, "linear": 0.5},
    )
    output = composer(x)
    assert output.loss.device == x.device
    assert set(output.terms) == {"squared", "linear"}
    output.loss.backward()
    expected = 2.0 * x.detach() + 0.25
    assert torch.allclose(x.grad, expected)


def main():
    tests = [
        test_registry_failures_are_explicit,
        test_drone_factory_preserves_constructor_mapping,
        test_legacy_drone_loss_is_built_unchanged,
        test_weighted_composer_keeps_gradient_on_device,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} diffsim contract checks passed.")


if __name__ == "__main__":
    main()
