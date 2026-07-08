#!/usr/bin/env python3
"""CPU float64 gradient-correctness checks for drone_dynamics.py."""

import os
import sys
import traceback
from pathlib import Path

import torch
from torch.autograd import gradcheck


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from drone_dynamics import (  # noqa: E402
    g_decay,
    simulate_position_step,
    solve_attitude_from_thrust_and_goal_vec,
)


DTYPE = torch.float64
DEVICE = torch.device("cpu")
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-5
GRADCHECK_RTOL = 1e-3

PASS = 0
FAIL = 0


def _assert_allclose(name, actual, expected, atol=1e-10, rtol=1e-10):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name} mismatch\nactual={actual}\nexpected={expected}\n"
        f"max_abs={(actual - expected).abs().max().item():.3e}"
    )


def _make_sim_inputs(requires_grad=True):
    kwargs = {"dtype": DTYPE, "device": DEVICE}
    p = torch.tensor([[0.20, -0.10, 1.10], [-0.30, 0.40, 1.40]], **kwargs)
    v = torch.tensor([[0.31, -0.27, 0.15], [-0.22, 0.18, -0.13]], **kwargs)
    a = torch.tensor([[0.04, -0.03, 0.02], [-0.01, 0.05, -0.04]], **kwargs)
    act_curr = torch.tensor([[0.35, -0.21, 0.18], [-0.28, 0.24, 0.32]], **kwargs)
    act_cmd = torch.tensor([[0.45, -0.12, 0.30], [-0.18, 0.31, 0.40]], **kwargs)
    R = torch.eye(3, **kwargs).unsqueeze(0).repeat(2, 1, 1)
    tensors = (p, v, a, act_curr, act_cmd, R)
    if requires_grad:
        tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    return tensors


def _run_gradcheck(label, func, inputs, input_names):
    try:
        gradcheck(
            func,
            inputs,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            raise_exception=True,
        )
    except Exception as exc:
        failing_inputs = []
        base_inputs = tuple(t.detach().clone() for t in inputs)
        for idx, name in enumerate(input_names):
            single_inputs = list(base_inputs)
            single_inputs[idx] = single_inputs[idx].detach().clone().requires_grad_(True)

            def single_arg_func(x, idx=idx, single_inputs=single_inputs):
                args = list(single_inputs)
                args[idx] = x
                return func(*args)

            try:
                gradcheck(
                    single_arg_func,
                    (single_inputs[idx],),
                    eps=GRADCHECK_EPS,
                    atol=GRADCHECK_ATOL,
                    rtol=GRADCHECK_RTOL,
                    raise_exception=True,
                )
            except Exception as single_exc:
                failing_inputs.append(f"{name}: {single_exc}")
        failed = "; ".join(failing_inputs) if failing_inputs else "unable to isolate input"
        raise AssertionError(
            f"{label} gradcheck FAILED. Failing input(s): {failed}. "
            f"Original error: {exc}"
        ) from exc
    print(f"  [PASS] {label} gradcheck passed for {', '.join(input_names)}")


def test_g_decay_formula():
    x = torch.tensor(
        [[1.0, -2.0, 3.5], [0.25, -0.75, 1.25]],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    y = g_decay(x, 0.4)
    assert torch.equal(y, x), "g_decay forward must be exact identity"
    print("  [PASS] g_decay forward identity")

    upstream = torch.tensor(
        [[0.7, -1.1, 1.3], [-0.2, 0.5, -0.9]],
        dtype=DTYPE,
        device=DEVICE,
    )
    y.backward(upstream)
    expected_scalar = upstream * 0.4
    _assert_allclose("scalar decay backward", x.grad, expected_scalar)
    # This intentionally diverges from the true identity gradient; that is the decay mechanism.
    assert not torch.allclose(x.grad, upstream), "param < 1 must differ from identity gradient"
    print("  [PASS] scalar backward gradient is upstream * 0.4")

    x_sample = torch.tensor(
        [[-0.8, 0.2, 1.4], [1.1, -1.5, 0.6]],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    per_sample = torch.tensor([0.2, 0.7], dtype=DTYPE, device=DEVICE)
    upstream_sample = torch.tensor(
        [[1.5, -0.5, 0.25], [-0.75, 1.25, -1.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    g_decay(x_sample, per_sample).backward(upstream_sample)
    expected_sample = upstream_sample * per_sample.unsqueeze(-1)
    _assert_allclose("per-sample decay backward", x_sample.grad, expected_sample)
    assert not torch.allclose(x_sample.grad, upstream_sample), (
        "per-sample param < 1 must differ from identity gradient"
    )
    print("  [PASS] per-sample backward gradient broadcasts over (B, 3)")


def test_simulate_position_step_gradcheck():
    inputs = _make_sim_inputs(requires_grad=True)

    def sim_func(p, v, a, act_curr, act_cmd, R):
        return simulate_position_step(
            p,
            v,
            a,
            act_curr,
            act_cmd,
            R,
            dt=1.0 / 15.0,
            grad_decay=1.0,
            enable_airmode=True,
            enable_induced_drag=False,
        )

    _run_gradcheck(
        "simulate_position_step true-gradient (grad_decay=1.0)",
        sim_func,
        inputs,
        ("p", "v", "a", "act_curr", "act_cmd", "R"),
    )


def _decay_wiring_grads(grad_decay):
    p, v, a, act_curr, act_cmd, R = _make_sim_inputs(requires_grad=True)
    p_next, v_next, _, _ = simulate_position_step(
        p,
        v,
        a,
        act_curr,
        act_cmd,
        R,
        dt=1.0 / 15.0,
        drag_coef_lin=0.0,
        drag_coef_quad=0.0,
        enable_airmode=False,
        enable_induced_drag=False,
        grad_decay=grad_decay,
    )

    # Check the direct decayed outputs separately; p_next also has the
    # non-decayed Verlet v*dt term, and drag uses raw v before g_decay.
    p_grad = torch.autograd.grad(p_next.sum(), p, retain_graph=True)[0]
    v_grad, act_curr_grad, act_cmd_grad = torch.autograd.grad(
        v_next.sum(),
        (v, act_curr, act_cmd),
    )
    return p_grad, v_grad, act_curr_grad, act_cmd_grad


def test_simulate_position_step_decay_wiring():
    dt = 1.0 / 15.0
    decay_factor = 0.9 ** dt
    p_grad_1, v_grad_1, act_curr_grad_1, act_cmd_grad_1 = _decay_wiring_grads(1.0)
    p_grad_decay, v_grad_decay, act_curr_grad_decay, act_cmd_grad_decay = _decay_wiring_grads(0.9)

    _assert_allclose("p gradient decay wiring", p_grad_decay, p_grad_1 * decay_factor)
    _assert_allclose("v gradient decay wiring", v_grad_decay, v_grad_1 * decay_factor)
    _assert_allclose("act_curr gradient unaffected", act_curr_grad_decay, act_curr_grad_1)
    _assert_allclose("act_cmd gradient unaffected", act_cmd_grad_decay, act_cmd_grad_1)
    print(f"  [PASS] p and v gradients scale by grad_decay**dt = {decay_factor:.12f}")
    print("  [PASS] action gradients are unchanged by grad_decay")


def test_solve_attitude_gradcheck():
    kwargs = {"dtype": DTYPE, "device": DEVICE}
    thrust_vector = torch.tensor(
        [[0.40, -0.20, 9.80], [-0.30, 0.50, 9.70]],
        **kwargs,
        requires_grad=True,
    )
    velocity = torch.tensor(
        [[1.20, 0.30, 0.20], [-0.40, 0.90, 0.10]],
        **kwargs,
        requires_grad=True,
    )
    R_old = torch.eye(3, **kwargs).unsqueeze(0).repeat(2, 1, 1)

    def attitude_func(thrust, vel):
        return solve_attitude_from_thrust_and_goal_vec(
            thrust,
            vel,
            R_old,
            yaw_inertia=5.0,
            dt=0.02,
            yaw_ctl_delay=12.0,
        )

    _run_gradcheck(
        "solve_attitude_from_thrust_and_goal_vec",
        attitude_func,
        (thrust_vector, velocity),
        ("thrust_vector", "velocity"),
    )


def test_verlet_forward_closed_form():
    """前向正确性（补梯度证据）：无阻力/airmode、act_cmd=act_curr 时 Verlet = 闭式。"""
    dt = 1.0 / 15.0
    p, v, a, act_curr, _, R = _make_sim_inputs(requires_grad=False)
    p_next, v_next, a_next, act_next = simulate_position_step(
        p, v, a, act_curr, act_curr, R, dt=dt,
        drag_coef_lin=0.0, drag_coef_quad=0.0,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0,
    )
    _assert_allclose("act_next==act_curr", act_next, act_curr)
    _assert_allclose("a_next==net thrust (no drag)", a_next, act_curr)
    _assert_allclose("Verlet position", p_next, p + v * dt + 0.5 * a * dt ** 2)
    _assert_allclose("Verlet velocity", v_next, v + 0.5 * (a + a_next) * dt)
    print("  [PASS] Verlet forward closed-form (position/velocity) + no-drag a_next")


def test_solve_attitude_orthonormal():
    """姿态解算输出必须是合法旋转矩阵：R^T R = I 且 det = +1。"""
    kwargs = {"dtype": DTYPE, "device": DEVICE}
    thrust_vector = torch.tensor([[0.40, -0.20, 9.80], [-0.30, 0.50, 9.70]], **kwargs)
    velocity = torch.tensor([[1.20, 0.30, 0.20], [-0.40, 0.90, 0.10]], **kwargs)
    R_old = torch.eye(3, **kwargs).unsqueeze(0).repeat(2, 1, 1)
    R_new = solve_attitude_from_thrust_and_goal_vec(thrust_vector, velocity, R_old, dt=0.02)
    B = R_new.shape[0]
    eye = torch.eye(3, **kwargs).expand(B, 3, 3)
    _assert_allclose("R^T R == I", torch.bmm(R_new.transpose(1, 2), R_new), eye, atol=1e-9)
    _assert_allclose("det(R) == 1", torch.linalg.det(R_new), torch.ones(B, **kwargs), atol=1e-9)
    print("  [PASS] solve_attitude output is orthonormal with det=+1")


def test_simulate_position_step_gradcheck_induced_drag():
    """诱导阻力(H-force)+二次阻力路径的解析梯度=有限差分（覆盖阻力轴）。"""
    inputs = _make_sim_inputs(requires_grad=True)

    def sim_func(p, v, a, act_curr, act_cmd, R):
        return simulate_position_step(
            p, v, a, act_curr, act_cmd, R,
            dt=1.0 / 15.0, grad_decay=1.0,
            enable_airmode=False, enable_induced_drag=True,
            drag_coef_lin=0.375, drag_coef_quad=0.1, rotor_drag_coef=0.07,
        )

    _run_gradcheck(
        "simulate_position_step induced+quadratic drag (grad_decay=1.0)",
        sim_func,
        inputs,
        ("p", "v", "a", "act_curr", "act_cmd", "R"),
    )


def _run_case(name, func):
    global PASS, FAIL
    print(f"\n[TEST] {name}")
    try:
        func()
    except Exception as exc:
        FAIL += 1
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
    else:
        PASS += 1
        print(f"[PASS] {name}")


def main():
    torch.set_default_dtype(DTYPE)
    torch.set_printoptions(precision=10, sci_mode=False)
    print("=" * 70)
    print("Gradient correctness tests (CPU, float64)")
    print("=" * 70)

    _run_case("GDecay formula conformance", test_g_decay_formula)
    _run_case("Verlet forward closed-form", test_verlet_forward_closed_form)
    _run_case("simulate_position_step gradcheck at grad_decay=1.0", test_simulate_position_step_gradcheck)
    _run_case("simulate_position_step gradcheck (induced+quad drag)", test_simulate_position_step_gradcheck_induced_drag)
    _run_case("simulate_position_step grad_decay wiring", test_simulate_position_step_decay_wiring)
    _run_case("solve_attitude_from_thrust_and_goal_vec gradcheck", test_solve_attitude_gradcheck)
    _run_case("solve_attitude orthonormal output", test_solve_attitude_orthonormal)

    total = PASS + FAIL
    print("\n" + "=" * 70)
    print(f"SUMMARY: {PASS} PASS, {FAIL} FAIL ({total} total)")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
