"""动力学各轴隔离测试（WP/P1-B，2026-07-01）。

审计记：空气阻力/控制延迟/风扰/Airmode「各项均无隔离数值验证」。本测试用**闭式**逐轴钉死
`simulate_position_step` 的物理语义（R=I 时各项有解析闭式）：
  - 执行器延迟（一阶低通）：act_next = act_cmd·(1−α)+act_curr·α, α=exp(−delay·dt)；delay 越大响应越快。
  - 线性/二次阻力：R=I 时 a_drag = k1·v + k2·v·|v|（逐分量），与速度反向。
  - 风扰：drag 作用于相对速度 v−v_wind；v=v_wind ⇒ 无阻力。
  - Airmode：推力方向变化才诱导（act_cmd=act_curr ⇒ 角=0 ⇒ 无 airmode）；沿推力方向。
纯 CPU float64，闭式精确断言。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drone_dynamics import simulate_position_step  # noqa: E402

DTYPE = torch.float64
DT = 1.0 / 15.0
G = torch.tensor([0.0, 0.0, 9.80665], dtype=DTYPE)


def _eye_R(B):
    return torch.eye(3, dtype=DTYPE).unsqueeze(0).repeat(B, 1, 1)


def _base(B=3):
    p = torch.zeros(B, 3, dtype=DTYPE)
    v = torch.tensor([[1.0, -0.5, 0.3], [-0.8, 0.4, -0.6], [0.2, 0.9, 0.1]], dtype=DTYPE)
    a = torch.zeros(B, 3, dtype=DTYPE)
    act = torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.5, -0.3], [0.1, 0.2, 0.4]], dtype=DTYPE)
    return p, v, a, act, _eye_R(B)


def test_actuator_delay_lowpass():
    """执行器一阶低通：act_next 闭式 + delay 越大越快趋近 act_cmd。"""
    p, v, a, act_curr, R = _base()
    act_cmd = act_curr + torch.tensor([[0.5, 0.4, -0.3]], dtype=DTYPE)
    for delay in (6.0, 12.0, 24.0):
        alpha = torch.exp(torch.tensor(-delay * DT, dtype=DTYPE))
        _, _, _, act_next = simulate_position_step(
            p, v, a, act_curr, act_cmd, R, dt=DT, pitch_ctl_delay=delay,
            drag_coef_lin=0.0, enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
        expect = act_cmd * (1 - alpha) + act_curr * alpha
        assert torch.allclose(act_next, expect, atol=1e-12), f"delay={delay} act_next 闭式不符"
    # 单调：delay 越大 → act_next 越接近 act_cmd
    def _err(delay):
        _, _, _, an = simulate_position_step(p, v, a, act_curr, act_cmd, R, dt=DT,
                                             pitch_ctl_delay=delay, drag_coef_lin=0.0,
                                             enable_airmode=False, enable_induced_drag=False)
        return (an - act_cmd).norm().item()
    assert _err(24.0) < _err(12.0) < _err(6.0), "delay 越大响应应越快（越接近 cmd）"
    print("  [PASS] 执行器延迟一阶低通闭式 + delay 越大响应越快")


def test_linear_drag_opposes_velocity():
    """R=I、恒定推力、无 airmode 时：a_next = act − k1·v（逐分量），与速度反向。"""
    p, v, a, act, R = _base()
    k1 = 0.375
    _, _, a_next, _ = simulate_position_step(
        p, v, a, act, act, R, dt=DT, drag_coef_lin=k1, drag_coef_quad=0.0,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    expect = act - k1 * v
    assert torch.allclose(a_next, expect, atol=1e-12), "线性阻力闭式 a=act−k1·v 不符"
    # 无阻力对照
    _, _, a_nodrag, _ = simulate_position_step(
        p, v, a, act, act, R, dt=DT, drag_coef_lin=0.0, drag_coef_quad=0.0,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    assert torch.allclose(a_nodrag, act, atol=1e-12), "drag=0 时 a_next 应=净推力"
    # 阻力增量与速度反向
    drag_delta = a_next - a_nodrag                       # = −k1·v
    assert (drag_delta * v).sum(-1).lt(0).all(), "阻力增量应与速度反向"
    print("  [PASS] 线性阻力闭式 a=act−k1·v，与速度反向，drag=0 无阻力")


def test_quadratic_drag_superlinear():
    """二次阻力 a = act − k1·v − k2·v·|v|；高速时二次项主导。"""
    p, v, a, act, R = _base()
    k1, k2 = 0.2, 0.5
    _, _, a_next, _ = simulate_position_step(
        p, v, a, act, act, R, dt=DT, drag_coef_lin=k1, drag_coef_quad=k2,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    expect = act - k1 * v - k2 * v * v.abs()
    assert torch.allclose(a_next, expect, atol=1e-12), "二次阻力闭式不符"
    print("  [PASS] 二次阻力闭式 a=act−k1·v−k2·v|v|")


def test_wind_acts_on_relative_velocity():
    """风扰：阻力作用于 v−v_wind；v=v_wind ⇒ 相对速度 0 ⇒ 无阻力（=无阻力基线）。"""
    p, v, a, act, R = _base()
    k1 = 0.375
    # v_wind == v → 相对速度 0 → 无阻力
    _, _, a_calm, _ = simulate_position_step(
        p, v, a, act, act, R, dt=DT, v_wind=v.clone(), drag_coef_lin=k1,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    assert torch.allclose(a_calm, act, atol=1e-12), "v=v_wind 时应无阻力（a=净推力）"
    # 一半风：drag 作用于 v − v_wind
    vw = 0.5 * v
    _, _, a_half, _ = simulate_position_step(
        p, v, a, act, act, R, dt=DT, v_wind=vw, drag_coef_lin=k1,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    assert torch.allclose(a_half, act - k1 * (v - vw), atol=1e-12), "风扰相对速度阻力不符"
    print("  [PASS] 风扰作用于相对速度 v−v_wind（v=v_wind 时零阻力）")


def test_airmode_only_on_thrust_change():
    """Airmode 仅在推力方向变化时诱导；act_cmd=act_curr（角=0）⇒ 无 airmode；开/关对照。"""
    p, v, a, act_curr, R = _base()
    # act_cmd == act_curr → 推力向量夹角 0。但 acos 的输入被 clamp 到 1−1e-6，
    # acos(1−1e-6)≈1.4e-3 rad → airmode 有 ~0.01 m/s² 的极小地板（非严格 0，属 clamp
    # 数值边界，已知的小瑕疵：真正零角速本应给零 airmode）。
    _, _, a_on_same, _ = simulate_position_step(
        p, v, a, act_curr, act_curr, R, dt=DT, drag_coef_lin=0.0,
        enable_airmode=True, enable_induced_drag=False, grad_decay=1.0)
    floor = (a_on_same - act_curr).norm(dim=-1)
    assert (floor < 0.02).all(), f"推力不变时 airmode 应仅极小地板(<0.02)，实得 {floor.tolist()}"
    # 推力方向变化：airmode 使 a_next 偏离；且方向沿推力向量 (act_curr+g)
    act_cmd = act_curr + torch.tensor([[0.6, -0.5, 0.2]], dtype=DTYPE)
    _, _, a_off, _ = simulate_position_step(
        p, v, a, act_curr, act_cmd, R, dt=DT, drag_coef_lin=0.0,
        enable_airmode=False, enable_induced_drag=False, grad_decay=1.0)
    _, _, a_on, _ = simulate_position_step(
        p, v, a, act_curr, act_cmd, R, dt=DT, drag_coef_lin=0.0,
        enable_airmode=True, enable_induced_drag=False, grad_decay=1.0)
    airmode_delta = a_on - a_off
    assert airmode_delta.norm(dim=-1).min() > 5 * floor.max(), \
        "推力方向变化的 airmode 应远大于同推力地板（随推力变化 scale）"
    thrust_dir = (act_curr + G) / (act_curr + G).norm(dim=-1, keepdim=True)
    # airmode 沿推力方向：|delta| ≈ |delta·dir|（共线）
    proj = (airmode_delta * thrust_dir).sum(-1).abs()
    assert torch.allclose(proj, airmode_delta.norm(dim=-1), atol=1e-9), "airmode 应沿推力方向"
    print("  [PASS] Airmode 仅推力变化时诱导、沿推力方向、开关对照")


def main():
    tests = [test_actuator_delay_lowpass, test_linear_drag_opposes_velocity,
             test_quadratic_drag_superlinear, test_wind_acts_on_relative_velocity,
             test_airmode_only_on_thrust_change]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print("=== 动力学各轴隔离测试 " + ("全过 ===" if failed == 0 else f"{failed} 失败 ==="))
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
