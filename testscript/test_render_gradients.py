"""WP1b 渲染梯度路径测试（GPU，2026-07-01）。

审计 P0-A 的另一半：验证 PyTorch3D 渲染链「无人机位姿 → 视图矩阵 → 光栅化 → 深度」
端到端可微且梯度有意义。

重要：渲染器用**硬光栅**（drone_renderer.py:203 `blur_radius=0.0, faces_per_pixel=1`），
深度对相机位姿在**轮廓边缘不连续** → 严格逐像素有限差分 gradcheck 不适用（其失败是硬
光栅的已知性质，非 bug）。故本测试做**诚实**的两件事：
  1. 梯度连通/有限/非退化：深度 loss 反传到 drone p_ros 与相机平移 T_view，梯度非 None、
     有限、非零（证明可微路径连通）。
  2. 平滑区方向有限差分：相机正对 bulk 几何时，固定掩码上 mean-depth 的解析梯度与中心
     差分**同号且量级相符**（证明梯度方向正确，不是垃圾）。

需 CUDA + 样例网格；缺任一则 SKIP。
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from drone_renderer import DroneRenderer  # noqa: E402

_M4 = ROOT / "data" / "sample" / "sample4.obj"
MESH = str(_M4 if _M4.exists() else ROOT / "data" / "sample" / "sample.obj")
POSE = (-6.0, -6.0, 1.0)   # ROS 位姿：probe 实测此处相机见 100% 几何（深度 0.31–31m）


def _renderer(dev):
    return DroneRenderer(mesh_path=MESH, device=dev, image_size=(48, 64),
                         focal_length=32.0, num_samples=100, subdivide_times=0)


def _view(r, p_ros, dev):
    Rr = torch.eye(3, device=dev).unsqueeze(0)
    return r.compute_view_matrix(
        p_ros=p_ros, R_ros=Rr,
        cam_mount_R=torch.eye(3, device=dev),
        cam_offset_body=torch.zeros(3, device=dev))


def _depth(r, p_ros, dev):
    Rv, Tv = _view(r, p_ros, dev)
    _, dep = r.render(Rv, Tv, return_tensor=True, return_rgb=False, return_depth=True)
    return dep  # (1,H,W)，背景 zbuf = -1


def test_grad_connectivity_pose(dev):
    r = _renderer(dev)
    p = torch.tensor([POSE], device=dev, requires_grad=True)
    dep = _depth(r, p, dev)
    mask = dep > 0
    assert mask.any(), "相机未见几何（pose 需重选）"
    dep[mask].mean().backward()
    g = p.grad
    assert g is not None and torch.isfinite(g).all(), f"p_ros 梯度非有限: {g}"
    assert g.norm().item() > 0, "p_ros 梯度全零 → 渲染梯度路径断裂"
    print(f"  [PASS] 渲染→深度→drone p_ros 梯度连通/有限/非零 (|g|={g.norm().item():.4f})")


def test_grad_connectivity_camera_translation(dev):
    r = _renderer(dev)
    p = torch.tensor([POSE], device=dev)
    Rv, Tv = _view(r, p, dev)
    Tv = Tv.detach().requires_grad_(True)
    _, dep = r.render(Rv, Tv, return_tensor=True, return_rgb=False, return_depth=True)
    mask = dep > 0
    dep[mask].mean().backward()
    g = Tv.grad
    assert g is not None and torch.isfinite(g).all(), f"T_view 梯度非有限: {g}"
    assert g.norm().item() > 0, "T_view 梯度全零"
    print(f"  [PASS] 深度对相机平移 T_view 梯度有限非零 (|g|={g.norm().item():.4f})")


def test_smooth_regime_directional_fd(dev):
    r = _renderer(dev)
    p0 = torch.tensor([POSE], device=dev)
    base = _depth(r, p0, dev)
    mask = (base > 0).detach()

    def masked_mean(p):
        return _depth(r, p, dev)[mask].mean()

    pa = p0.clone().requires_grad_(True)
    masked_mean(pa).backward()
    g = pa.grad[0]                      # (3,)
    axis = int(g.abs().argmax())
    ana = g[axis].item()
    delta = 1e-3
    e = torch.zeros(3, device=dev)
    e[axis] = delta
    with torch.no_grad():
        fp = masked_mean(p0 + e).item()
        fm = masked_mean(p0 - e).item()
    fd = (fp - fm) / (2 * delta)
    rel = abs(fd - ana) / (abs(ana) + 1e-8)
    print(f"  [diag] axis={axis} analytic={ana:.5f} central-FD={fd:.5f} rel-err={rel:.1%}")
    assert (fd > 0) == (ana > 0), f"平滑区 FD 与解析梯度异号: fd={fd:.5f} ana={ana:.5f}"
    assert rel < 0.30, f"平滑区 FD 与解析相对误差过大 {rel:.1%}（轮廓不连续主导？）"
    print(f"  [PASS] 平滑区方向 FD ≈ 解析梯度（同号，rel-err {rel:.1%} < 30%）")


def main():
    if not torch.cuda.is_available():
        print("SKIP: 无 CUDA（渲染梯度测试需 GPU）")
        return 0
    if not os.path.exists(MESH):
        print(f"SKIP: 样例网格缺失 {MESH}")
        return 0
    dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"渲染梯度测试 @ {dev}, mesh={os.path.basename(MESH)}")
    tests = [test_grad_connectivity_pose,
             test_grad_connectivity_camera_translation,
             test_smooth_regime_directional_fd]
    failed = 0
    for t in tests:
        try:
            t(dev)
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print("=== 渲染梯度测试 " + ("全部通过 ===" if failed == 0 else f"{failed} 失败 ==="))
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
