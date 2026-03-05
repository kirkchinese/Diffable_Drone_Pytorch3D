#!/usr/bin/env python3
"""
Unit tests for DecayController.
Standalone script with assert + print (no pytest).
"""

import sys
sys.path.insert(0, '/home/misaka/Diffable_Drone_Pytorch3D')

import torch
from model import DecayController


def test_param_count():
    """num_params should be 257 (256 weights + 1 bias for Linear(256,1))."""
    print("\n=== Test 1: Parameter count ===")
    dc = DecayController()
    assert dc.num_params == 257, f"Expected 257, got {dc.num_params}"
    print(f"  - num_params = {dc.num_params}: OK")
    print("✓ test_param_count passed")


def test_output_shape():
    """For input (B, 256), output must be (B,)."""
    print("\n=== Test 2: Output shape ===")
    dc = DecayController()
    for B in [1, 4, 16, 32]:
        x = torch.randn(B, 256)
        out = dc(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
        print(f"  - B={B}: shape={out.shape}: OK")
    print("✓ test_output_shape passed")


def test_output_range():
    """Output must be in [0.2, 1.0] (default decay_min=0.2, decay_range=0.8)."""
    print("\n=== Test 3: Output range ===")
    dc = DecayController()
    # Use CMA-ES-like random weights to push outputs to extremes
    with torch.no_grad():
        dc.linear.weight.normal_(0, 2.0)
        dc.linear.bias.fill_(0.0)
    x = torch.randn(1000, 256)
    out = dc(x)
    assert out.min() >= 0.2 - 1e-6, f"Output {out.min():.6f} below 0.2"
    assert out.max() <= 1.0 + 1e-6, f"Output {out.max():.6f} above 1.0"
    print(f"  - Range: [{out.min().item():.4f}, {out.max().item():.4f}]: OK")
    print("✓ test_output_range passed")


def test_zero_initialization():
    """Zero-init: sigmoid(0)=0.5 → output = 0.2 + 0.8*0.5 = 0.6."""
    print("\n=== Test 4: Zero initialization ===")
    dc = DecayController()
    x = torch.randn(16, 256)
    out = dc(x)
    expected = 0.6
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6), \
        f"Expected all outputs ≈ {expected}, got {out}"
    print(f"  - All outputs = {out[0].item():.4f} (expected {expected}): OK")
    print("✓ test_zero_initialization passed")


def test_params_vector_roundtrip():
    """get_params_vector / set_params_vector must be lossless."""
    print("\n=== Test 5: Params vector roundtrip ===")
    dc = DecayController()

    # Set some non-trivial weights
    with torch.no_grad():
        dc.linear.weight.normal_(0, 1.0)
        dc.linear.bias.fill_(0.42)

    vec = dc.get_params_vector()
    assert vec.shape == (257,), f"Expected vector length 257, got {vec.shape}"
    print(f"  - Vector length = {vec.shape[0]}: OK")

    # Create new controller, set from vector, verify match
    dc2 = DecayController()
    dc2.set_params_vector(vec.clone())
    vec2 = dc2.get_params_vector()
    assert torch.allclose(vec, vec2, atol=1e-7), "Roundtrip mismatch!"
    print("  - Roundtrip exact match: OK")

    # Verify same output
    x = torch.randn(8, 256)
    out1 = dc(x)
    out2 = dc2(x)
    assert torch.allclose(out1, out2, atol=1e-6), "Output mismatch after roundtrip!"
    print("  - Output match after roundtrip: OK")
    print("✓ test_params_vector_roundtrip passed")


def test_custom_range():
    """DecayController(decay_min=0.3, decay_range=0.5) → output in [0.3, 0.8]."""
    print("\n=== Test 6: Custom range ===")
    dc = DecayController(decay_min=0.3, decay_range=0.5)
    # Push weights to get extreme outputs
    with torch.no_grad():
        dc.linear.weight.normal_(0, 2.0)
    x = torch.randn(1000, 256)
    out = dc(x)
    assert out.min() >= 0.3 - 1e-6, f"Output {out.min():.6f} below 0.3"
    assert out.max() <= 0.8 + 1e-6, f"Output {out.max():.6f} above 0.8"
    print(f"  - Range: [{out.min().item():.4f}, {out.max().item():.4f}] ⊂ [0.3, 0.8]: OK")

    # Zero-init should give 0.3 + 0.5*0.5 = 0.55
    dc_zero = DecayController(decay_min=0.3, decay_range=0.5)
    out_zero = dc_zero(torch.randn(4, 256))
    assert torch.allclose(out_zero, torch.full_like(out_zero, 0.55), atol=1e-6), \
        f"Expected 0.55, got {out_zero}"
    print(f"  - Zero-init output = {out_zero[0].item():.4f} (expected 0.55): OK")
    print("✓ test_custom_range passed")


def test_gradient_isolation():
    """decay_controller(img_feat.detach()) must not produce gradients on img_feat."""
    print("\n=== Test 7: Gradient isolation ===")
    dc = DecayController()
    img_feat = torch.randn(4, 256, requires_grad=True)

    # With detach: no gradient should flow back
    out = dc(img_feat.detach())
    loss = out.sum()
    loss.backward()
    assert img_feat.grad is None, "img_feat should have no grad when detached"
    print("  - Detached: img_feat.grad is None: OK")

    # Without detach: gradient should flow back (sanity check)
    img_feat2 = torch.randn(4, 256, requires_grad=True)
    out2 = dc(img_feat2)
    loss2 = out2.sum()
    loss2.backward()
    assert img_feat2.grad is not None, "img_feat should have grad when NOT detached"
    print("  - Non-detached: img_feat.grad exists: OK")
    print("✓ test_gradient_isolation passed")


if __name__ == '__main__':
    print("=" * 60)
    print("DecayController Unit Tests")
    print("=" * 60)

    test_param_count()
    test_output_shape()
    test_output_range()
    test_zero_initialization()
    test_params_vector_roundtrip()
    test_custom_range()
    test_gradient_isolation()

    print("\n" + "=" * 60)
    print("=== All DecayController tests passed! ===")
    print("=" * 60)
