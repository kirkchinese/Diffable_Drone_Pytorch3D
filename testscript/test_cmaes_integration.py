#!/usr/bin/env python3
"""
Integration tests for CMA-ES + DecayController + dynamics pipeline.
Standalone script with assert + print (no pytest).
"""

import sys
sys.path.insert(0, '/home/misaka/Diffable_Drone_Pytorch3D')

import torch
import cma
from model import DecayController
from drone_dynamics import simulate_position_step


def test_cma_ask_tell():
    """Test CMA-ES ask/tell cycle works correctly."""
    print("\n=== Test 1: CMA-ES ask/tell cycle ===")
    
    dc = DecayController()
    x0 = dc.get_params_vector().cpu().numpy()
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.5, {'popsize': 5, 'verbose': -1})
    solutions = es.ask()
    
    assert len(solutions) == 5, f"Expected 5 solutions, got {len(solutions)}"
    
    fitnesses = [float(i) for i in range(5)]
    es.tell(solutions, fitnesses)
    
    # Verify es.result.xbest is not None after tell
    assert es.result.xbest is not None, "es.result.xbest should not be None after tell"
    
    print("  - CMA-ES asked 5 solutions: OK")
    print("  - CMA-ES tell completed: OK")
    print("  - es.result.xbest available: OK")
    print("✓ test_cma_ask_tell passed")


def test_cma_to_decay_controller_transfer():
    """Test CMA-ES → DecayController parameter transfer."""
    print("\n=== Test 2: CMA-ES → DecayController parameter transfer ===")
    
    dc = DecayController()
    
    # Get initial output
    img_feat = torch.randn(4, 256)
    output_before = dc(img_feat)
    
    print(f"  - Initial output range: [{output_before.min().item():.4f}, {output_before.max().item():.4f}]")
    
    # Take a CMA-ES solution and set it on controller
    x0 = dc.get_params_vector().cpu().numpy()
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.5, {'popsize': 5, 'verbose': -1})
    solutions = es.ask()
    sol = solutions[0]
    
    # Set params from CMA-ES solution
    dc.set_params_vector(torch.tensor(sol, dtype=torch.float32))
    
    # Get output after CMA-ES params
    output_after = dc(torch.randn(4, 256))
    
    print(f"  - After CMA-ES params range: [{output_after.min().item():.4f}, {output_after.max().item():.4f}]")
    
    # Verify output is still in [0.2, 1.0] even with CMA-ES params
    assert output_after.min() >= 0.2 - 1e-6, f"Output {output_after.min()} below 0.2"
    assert output_after.max() <= 1.0 + 1e-6, f"Output {output_after.max()} above 1.0"
    
    print("  - Output in [0.2, 1.0] range: OK")
    print("✓ test_cma_to_decay_controller_transfer passed")


def test_tensor_grad_decay_in_simulate():
    """Test tensor grad_decay in simulate_position_step."""
    print("\n=== Test 3: Tensor grad_decay in simulate_position_step ===")
    
    B = 4
    p = torch.randn(B, 3)
    v = torch.randn(B, 3)
    a = torch.zeros(B, 3)
    act_curr = torch.zeros(B, 3)
    act_cmd = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    
    # Scalar mode (backward compatibility)
    p1, v1, _, _ = simulate_position_step(
        p, v, a, act_curr, act_cmd, R, dt=1/15, grad_decay=0.4
    )
    assert p1.shape == (B, 3), f"Scalar mode: expected shape (4, 3), got {p1.shape}"
    print("  - Scalar mode shape correct: OK")
    
    # Tensor mode (new per-sample decay)
    grad_decay_t = torch.tensor([0.3, 0.5, 0.7, 0.9])
    p2, v2, _, _ = simulate_position_step(
        p, v, a, act_curr, act_cmd, R, dt=1/15, grad_decay=grad_decay_t
    )
    assert p2.shape == (B, 3), f"Tensor mode: expected shape (4, 3), got {p2.shape}"
    print("  - Tensor mode shape correct: OK")
    
    # GDecay is identity in forward pass, so p1 and p2 should be equal in forward
    # (grad_decay only affects backward pass)
    assert torch.allclose(p1, p2, atol=1e-6), "Forward pass should be identity regardless of grad_decay"
    print("  - Forward pass identical (GDecay identity): OK")
    
    print("✓ test_tensor_grad_decay_in_simulate passed")


def compute_fitness(p_history, p_start, p_target, vec_to_obj_history, margin):
    """
    Standalone fitness computation for testing.
    Matches the logic used in CMA-ES optimization.
    """
    total_dist = (p_target - p_start).norm(dim=-1)
    final_dist = (p_target - p_history[-1]).norm(dim=-1)
    progress = (1.0 - final_dist / (total_dist + 1e-6)).clamp(0, 1)
    
    distance = vec_to_obj_history.norm(dim=-1) - margin
    collision = (distance < 0).any(dim=0).float()
    
    avg_speed = (p_history[1:] - p_history[:-1]).norm(dim=-1).mean(0) if p_history.shape[0] > 1 else torch.zeros(p_history.shape[1])
    
    per_sample = progress * (1.0 - collision) * (1.0 + 0.1 * avg_speed)
    return per_sample.mean().item()


def test_fitness_function():
    """Test fitness function correctness."""
    print("\n=== Test 4: Fitness function correctness ===")
    
    B = 4
    
    # Test 1: collision → fitness ≈ 0
    p_start = torch.zeros(B, 3)
    p_target = torch.tensor([10.0, 0.0, 0.0]).unsqueeze(0).expand(B, -1)
    # Simulate trajectory that ends at target but collides
    p_history = torch.stack([p_start, p_target * 0.5, p_target * 0.8, p_target], dim=0)
    # Points very close to origin (collision with object at origin)
    vec_to_obj_history = torch.stack([
        torch.tensor([0.1, 0.0, 0.0]),
        torch.tensor([0.05, 0.0, 0.0]),
        torch.tensor([0.02, 0.0, 0.0]),
        torch.tensor([0.01, 0.0, 0.0]),
    ], dim=0).unsqueeze(1).expand(-1, B, -1)
    margin = 0.5
    
    fitness_collision = compute_fitness(p_history, p_start, p_target, vec_to_obj_history, margin)
    assert fitness_collision < 0.1, f"Collision should give low fitness, got {fitness_collision}"
    print(f"  - Collision case fitness: {fitness_collision:.4f} (expected < 0.1): OK")
    
    # Test 2: no progress (start==end) → fitness ≈ 0
    p_history_no_progress = torch.stack([p_start, p_start, p_start, p_start], dim=0)
    vec_to_obj_no_collision = torch.stack([
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
    ], dim=0).unsqueeze(1).expand(-1, B, -1)
    
    fitness_no_progress = compute_fitness(p_history_no_progress, p_start, p_target, vec_to_obj_no_collision, margin)
    assert fitness_no_progress < 0.1, f"No progress should give low fitness, got {fitness_no_progress}"
    print(f"  - No progress case fitness: {fitness_no_progress:.4f} (expected < 0.1): OK")
    
    # Test 3: full progress, no collision → fitness > 0
    p_history_full = torch.stack([p_start, p_start + p_target * 0.33, p_start + p_target * 0.66, p_target], dim=0)
    vec_to_obj_no_collision = torch.stack([
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([4.0, 0.0, 0.0]),
        torch.tensor([3.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
    ], dim=0).unsqueeze(1).expand(-1, B, -1)
    
    fitness_full = compute_fitness(p_history_full, p_start, p_target, vec_to_obj_no_collision, margin)
    assert fitness_full > 0.5, f"Full progress should give high fitness, got {fitness_full}"
    print(f"  - Full progress case fitness: {fitness_full:.4f} (expected > 0.5): OK")
    
    # Test 4: partial progress, no collision → 0 < fitness < full_progress_fitness
    p_history_partial = torch.stack([p_start, p_start + p_target * 0.2, p_start + p_target * 0.4, p_start + p_target * 0.5], dim=0)
    
    fitness_partial = compute_fitness(p_history_partial, p_start, p_target, vec_to_obj_no_collision, margin)
    assert 0 < fitness_partial < fitness_full, f"Partial progress fitness {fitness_partial} should be between 0 and full {fitness_full}"
    print(f"  - Partial progress case fitness: {fitness_partial:.4f} (0 < partial < full): OK")
    
    print("✓ test_fitness_function passed")


def test_end_to_end_pipeline():
    """Test end-to-end mini pipeline: DecayController → simulate_position_step."""
    print("\n=== Test 5: End-to-end mini pipeline ===")
    
    # Setup
    B = 4
    dc = DecayController()
    
    # Simulate CNN features (detached from main network)
    img_feat = torch.randn(B, 256)
    
    # Get decay from DecayController
    grad_decay = dc(img_feat)  # Shape: (B,), in [0.2, 1.0]
    
    print(f"  - DecayController output: {grad_decay}")
    print(f"  - Decay range: [{grad_decay.min().item():.4f}, {grad_decay.max().item():.4f}]")
    
    # Use decay in simulate_position_step as tensor grad_decay
    p = torch.randn(B, 3)
    v = torch.randn(B, 3)
    a = torch.zeros(B, 3)
    act_curr = torch.zeros(B, 3)
    act_cmd = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    
    # This should work without errors
    p_next, v_next, a_next, act_next = simulate_position_step(
        p, v, a, act_curr, act_cmd, R, dt=1/15, grad_decay=grad_decay
    )
    
    assert p_next.shape == (B, 3), f"Expected p_next shape (4, 3), got {p_next.shape}"
    assert v_next.shape == (B, 3), f"Expected v_next shape (4, 3), got {v_next.shape}"
    assert a_next.shape == (B, 3), f"Expected a_next shape (4, 3), got {a_next.shape}"
    assert act_next.shape == (B, 3), f"Expected act_next shape (4, 3), got {act_next.shape}"
    
    print("  - Output shapes correct: OK")
    
    # Verify outputs are valid (not NaN or Inf)
    assert not torch.isnan(p_next).any(), "p_next contains NaN"
    assert not torch.isinf(p_next).any(), "p_next contains Inf"
    assert not torch.isnan(v_next).any(), "v_next contains NaN"
    assert not torch.isinf(v_next).any(), "v_next contains Inf"
    
    print("  - No NaN/Inf in outputs: OK")
    print("✓ test_end_to_end_pipeline passed")


if __name__ == '__main__':
    print("=" * 60)
    print("CMA-ES Integration Tests")
    print("=" * 60)
    
    test_cma_ask_tell()
    test_cma_to_decay_controller_transfer()
    test_tensor_grad_decay_in_simulate()
    test_fitness_function()
    test_end_to_end_pipeline()
    
    print("\n" + "=" * 60)
    print("=== All CMA-ES integration tests passed! ===")
    print("=" * 60)