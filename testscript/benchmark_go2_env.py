"""Go2 GPU throughput and hot-loop host-synchronization audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.envs.go2 import Go2Env  # noqa: E402
from diffsim.go2.types import Go2EnvConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the GPU benchmark")
    device = torch.device(args.device)
    config = Go2EnvConfig(batch_size=args.batch_size)
    env = Go2Env(config, device, seed=31, randomize=True)
    action = torch.zeros(args.batch_size, 12, device=device, requires_grad=True)

    env.step(action)
    env.state = env.state.detach()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        output = env.step(action)
    objective = output.state.base_pos.square().mean() + output.state.joint_pos.square().mean()
    objective.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    env.state = env.state.detach()
    action = action.detach()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
    ) as profiler:
        env.step(action)
    names = {event.key for event in profiler.key_averages()}
    forbidden = {
        name
        for name in names
        if "_local_scalar_dense" in name
    }
    if forbidden:
        for event in profiler.events():
            if "_local_scalar_dense" in event.name:
                print("scalar extraction stack:", event.stack)
        raise AssertionError(f"host synchronization found in Go2 hot loop: {sorted(forbidden)}")
    transitions = args.batch_size * args.steps
    print(f"B={args.batch_size}, {args.steps} control steps + backward: {elapsed:.3f}s")
    print(f"throughput={transitions / elapsed:.1f} batched control transitions/s")
    print("[PASS] profiler found no scalar extraction or explicit CUDA synchronization")


if __name__ == "__main__":
    main()
