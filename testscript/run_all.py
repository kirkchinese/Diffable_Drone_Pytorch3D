#!/usr/bin/env python3
"""Standalone runner for testscript/test_*.py scripts.

These tests are script-style checks with their own pass counters and device
arguments, so this runner intentionally uses subprocesses instead of pytest.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "testscript"
EXCLUDED_TESTS = {"xbox_controller_test.py"}
SKIP_RE = re.compile(r"(^|\b|\[)SKIP(:|\]|\b)", re.IGNORECASE)
# --cpu-only 模式：这些失败签名 = 测试需要 GPU（记为 SKIP 而非 FAIL，供无 CUDA 环境/CI 用）
CUDA_ABSENT_RE = re.compile(
    r"No CUDA GPUs are available|Torch not compiled with CUDA|"
    r"no NVIDIA driver|CUDA driver|CUDA error", re.IGNORECASE)


def _env_with_project_root():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}{os.pathsep}{existing}"
    return env


def _last_nonempty_line(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _accepts_gpu_arg(path):
    try:
        return "--gpu" in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def run_test(path, args, env):
    cmd = [sys.executable, str(path)]
    if _accepts_gpu_arg(path):
        cmd.extend(["--gpu", str(args.gpu)])

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - started
        combined = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode == 0 and SKIP_RE.search(combined):
            status = "SKIP"
        elif proc.returncode == 0:
            status = "PASS"
        elif getattr(args, "cpu_only", False) and CUDA_ABSENT_RE.search(combined):
            status = "SKIP"   # cpu-only：需 GPU 的测试记 SKIP，不算失败
        else:
            status = "FAIL"
        stdout_note = _last_nonempty_line(proc.stdout)
        stderr_note = _last_nonempty_line(proc.stderr)
        if status in {"PASS", "SKIP"}:
            note = stdout_note or stderr_note
        else:
            note = stderr_note or stdout_note
        return {
            "name": path.name,
            "status": status,
            "code": proc.returncode,
            "time": elapsed,
            "note": note,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        combined = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        note = _last_nonempty_line(combined) or f"timeout after {args.timeout}s"
        return {
            "name": path.name,
            "status": "FAIL",
            "code": "TIMEOUT",
            "time": elapsed,
            "note": note,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU id passed to tests that accept --gpu")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-test timeout in seconds")
    parser.add_argument("--cpu-only", action="store_true",
                        help="隐藏 CUDA 只跑 CPU 安全子集（需 GPU 的测试记 SKIP）；供无 GPU 环境/CI")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    env = _env_with_project_root()
    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    tests = [
        path for path in sorted(TEST_DIR.glob("test_*.py"))
        if path.name not in EXCLUDED_TESTS
    ]

    results = [run_test(path, args, env) for path in tests]

    name_width = max(34, *(len(result["name"]) for result in results))
    table_width = name_width + 44
    print(f"{'TEST':<{name_width}} STATUS  CODE     TIME     NOTE")
    print("-" * table_width)
    for result in results:
        note = result["note"]
        if len(note) > 45:
            note = note[:42] + "..."
        print(
            f"{result['name']:<{name_width}} "
            f"{result['status']:<7} "
            f"{str(result['code']):<8} "
            f"{result['time']:>6.1f}s  "
            f"{note}"
        )

    pass_count = sum(r["status"] == "PASS" for r in results)
    skip_count = sum(r["status"] == "SKIP" for r in results)
    fail_count = sum(r["status"] == "FAIL" for r in results)
    total = len(results)
    print("-" * table_width)
    print(f"SUMMARY: {pass_count} PASS, {skip_count} SKIP, {fail_count} FAIL ({total} total)")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
