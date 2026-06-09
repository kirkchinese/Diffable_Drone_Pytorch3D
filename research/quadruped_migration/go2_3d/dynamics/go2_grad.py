"""Reuse the drone project's temporal gradient-decay (GDecay) for the Go2 SRBD.

Single source of truth: the same custom autograd `GDecay` that the drone dynamics use
(forward identity, backward multiplies the gradient by decay**dt). Importing it here keeps
the quadruped SRBD's BPTT gradient-taming mechanism identical to the drone framework, which
is exactly the cross-platform continuity the migration study is about.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../Diffable_Drone_Pytorch3D
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drone_dynamics import g_decay, GDecay  # noqa: F401,E402
