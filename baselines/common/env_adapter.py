"""
[SAFETAIL][POLICY] Read-only helpers for interpreting a PolicyContext.

`PolicyContext` (defined in src/policy_registry.py) is already the read-only
view; this module only adds small derived conveniences so individual baselines
don't each re-derive them. No mutation, no import of src/ internals beyond the
seam module.

plan.md section 8.4.
"""
from __future__ import annotations

from typing import Sequence

from policy_registry import PolicyContext  # the seam module (always importable)


def free_server_indices(ctx: PolicyContext) -> list[int]:
    """0-based indices of servers with at least one free slot."""
    return [i for i, f in enumerate(ctx.free_slots) if f != -1]


def all_server_indices(ctx: PolicyContext) -> list[int]:
    return list(range(ctx.beta))


def est_total_delay(ctx: PolicyContext, i: int) -> float:
    return float(ctx.est_delay[i])


def est_propagation(ctx: PolicyContext, i: int) -> float:
    return float(ctx.est_components[i][1])


def est_computation(ctx: PolicyContext, i: int) -> float:
    return float(ctx.est_components[i][0])


def active_load(ctx: PolicyContext, i: int) -> int:
    """
    Concurrent-request count on server i, derived from free_slots.
    free_slots[i] == -1 means full; otherwise it is the count of *free* slots,
    so active = MAX_CONCURRENT - free. We don't know MAX_CONCURRENT here without
    coupling, so callers that need an absolute load should read
    ctx.server_dynamic[i]. This returns a monotone proxy: higher == busier.
    """
    f = ctx.free_slots[i]
    if f == -1:
        return 10 ** 6  # sentinel: treat "full" as maximally loaded
    return -int(f)


def pick_min(indices: Sequence[int], key) -> int:
    return min(indices, key=key)
