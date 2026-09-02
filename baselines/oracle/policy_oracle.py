"""
[SAFETAIL][POLICY][M-01] Oracle / optimal baseline.

plan.md section 8.7 + M-01. The oracle dispatches to *every* server and the
harness records min(realised latency). It is the optimality-gap reference
(ST Table V analogue, figure F4).

It is intentionally trivial: ~1 line of logic. Its real job is to be the first
policy through the seam, proving src/policy_registry.py + the controller branch +
G4 all work end to end, independently of the much larger SafeTail-1.0 port.
"""
from __future__ import annotations

from typing import Sequence

from policy_registry import BasePolicy, PolicyContext


class OraclePolicy(BasePolicy):
    name = "oracle"

    def select(self, ctx: PolicyContext) -> Sequence[int]:
        # Schedule to all beta servers; the environment takes the min over them.
        return list(range(ctx.beta))


def factory() -> OraclePolicy:
    return OraclePolicy()
