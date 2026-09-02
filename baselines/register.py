"""
[SAFETAIL][SEAM] The ONE file in baselines/ that touches src/policy_registry.

Importing this module registers every baseline policy into the seam's registry.
`baselines/run_baseline.py` imports it before calling src/main.main().

src/ never imports this. The dependency arrow points only this way:

        baselines/register.py  ------>  src/policy_registry.py

Deleting baselines/ removes this file; src/ keeps working because
constants.POLICY defaults to "native" and the controller's native path never
consults the registry.
"""
from __future__ import annotations

import policy_registry  # from src/ -- the only permitted src import in baselines/

_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    # --- M-01 oracle -------------------------------------------------------- #
    from oracle.policy_oracle import factory as oracle_factory
    policy_registry.register("oracle", oracle_factory)

    # --- M-02 SafeTail 1.0 port ------------------------------------------------ #
    # Registered only once the port exists; keep the import guarded so a partial
    # checkout (oracle done, v1 not yet) still runs.
    try:
        from safetail_v1.policy_v1 import factory as v1_factory
    except Exception:  # noqa: BLE001 -- port not implemented yet
        v1_factory = None
    if v1_factory is not None:
        policy_registry.register("safetail_v1", v1_factory)

    _REGISTERED = True


# Register on import for the common case.
register_all()
