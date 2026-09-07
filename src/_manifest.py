"""
[SAFETAIL][MAIN][FIX][G5] Run manifest -- "a result without a manifest is not a result".

plan.md section 9.1 requires every run directory to carry a `manifest.json`
recording git SHA + dirty flag, seed, the full `constants` dump, package
versions, host, wall-clock start/end, the SHA-256 of every input CSV/pkl, and
(section 10.3) the `[DEGRADED]` ledger. Gate **G5** (`tools/check_manifest.py`)
enforces it, and a run whose ledger is non-empty is **not publishable**.

Until now none of this existed: `_safetail_log.degraded_report()` was written
for the manifest, `print_constants()` dumped the config to stdout where it was
lost with the terminal scrollback, and every directory under `results/` is
therefore un-provenanced -- there is no machine-checkable record of which code,
seed or environment flags produced any of them.

This module is stdlib-only apart from an optional `importlib.metadata` probe, so
importing it can never be the thing that breaks a run. Every failure inside
`write_manifest` is caught and reported: a manifest must never abort a run that
has already produced its numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Files whose content changes a number. Hashed so a manifest pins the exact
# inputs, not just their names.
_INPUT_GLOBS = ("dataset/*.csv", "models/*/*.pkl")

# Packages whose version can move a number.
_TRACKED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "tensorflow", "keras",
    "matplotlib", "seaborn", "joblib", "threadpoolctl",
)

MANIFEST_NAME = "manifest.json"
_SCHEMA = 1


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _git_info() -> dict:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha or None,
        "sha7": (sha[:7] if sha else None),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        # G5 refuses to bless a run made from a dirty tree: the SHA would not
        # describe the code that actually ran.
        "dirty": bool(status),
        # [SAFETAIL][MAIN][FIX] `_git()` does `.strip()`, which removes the
        # leading space of `git status --porcelain`'s FIRST line only. `l[3:]`
        # is right for every later line (" M plan.md") but ate one character too
        # many from that first one -- every manifest recorded "HANGELOG.md"
        # instead of "CHANGELOG.md". Cosmetic (the `dirty` boolean the gate tests
        # was always correct), but a manifest exists to get provenance right.
        # `l[2:].strip()` handles both the stripped and the unstripped form.
        # Found by the Claude Code verification session, 8 Sep 2026.
        "dirty_files": [l[2:].strip() for l in status.splitlines()][:50] if status else [],
    }


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _input_hashes() -> dict:
    out = {}
    for pattern in _INPUT_GLOBS:
        for p in sorted(_REPO.glob(pattern)):
            try:
                out[p.relative_to(_REPO).as_posix()] = _sha256(p)
            except Exception as exc:  # noqa: BLE001
                out[p.relative_to(_REPO).as_posix()] = f"<unreadable: {exc}>"
    return out


def _packages() -> dict:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return {}
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def _constants_dump(constants) -> dict:
    """
    plan.md 10.5: `print_constants()` dumps the module at startup -- it must go
    into manifest.json, not just stdout.
    """
    out = {}
    for k in dir(constants):
        if k.startswith("_"):
            continue
        v = getattr(constants, k)
        if callable(v) or type(v).__name__ == "module":
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def _safetail_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(("SAFETAIL_", "POLICY", "BASELINE_MODE", "TRAINING_LOG_FOLDER"))}


def write_manifest(log_folder, constants, controller=None, started_at=None,
                   extra=None) -> Path | None:
    """
    Write `<log_folder>/manifest.json`. Returns the path, or None on failure
    (never raises -- a bookkeeping failure must not destroy a finished run).
    """
    try:
        from _safetail_log import degraded_report
        degraded = degraded_report()
    except Exception:
        degraded = {}

    try:
        out_dir = Path(log_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()

        run = {}
        if controller is not None:
            run = {
                "episodes_completed": int(getattr(controller, "current_episode", 0) or 0),
                "dropped_requests": int(getattr(controller, "dropped_requests", 0) or 0),
                "baseline_mode": getattr(controller, "BASELINE_MODE", None),
                "policy": (getattr(getattr(controller, "policy", None), "name", None)
                           or "native"),
                "testing_phase_active": bool(getattr(controller, "testing_phase_active", False)),
                "tau_out_of_band": int(getattr(controller, "_tau_out_of_band", 0) or 0),
            }

        manifest = {
            "schema": _SCHEMA,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "started_at": (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at))
                           if started_at else None),
            "wall_seconds": (round(now - started_at, 3) if started_at else None),
            "git": _git_info(),
            "seed": getattr(constants, "SEED", None),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "packages": _packages(),
            "env": _safetail_env(),
            "constants": _constants_dump(constants),
            "inputs_sha256": _input_hashes(),
            # plan.md 10.3 / gate G5: any count > 0 => NOT PUBLISHABLE.
            "degraded": degraded,
            "publishable": (not degraded) and (not _git_info()["dirty"]),
            "run": run,
        }
        if extra:
            manifest["extra"] = extra

        path = out_dir / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str),
                        encoding="utf-8")
        flag = "PUBLISHABLE" if manifest["publishable"] else "NOT PUBLISHABLE"
        print(f"[SAFETAIL][MAIN][G5] manifest -> {path}  ({flag}"
              f"{'' if not degraded else f'; degraded={degraded}'})")
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[SAFETAIL][MAIN][G5] failed to write manifest: {type(exc).__name__} - {exc}")
        return None
