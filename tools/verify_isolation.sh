#!/usr/bin/env bash
# [SAFETAIL][SEAM] Gate G4 -- baselines/ is a deletable unit (plan.md R3, section 11).
#
#   (b) src/ has no CODE dependency on baselines/  (comments/docstrings are fine)   HARD
#   (c) baselines/ never imports _spec_source/                                      HARD
#   (d) no sys.path manipulation whose target could be OUTSIDE heterogenous/        HARD
#   (a) with baselines/ hidden, src/ core still imports and native --smoke runs     HARD / SKIP(no TF)
#       + POLICY=safetail_v1 with baselines/ hidden must fail loudly                HARD
#
# Exits non-zero on any HARD failure. Smoke SKIPs (no tensorflow) do not fail the
# gate unless SAFETAIL_REQUIRE_SMOKE=1.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="${SAFETAIL_PY:-$REPO/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"

fail=0
note() { printf '[SAFETAIL][SEAM][G4] %s\n' "$*"; }
hard() { printf '[SAFETAIL][SEAM][G4][FAIL] %s\n' "$*"; fail=1; }
skip() { printf '[SAFETAIL][SEAM][G4][skip] %s\n' "$*"; [ "${SAFETAIL_REQUIRE_SMOKE:-0}" = "1" ] && fail=1; }
ok()   { printf '[SAFETAIL][SEAM][G4][ok] %s\n' "$*"; }

HIDDEN="$REPO/baselines.__g4_hidden__"
# recover from a previously interrupted run
[ -d "$HIDDEN" ] && [ ! -d "$REPO/baselines" ] && mv "$HIDDEN" "$REPO/baselines"

# --- (b) code-level scan, comments/strings stripped via tokenize --------------
"$PY" - <<'PYEOF'
import io, sys, tokenize, pathlib
bad = []
for f in pathlib.Path("src").rglob("*.py"):
    try:
        toks = tokenize.tokenize(io.BytesIO(f.read_bytes()).readline)
        for tok in toks:
            if tok.type == tokenize.NAME and tok.string == "baselines":
                bad.append(f"{f}:{tok.start[0]} -> NAME 'baselines' in code")
            if tok.type == tokenize.STRING and "baselines" in tok.string.lower() \
               and ("/" in tok.string or "\\" in tok.string):
                bad.append(f"{f}:{tok.start[0]} -> path-like string mentions baselines")
    except Exception as e:
        bad.append(f"{f}: tokenize failed: {e}")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("clean")
PYEOF
[ $? -eq 0 ] && ok "(b) src/ has no code-level reference to baselines/" \
             || hard "(b) src/ references baselines/ in code (see above)"

# --- (c) _spec_source imports ----------------------------------------------
if grep -rniIn --include='*.py' -E '^[[:space:]]*(import|from)[[:space:]].*_spec_source' baselines/ ; then
  hard "(c) baselines/ imports from _spec_source/"
else
  ok "(c) baselines/ never imports _spec_source/"
fi

# --- (d) sys.path targets -------------------------------------------------------
# flag a sys.path.insert/append line only if it carries a string literal that
# could escape the repo: '..', an absolute posix '/home|/Users', a drive 'X:\',
# or the name 'SafeTail'. __file__-derived / bare-name inserts are fine.
dpaths=$(grep -rniIn --include='*.py' -E 'sys\.path\.(insert|append)' src/ baselines/ \
         | grep -E "\.\.|/home/|/Users/|[A-Za-z]:\\\\|SafeTail")
if [ -n "$dpaths" ]; then
  printf '%s\n' "$dpaths"
  hard "(d) sys.path manipulation with a suspicious target literal"
else
  ok "(d) no sys.path insert/append targets a path outside heterogenous/"
fi
if grep -rniIn --include='*.py' -E "SafeTail[\\\\/](SafeTail|aman|SafeTail-1)" src/ baselines/ ; then
  hard "(d/R1) reference to the SafeTail 1.0 upstream tree"
else
  ok "(d/R1) no reference to ..\\SafeTail upstream"
fi

# --- (a) hide baselines/, test core, restore ---------------------------------
restore() { [ -d "$HIDDEN" ] && mv "$HIDDEN" "$REPO/baselines"; }
trap restore EXIT
mv "$REPO/baselines" "$HIDDEN" || { hard "(a) could not hide baselines/"; note FAIL; exit 1; }

"$PY" -c "import sys; sys.path.insert(0,'src'); import policy_registry, constants, user, servers; print('base ok')" \
  && ok "(a) policy_registry+constants+user+servers import without baselines/" \
  || hard "(a) core stdlib-side imports broke without baselines/"

if "$PY" -c "import tensorflow" 2>/dev/null ; then
  "$PY" -c "import sys; sys.path.insert(0,'src'); import agent, controller, main; print('tf ok')" \
    && ok "(a) agent+controller+main import without baselines/" \
    || hard "(a) agent/controller/main import broke without baselines/"
  for mode in safetail minload_2 ; do
    if POLICY=native BASELINE_MODE="$mode" SAFETAIL_SMOKE=1 "$PY" src/main.py --smoke >"$REPO/tools/out/g4_smoke_$mode.log" 2>&1 ; then
      ok "(a) native BASELINE_MODE=$mode --smoke ran without baselines/"
    else
      hard "(a) native BASELINE_MODE=$mode --smoke FAILED (tools/out/g4_smoke_$mode.log)"
    fi
  done
  if POLICY=safetail_v1 SAFETAIL_SMOKE=1 "$PY" src/main.py --smoke >"$REPO/tools/out/g4_smoke_badpolicy.log" 2>&1 ; then
    hard "(a) POLICY=safetail_v1 without baselines/ exited 0 -- must fail loudly"
  else
    ok "(a) POLICY=safetail_v1 without baselines/ fails loudly (KeyError listing known names)"
  fi
else
  skip "(a) tensorflow unavailable -- agent/controller/main + native smoke not exercised"
fi

restore; trap - EXIT
[ "$fail" -eq 0 ] && { note PASS; exit 0; } || { note FAIL; exit 1; }
