# Push + final verification — run these on your machine

Two commits are made locally. **The push could not be done from the Cowork
sandbox** — it has no GitHub credentials, and asking you for a token would be
the wrong way to solve that. Your machine already has them, so run this there
(or hand this file to the Claude Code session).

```powershell
cd E:\Project\IP_Arani\heterogenous
git log --oneline -3        # expect 2aae6ab, 276bf20, fd6cc64
```

## 1. Push

Local `master` tracks the fork's **`heterogenous-workspace`** branch (both were
at `fd6cc64` before these commits — that is the full working repo; the
`heterogenous` branch is the `src/`-only one).

```powershell
git push fork master:heterogenous-workspace
```

## 2. Then prove the provenance gate actually works

This is the payoff for G5, and it can only be shown after a commit — every run
made before one is stamped `NOT PUBLISHABLE` because the tree was dirty.

```powershell
$env:SAFETAIL_SEED="0"
.\.venv\Scripts\python.exe src\main.py --smoke
.\.venv\Scripts\python.exe tools\check_manifest.py --dir tools\out\smoke_logs
```

**Expect it to flip to `ok` / `PUBLISHABLE`.** Before the commit it correctly
failed with `git tree was DIRTY at run time`. If it still fails, the message
says why.

## 3. Full re-verify (optional, ~1 min)

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # expect 48 passed
.\.venv\Scripts\python.exe tools\verify_env.py            # G0
.\.venv\Scripts\python.exe tools\verify_heterogeneity.py  # G1
.\.venv\Scripts\python.exe tools\audit_replay.py          # G2
.\.venv\Scripts\python.exe tools\audit_reward.py          # G3
bash tools\verify_isolation.sh                            # G4
.\.venv\Scripts\python.exe tools\check_manifest.py        # G5
.\.venv\Scripts\python.exe tools\verify_figures.py        # G6
.\.venv\Scripts\python.exe tools\verify_types.py          # G7
.\.venv\Scripts\python.exe tools\verify_dataset.py        # G8 (advisory)
```

---

## What is deliberately NOT in the push

`audit/dossier_uttam/` — Uttam's defect dossier and its OCR transcript.

**This fork is public** (anonymous `git ls-remote` succeeds), so committing it
would publish a colleague's document under your account. That is your call and
his, not something to do by default. The files are still on disk; only the
tracking is suppressed, via an explanatory block in `.gitignore`.

**Our analysis of it is committed and is public-safe** — `FIXES_REPORT` §4b and
`plan.md` §4.4b carry every finding (D-38…D-45) in our own words with our own
reproductions.

If Uttam is happy for it to be public:

```powershell
git add -f audit/dossier_uttam/
git commit -m "audit: add the external defect dossier (Uttam, 3 Sep 2026)"
```

---

## Immediately after the push — the things that are now owed

Both are consequences of what was just committed, and neither is optional if
any number is going to be quoted:

1. **Re-run the SafeTail 2.0 side.** `D-38` changed the episodic reward and
   `D-44` added 2 elements to the state vector. Baselines, Oracle and SafeTail
   1.0 are unaffected — none uses that reward or that state builder — so the
   comparison structure survives, but every 2.0 row is stale.
2. **Record the sampling boundary.** `TRACE_SAMPLING` now defaults to `"sample"`
   and `server5.csv` has six multi-row contention strings, so **runs after
   `276bf20` are not comparable to runs before it** unless
   `SAFETAIL_TRACE_SAMPLING=first` is set. `SAFETAIL_LEGACY_ENV=1` pins `first`,
   so `results/reference_v0` reproduction is safe.

Everything still open, sorted by who decides it, is in
`FIXES_REPORT_SafeTail2.0_to_heterogenous.md` **§8b** — six advisor decisions,
six of yours, eight technical items, five unresolved doubts.

---

*Prepared 8 September 2026. Commits `276bf20` and `2aae6ab`; `baselines/`,
`results/` and `results/reference_v0/` untouched by both.*
