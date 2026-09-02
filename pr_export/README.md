# `pr_export/` — the SafeTail 2.0 `heterogenous` PR, ready to push

A real GitHub PR could not be opened from the session that produced this
(`SafeTail-2.0-main/` on disk was not a git repo, no `gh` CLI, no GitHub auth).
Everything needed to open it yourself is here.

## Scope

`src/` **correctness fixes only** (`D-01 … D-35`) + `requirements.txt`,
`README.md`, `run_all_baselines.sh`. One commit. ~1337 insertions / 204
deletions across 7 edited `src/` files, 5 new `src/` modules, and the 15
per-server regressor wrappers moved to `src/_legacy_regressor_wrappers/`.
**No new top-level directories.** Backwards-compatible defaults.

## Files

| File | Use |
|---|---|
| `PR_DESCRIPTION.md` | paste into the GitHub PR box (title on the first line) |
| `OPEN_PR.md` | step-by-step: clone, branch, apply, push, open PR |
| `0001-src-correctness-repairs.patch` | `git am --3way` — one commit, message + author preserved (**preferred**) |
| `heterogenous.patch` | plain `git diff` — `git apply --3way` fallback |
| `heterogenous.bundle` | `git fetch <bundle> heterogenous:…` — no patching |
| `commit_message.txt` | the full commit message, if you apply by hand |

## Also prepared

`E:\Project\IP_Arani\SafeTail-2.0-main\SafeTail-2.0-main\` is now a local git
repo with:

* `main` — the upstream ZIP snapshot
* `heterogenous` — the fixes applied

If that snapshot equals your GitHub repo's default-branch tip (it should — the
folder is a `…-main.zip` extract), you can push straight from there:

```bash
cd E:\Project\IP_Arani\SafeTail-2.0-main\SafeTail-2.0-main
git remote add origin <your-repo-url>
git push origin heterogenous
```

Otherwise use `OPEN_PR.md` Option A (fresh clone + `git am`).

## Verify the branch

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
SAFETAIL_SMOKE=1 SAFETAIL_SEED=0 python src/main.py --smoke
```

Smoke-tested green on this machine (`exit 0`, no traceback, per-server
computation delays 20–30× apart).
