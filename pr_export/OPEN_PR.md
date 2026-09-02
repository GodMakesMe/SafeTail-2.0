# How to push the `heterogenous` branch and open the PR

I can't do this step from here — there is no `gh` CLI on this machine, the
`SafeTail-2.0-main/` folder on disk is **not a git repo**, and I can't
authenticate to GitHub. Everything is prepared so you can do it in a few
commands.

## What's in this folder

| File | Use |
|---|---|
| `0001-src-correctness-repairs.patch` | `git am` — applies as **one commit** with the full message + author preserved. **Preferred.** |
| `heterogenous.patch` | plain unified diff — `git apply` if `git am` fails on line endings |
| `heterogenous.bundle` | a git bundle of the branch — `git fetch` it directly, no patching |
| `PR_DESCRIPTION.md` | paste into the GitHub "Open a pull request" box (title is at the top) |

Base of the branch = the pristine `SafeTail-2.0-main` snapshot, which is
byte-identical (modulo line endings) to your repo's default branch at the point
the ZIP was taken. If your `main` has moved on since, use the `--3way` forms
below and resolve any conflicts.

---

## Option A — `git am` (cleanest)

```bash
# from a fresh clone of YOUR SafeTail 2.0 repo
git clone git@github.com:<you>/<SafeTail-2.0-repo>.git
cd <SafeTail-2.0-repo>

git checkout -b heterogenous
git am --3way /path/to/pr_export/0001-src-correctness-repairs.patch

git push -u origin heterogenous
```

Then open the PR: GitHub will show a "Compare & pull request" button, or go to
`https://github.com/<you>/<repo>/compare/main...heterogenous`.
Paste the body from `PR_DESCRIPTION.md`.

## Option B — `git apply` (if `am` complains about whitespace/CRLF)

```bash
git checkout -b heterogenous
git apply --3way --whitespace=nowarn /path/to/pr_export/heterogenous.patch
git add -A
git commit -F /path/to/pr_export/commit_message.txt   # see below, or write your own
git push -u origin heterogenous
```

## Option C — fetch the branch from the bundle (no patching at all)

```bash
git fetch /path/to/pr_export/heterogenous.bundle heterogenous:heterogenous-import
git checkout heterogenous-import
# inspect, then:
git checkout -b heterogenous
git push -u origin heterogenous
```
Note: the bundle's base commit is my synthetic "upstream snapshot", so the
branch will have **one extra root-ish commit** unless your `main` tip is the
exact ZIP snapshot. If the diff looks right, `git rebase --onto origin/main
<snapshot-sha> heterogenous` cleans that up. Option A avoids this entirely.

---

## If `git am` / `git apply` fails on line endings

Your GitHub repo stores `src/*.py` with **LF** (that's what the ZIP contains).
The patch is LF-based, so this should just work. If your local clone has
`core.autocrlf=true` smudging them to CRLF:

```bash
git -c core.autocrlf=false am --3way 0001-src-correctness-repairs.patch
```

## Commit message (for Option B)

The full message is the body of `0001-src-correctness-repairs.patch` (everything
between the `Subject:` line and the `---` before the diff). Copy it into
`commit_message.txt`, or just reuse the first line:

```
src: correctness repairs to the heterogeneous scheduler (D-01 .. D-35)
```
