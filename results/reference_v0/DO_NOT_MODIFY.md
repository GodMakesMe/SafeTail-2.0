# Frozen: pre-fix evidence base

These are the results shipped with `SafeTail-2.0-main`, produced **before** any repair in
`plan.md` §7. They are the baseline against which every "numbers moved" claim is made.

**Read-only (plan.md R6).** Do not overwrite, extend or regenerate. New runs go to
`results/<policy>_<k>_<seed>_<gitsha7>/` with a `manifest.json`.

Recomputed from these logs (all modes truncated to the common 14,723 requests,
`total_latency` in ms) — see plan.md §4.2 D-10:

```
mode          p50      p90      p95      p99
safetail    39.88    59.88    75.66   115.64
minload_3   31.88    49.93    54.29    81.67
minprop_2   29.93    46.63    49.08    53.37
rand_3      36.08    57.63    72.36   102.99
```

Note also: `total_latency` here **excludes** `queueing_delay` (mean 68 ms), and the
plotting notebook adds +5 ms to MinProp only. See plan.md D-03, D-11.
