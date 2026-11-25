# Retained Metal path isolation v4

Measured source `faad1de0acd7c20c24d187f89d5d93cdd91d3c8b`, [original workflow](https://github.com/mitukx/VARE/actions/runs/45604930140).
Both registered Metal arms completed. The initial run stopped in a CPU
analytic control before loading weights; that failed workflow is retained
in the manifest and must not count as a numerical result.

| Precision | Mode | Max plain/cached vs differentiated (nat) | Passing contexts | Complete path criterion |
| --- | --- | ---: | ---: | --- |
| bf16 | stock_selected | 0.3551506996154785 | 0/2 | NO-GO |
| bf16 | stock_full | 0.3551506996154785 | 0/2 | NO-GO |
| bf16 | explicit_selected | 0 | 2/2 | PASS |
| bf16 | explicit_full | 0 | 2/2 | PASS |
| fp32 | stock_selected | 0.00001621246337890625 | 0/2 | NO-GO |
| fp32 | stock_full | 0.000029087066650390625 | 0/2 | NO-GO |
| fp32 | explicit_selected | 0.0000152587890625 | 0/2 | NO-GO |
| fp32 | explicit_full | 0.0000019073486328125 | 1/2 | NO-GO |

The isolated BF16 stock attention forward/VJP maximum output discrepancies
were 0.0078125 and 0.015625 for lengths 16 and 48; explicit attention gave
zero for both. These are output units, not nats. The BF16 fixed-target
real-weight explicit paths meet the score threshold in both contexts.
The FP32 explicit-full path reduces the worst difference to
1.9073486328125e-6 nat but misses the unchanged 1e-6 threshold on the long
context; no FP32 path passes both contexts.

All measured gradients were finite and nonzero; all parameter bytes were
unchanged. Explicit attention changes the plain score relative to stock
attention. BF16 explicit-selected also has a 0.125 maximum cached-vs-traced
logit discrepancy despite exact agreement on these two selected targets;
it is not a distribution-wide identity proof. These data localize a
plausible mechanism without proving all discrepancies come from one kernel.
No generated tokens, optimizer update, or task/held-out evaluation occurred.

Original [BF16 ZIP](bf16.zip), [FP32 ZIP](fp32.zip), [manifest](manifest.json).
Run `python -m pytest -q tests/test_grpo_metal_path_evidence.py` to verify
the original bytes, scope, coverage and exact reported scores. The next
experiment must prospectively test actual sampled multi-action groups with
a consistently scoped explicit attention/full projection path, preserve
sampled denominators, and compare compute/memory cost. No production
scoring change or training authorization follows from this diagnostic.
