# Native sampler alignment v2 — retained CPU evidence

The [frozen protocol](../../protocols/grpo_pretrained_sampling_alignment_v2.json)
at exact source commit `45dfb554b3e4d203a6af9c622411eba4f103a010` completed both CPU arms.
The [GitHub Actions run](https://github.com/mitukx/VARE/actions/runs/45596218464)
and all 24 check-run entries on the measured head succeeded. The archived
ZIPs are the original workflow artifacts, not recreated files.

| Precision | Tokens | Legacy plain-forward above 1e-6 | Candidate differentiated above 1e-6 | Legacy maximum (nat) | Candidate maximum (nat) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 16 | 16 | 0 | 0.013680458068847656 | 0 |
| FP32 | 16 | 7 | 0 | 0.000002384185791015625 | 0 |

Each arm used two neutral contexts, two seeds, two actions per member and
at most two completion tokens per action. Comparisons are within-arm on
identical recorded token sequences; the legacy control is ordinary
forward while the candidate score is from the differentiated forward.
The candidate ordinary and differentiated scores also agreed exactly in
these runs. Every group passed the unchanged strict guard hypothetically
if selected for an informative update. All rewards were dummy zero.
Pre/post full parameter digests match; no optimizer, grader, task data or
held-out evaluation was used.

- [Manifest and workflow markers](manifest.json)
- [BF16 original archive](bf16.zip)
- [FP32 original archive](fp32.zip)

Use `python -m pytest -q tests/test_grpo_native_alignment_evidence.py`
to verify the pinned archive and contained result hashes, protocol,
parameter identity, group coverage, legacy per-token arithmetic summary
and candidate score summaries without loading a model. The evidence test
does not independently rerun the differentiated computation. The raw
candidate metrics were produced by the audited workflow; they are not a
separate gradient proof.

These measurements do not prove real-model learning, Metal correctness,
quantized CPU/GPU alignment, nonzero-lag updates, long-context robustness,
or general model capability. V1's q4 timeout is retained as incomplete.
A new model or backend needs its own prospective diagnostic and resource
budget. The task-readiness NO-GO remains unchanged.
