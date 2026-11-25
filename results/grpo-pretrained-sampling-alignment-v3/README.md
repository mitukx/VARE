# Retained v3 quantized and Metal alignment evidence

Measured source: `639d7415664033dfba20bfefdc083d0324882ae3`. All 24 check-run entries succeeded.
A successful diagnostic workflow records an outcome; it is not an assertion
that every arm passed. [Original workflow](https://github.com/mitukx/VARE/actions/runs/45599188226).

| Arm | Sampled vs differentiated maximum (nat) | Above 1e-6 | Ordinary vs differentiated maximum (nat) | Result |
| --- | ---: | ---: | ---: | --- |
| Linux CPU q4 | 0 | 0/4 | 0 | PASS |
| Apple Silicon Metal BF16 | 0.07882946729660034 | 4/4 | 0.13482439517974854 | NO_GO_NUMERICAL_MISMATCH |
| Apple Silicon Metal q4 | 0.13085639476776123 | 4/4 | 0.13618206977844238 | NO_GO_NUMERICAL_MISMATCH |

The CPU q4 arm finished in 380.079544935 s with 211 quantized modules.
Both Metal arms used actual registered GPU devices; no CPU fallback occurred.
Full pre/post parameter digests matched in every arm. These are short-context
read-only numerical measurements (four generated tokens per arm), not
performance comparisons or learning results. The v1 q4 timeout is unchanged.
The v2 CPU FP32/BF16 results concern a different cohort.

Original archives: [cpu-q4.zip](cpu-q4.zip), [metal-bf16.zip](metal-bf16.zip),
[metal-q4.zip](metal-q4.zip). Each contains the unaltered environment listing,
backend preflight and result JSON. [Manifest](manifest.json) pins their
identities. Run `python -m pytest -q tests/test_grpo_backend_alignment_evidence.py`
for an offline byte and outcome consistency check; this does not rerun MLX
or independently reconstruct all differentiated logprobs from the raw result.

The Metal discrepancies remain blockers. Do not relax the guard, replace
recorded denominators, or authorize task/model training. Future measurements
need a separate frozen protocol; do not automatically repeat this cohort.
