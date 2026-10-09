# RVL GRPO weight-decay configuration with legacy compatibility — v2

**Result: the compatibility-preserving candidate passed the frozen CPU criteria.** The old and candidate trainers use the same effective `weight_decay=0.01` by default and produce matching no-signal updates. The candidate also allows an explicit `weight_decay=0.0`, which makes a zero-gradient step an exact identity. This is a small API/correctness improvement, not a new RL method or model-quality result.

## Why this follow-up exists

The earlier v1 finding showed that pinned RVL omitted the optimizer setting and silently inherited PyTorch AdamW's `0.01`. Its first candidate changed the default to `0.0`; that passed the identity-update fixture but changed normal training behavior. We therefore tested a compatibility-preserving design: expose the value at the end of `HFTTrainerConfig`, keep `0.01` as the default, and let users choose zero explicitly.

## Frozen protocol

The lock was committed before confirmatory base/candidate runs at VARE commit `9e8864d`. It pins the RVL base revision, candidate source/test hashes and commit, Python/PyTorch/Transformers versions, the tiny one-layer GPT-2 fixture, seeds `17, 23, 29`, all rewards, and success/failure criteria. The protocol is [`rvl_grpo_weight_decay_compat_v2.lock.json`](../protocols/rvl_grpo_weight_decay_compat_v2.lock.json), canonical SHA-256 `5e8cde24ace26bb1726dc75a15eccea6d56cbea170741899ebf43d9600876c98` (raw lock-file SHA `2c71766e3697c1728a05d7256d9bf4900b90dd46f1fdc4424171b898c412f394`).

The compared conditions are the pinned base default, candidate default, candidate explicit zero, candidate explicit historical `0.01`, and a mixed-reward update control. Every constant-reward group has four same-prompt responses and rewards `[1,1,1,1]`; the mixed control uses `[0,0,1,1]`. Runs are CPU-only, with no paid service or GPU.

## Results

| Condition | Effective decay | Zero-signal changed parameter tensors | Max parameter delta | Max decay-formula error |
| --- | ---: | ---: | ---: | ---: |
| Base default | 0.01 | 9 | `1.001358e-5` | 0 |
| Candidate default | 0.01 | 9 | `1.001358e-5` | 0 |
| Candidate explicit `0.01` | 0.01 | 9 | `1.001358e-5` | 0 |
| Candidate explicit `0.0` | 0.0 | 0 | 0 | 0 |

For all three seeds, base default, candidate default, and candidate explicit `0.01` had matching per-parameter metrics and exact agreement with the analytic AdamW decay-only formula. In the explicit-zero condition, all 16 gradient tensors were zero and every parameter remained bitwise unchanged. In the mixed-reward control, all 16 gradient tensors were nonzero and all 16 parameter tensors changed for every seed; pre-clip grad norms were 2.38–2.56. That establishes that the explicit-zero setting does not disable ordinary nonzero-gradient learning in this fixture.

The runner also emits a decay-only formula error for mixed-reward rows; that value is not interpretable because those rows contain gradient-driven updates. It is excluded from the outcome table and all claims. The runtime prints a PyTorch warning when converting this diagnostic tensor to a scalar; the raw values remain finite and the independent audit did not rely on this mixed-arm field.

Raw outputs are retained at [base](../results/rvl-grpo-weight-decay-compat-v2/base-run.json) (SHA-256 `f57d654f4feb31bfc0ba40e02bd86d4c5cd6d56b5d8cb380b674b1d2793d23f9`) and [candidate](../results/rvl-grpo-weight-decay-compat-v2/candidate-run.json) (SHA-256 `4860a66b3866f2043b2f1f7671b95ab43651a68629db026716c051514cdcc993`). The portable source diff is [here](../results/rvl-grpo-weight-decay-compat-v2/rvl-grpo-weight-decay-compat-v2.patch) (SHA-256 `0c7a540f9bd28db03b1594ebaff5686172adc9f6101423bda4d797f5c7e8474c`).

## Implementation and validation

The candidate is commit `2a47b3bf538d12c1f24fc3bc8bb188329ed32500` in branch `vare/grpo-zero-advantage-weight-decay`. `weight_decay` is appended after all prior config fields, preserving the prior positional argument order. The optimizer receives the config value explicitly and validation rejects non-finite or negative decay. A draft upstream PR is open at [Recursive-Verification-Lag #89](https://github.com/mitukx/Recursive-Verification-Lag/pull/89); it is awaiting review and has not been merged.

- Frozen base and candidate runners completed; source, runtime, protocol and runner hashes matched.
- Targeted test `python -m unittest tests.test_mini_lab_torch.TorchAcceptanceTests.test_weight_decay_is_explicit_and_preserves_legacy_default -v`: passed.
- Full `tests.test_mini_lab_torch` module: 12 passed.
- A separate read-only agent audit checked source/protocol hashes, the raw outcomes, and positional-field ordering; no discrepancy was found. Its record is [`independent-audit.json`](../results/rvl-grpo-weight-decay-compat-v2/independent-audit.json). This is not an outside human review or reproduction.
- VARE's broad CI was not run for this RVL-only candidate; the reported test status is the isolated RVL torch module above.
- Reproduction commands from VARE root:

```bash
python scripts/run_rvl_grpo_weight_decay_compat_v2.py \
  --rvl-source ../../work/rvl-arc-update-smoke-pinned \
  --arm base \
  --output /tmp/rvl-weight-decay-base.json

python scripts/run_rvl_grpo_weight_decay_compat_v2.py \
  --rvl-source ../../work/rvl-grpo-weight-decay-candidate \
  --arm candidate \
  --output /tmp/rvl-weight-decay-candidate.json
```

Each output path must be new. The runner verifies source hashes and locked dependency versions and refuses CUDA-enabled execution.

## Interpretation and limits

This supports exposing the optimizer setting while preserving the existing effective default. It does not establish that `0.01` or `0.0` is better for learning, that weight decay caused any prior task result, or that this behavior changes downstream task success. The fixture is tiny and randomly initialized; the three seeds are deterministic sanity replications, not inferential samples. It does not test a pretrained model, broad optimizer families, TRL, or an external system's acceptance of the patch.

**Decision: stop local work on this bounded finding while the draft PR is under review.** Do not run more model training for this issue. The portfolio's central unresolved evidence gap remains independent task-success improvement after a real policy update, and the full-Trainer result still lacks outside human reproduction.
