# Phase A Grader Integrity v7

## Hypothesis and scope

Candidate code can mutate the tokenizer mapping after its literal is initialized while leaving the initializer AST and sampled routes intact. The v7 grader must reject changes to the complete runtime mapping and override set. Legacy Git-backed agent workspaces must also be verified from their bytes before the model runs, including tasks without file-level SHA locks.

This is grader and workspace-integrity validation. It is not a Coding-Agent success or a policy-improvement experiment. No model inference, SFT, or GRPO was run.

## Frozen provenance

- Protocol: `vare-coding-agent-phase-a-integrity-v7`
- Protocol SHA-256: `eee325a414af59bd5d5c27d00ecba5a77779d7098ac017626396db640e43e9d3`
- Grader SHA-256: `8572b17653c26de7b9bf8ae7e1289cdaf0b669b8cc420f5a825c1b60438f5882`
- Tokenizer adapter SHA-256: `8bc2769384317e9c94a111238c84b204f62e23e972b5d4eb29cb7e1530c6e796`
- Frozen grading image and inspected image ID: `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Feature availability in that image: tokenizers=true, sentencepiece=false, g2p_en=false, mistral_common=false.
- Tokenizer task baseline: `07338b6c74a578868368e6e549dea83414e4b8cb`; pinned known fix: `462bf5a50f6434e9610513b34537a892967ccd84`.
- Fusion task baseline: `cb33194ad6152bd9fad6305378d92db385dd7b32`; pinned known fix: `527725cace0b038a0ac8ab738e921f118bb2668a`.

## Calibration

| Task | Candidate | Result | Candidate source SHA-256 |
|---|---|---|---|
| Tokenizer routing | Pinned baseline | Rejected (`wrong_output`) | `831ae31ec8ab6dd6c5224ea42d0c508d515a8314005064eb594a715f835a6f61` |
| Tokenizer routing | Pinned known fix | Passed | `89d301092f223af0c17fd813451dfd93f344282e51a151cf00604d7647036957` |
| Tokenizer routing | Known fix followed by dynamic `getattr(..., "pop")("bert")` mutation | Rejected (`wrong_output`) | `7f046ddb2b6bda7252c4dacece90b058a1e0c17cd68f5d83399fffc7a29ba096` |
| Fusion registration | Pinned baseline | Rejected (`wrong_output`) | `9a05ab8ccc0ab4756526bcb76c05099a8ec3f1b634c617c620305d1a81cc2f0c` |
| Fusion registration | Pinned known fix | Passed | `af053f2c2eaae2c744778df465002db73e99054dc059e82e394f43c1dca0b608` |

The tokenizer adapter now hashes the entire post-import runtime mapping and override set. The parent grader independently resolves the complete expected mapping from the locked registry AST and frozen optional-dependency state. The dynamic `getattr` red-team case bypasses the source-level mutation filter but still fails the runtime comparison. The expected registry, environment flags, image, protocol, grader, and adapter are hash-bound.

`verify_agent_workspace` now checks pre-run Git cleanliness, rejects `assume-unchanged` and `skip-worktree` flags, compares locked source hashes, and—for historical tasks without file hashes—compares every allowlisted file byte-for-byte with its pinned Git blob. Focused tests exercise dirty workspaces and both index flags.

## Verification

- `tests/test_grade_transformers_phase_a.py` + `tests/test_separated_grading.py`: 61 passed, 4 skipped.
- Ruff: passed.
- mypy: passed.
- `git diff --check`: passed.
- Raw grader JSON, test output, Ruff output, and mypy output are stored in `results/phase-a-integrity-v7/` and covered by `SHA256SUMS`.

## Limitations and decision

The exact historical runner source pinned by v2/v3 was not recovered from inspected Git history or retained caches. No hypothetical source was reconstructed; the existing reproducibility limitation remains. These calibrations validate two narrow Transformers task adapters and do not establish a general sandbox for arbitrary repository tests or any Coding-Agent capability. PR CI and independent review are still required. Phase A model feasibility remains NO-GO until a fresh task cohort and a repeatable model-generated verified repair are available.

## Reproduction

From a clean checkout with the locked Docker context and image:

```bash
PYTHONPATH=src DOCKER_CONTEXT=colima python scripts/prepare_transformers_phase_a_v1.py --task-id transformers-49116-tokenizer-routing-v2 --upstream-repo /path/to/transformers --output-root /tmp/vare-phase-a
PYTHONPATH=src DOCKER_CONTEXT=colima python scripts/grade_transformers_phase_a_v1.py --task-id transformers-49116-tokenizer-routing-v2 --workspace /tmp/vare-phase-a/transformers-49116-tokenizer-routing-v2/workspace --image 'vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b' --output /tmp/tokenizer-grade.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_grade_transformers_phase_a.py tests/test_separated_grading.py -q
```

Two superseded development calibrations are retained under `results/phase-a-integrity-v7-r1/` and `results/phase-a-integrity-v7-r2/`. They are explicitly excluded from v7 conclusions because they preceded final runtime-map checks and type-check correction.
