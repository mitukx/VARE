# vLLM #64830 scoped source qualification

**Status:** QUALIFIED_SOURCE_LEVEL for the two method-level cases only. **Split:** UNASSIGNED.

## Result

The unchanged upstream HiSparse and watermarking cases distinguish the pinned pre-merge method, a wrong mutant, and the upstream-fixed method under the frozen CPU runner:

| Arm | HiSparse | Watermarking | Pytest | Time | Peak process-tree RSS |
|---|---|---|---:|---:|---:|
| Baseline attempt-002 | intended assertion failure | intended assertion failure | 2 failed | 0.53 s | 3,932,160 bytes |
| Wrong-mutant attempt-002 | intended assertion failure | intended assertion failure | 2 failed | 0.529 s | 3,932,160 bytes |
| Upstream-fixed attempt-002 | passed | passed | 2 passed | 0.529 s | 3,932,160 bytes |

Each case failed at the unchanged post-override assertion in baseline and mutant; the fixed method passed both. All three selected nodes collected and ran. There were no skips/errors, timeouts, RSS breaches, or mismatched input hashes. Each command/profile was bound in execution Addendum 4 (SHA-256 `dae44a37fa8e7c66276acb0117b155a5d84f4d8a6d3aad7b86816a5a2d8b5396`).

## Isolation and exposure

Each arm used a default-deny, network-deny Seatbelt profile. Its only allowed write subtree exactly matched that arm's result directory; the source/test overlay was read-only to pytest. Limits were 600 seconds and 4 GiB process-tree RSS. The runtime was Python 3.12.12 / pytest 8.4.2 on macOS arm64.

The first baseline invocation is preserved as a pytest setup failure before collection (`/dev/null` denied by Seatbelt); it is not a behavioral result. Addendum 3 was superseded before invocation after its paths failed preflight. Addendum 4 corrected profile/output binding and all three gates passed.

The source and test were exposed to the research agent. No model or agent inference, training, Held-Out access, paid compute, or API use occurred. Public pretraining exposure remains UNKNOWN. The task remains UNASSIGNED.

## Scope limits

This result covers only `VllmConfig._get_v1_model_runner_unsupported_features` with the HiSparse and watermarking cases. `test_custom_speculator` is excluded because its full registration and VllmConfig validation fixtures are outside this method-only overlay. This is not a whole-PR qualification, agent repair, model capability result, Coding RL improvement, or pretraining-independence result.

## Reproduction

Verify `protocol-addendum-4-output-profile-binding.json.sha256` and the bundle `SHA256SUMS`. Inspect per-arm exact argv, inputs, receipts, and classifications under `baseline-attempt-002/`, `wrong-mutant-attempt-002/`, and `upstream-fixed-attempt-002/`. Reproduce only on a clean checkout with the pinned source inputs; preserve the original one-shot receipts.
