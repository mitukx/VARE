# Format versus correctness comparison v1

Protocol id: `format-vs-correctness-comparison-v1`.

This comparison is not authorized. `scripts/format_vs_correctness_comparison_v1.py` prints the refusal and exits 2. It does not read a smoke result, and a file named `smoke_result.json` does not change the exit code.

Protocol file: `protocols/format_vs_correctness_comparison_v1.json`. SHA-256 `fcc8b9abcc29287022f15aeb72876a41f7a75036da28b59ced2a0f23d85edeae`.

The exact-integer v2 base gate authorizes only the separate [one-update smoke](heldout-exact-integer-update-cost-smoke-v1.md). A retained smoke would still not start these arms. A later protocol would have to freeze its step count before any arm runs. This file leaves `optimizer_steps` as `not_frozen`.

The arms, if a later protocol is authorized, are `base`, `format_sft`, `format_reward_rl`, `correctness_reward_rl`, and `random_reward_rl`, with at least five seeds. The primary contrast is `correctness_reward_rl` minus `format_sft` on strict pass@1, using the committed confirm seeds and a paired bootstrap. Those confirm seeds are the `confirm_seeds` array in `benchmarks/heldout/integer_seed_split_v1/splits.json` (SHA-256 `4544a9a852f615f7cf9c52d7a4826ed70ea6c5f57829754ac2e10f3c1fea4287`). They are not a training set and they are not an early-stopping set. The 64-task exact-integer pack is not a training set either.

The empty report form is `docs/templates/paired-comparison-report.md`. Filling it in before a protocol is authorized does not create a result.
