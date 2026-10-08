# Protected evaluator integrity after commands

## Question and frozen rule

Can a test or metric command persistently change a protected evaluator file after the initial integrity check and still return a successful evaluation? The CPU-only fixture and decision rule were frozen in [`protected_evaluator_post_command_v1.json`](../protocols/protected_evaluator_post_command_v1.json) before the baseline run.

## Finding

At revision `46a890e`, the evaluator checked protected hashes once before running commands, then trusted the cached `integrity_ok` value when deciding the result. A deterministic test command overwrote its protected file and exited with status 0. The evaluator reported `passed=true`, `integrity_ok=true`, and score `0.9`, even though the protected bytes had changed. The raw pre-fix result is retained in [`baseline.json`](../results/protected-evaluator-post-command-v1/baseline.json).

This is a persistent post-command mutation gap in the evaluator's fail-closed contract. It does not demonstrate an escape from the documented host-process boundary.

## Change and validation

The evaluator now rechecks protected hashes after every test and metric command. It records changed or missing protected paths, stops before running later commands, does not trust a metric result produced during a mutation, sets the score to zero on integrity failure, and keeps `passed=false`. Existing pre-execution checking remains in place.

The new regression cases cover mutation from both a test command and a metric command. A standard-library reproducer is available as [`reproduce_protected_evaluator_post_command_v1.py`](../scripts/reproduce_protected_evaluator_post_command_v1.py); the captured fixed result is [`fixed-validation.json`](../results/protected-evaluator-post-command-v1/fixed-validation.json).

Observed on Python 3.12.12, macOS arm64:

- Test-command mutation: the command itself exits 0, but the evaluator reports `passed=false`, `integrity_ok=false`, score `0.0`, and `protected file modified: protected-test.py`.
- Metric-command mutation: the result is rejected, score is `0.0`, its metric value is discarded as NaN, and the later metric command is not run.
- Untampered control: passes with `integrity_ok=true` and score `0.9`.
- No model weights, GPU, network, paid API, or external compute were used.

## Limits

This is one deterministic harness-correctness witness, not a false-accept rate, a security certification, a model-learning result, or evidence of capability gain. A command that changes a protected file and restores it before exit can evade a post-command hash check; malicious code is still not sandboxed. Strong isolation requires a separate container or VM boundary as documented in [`environment_factory.md`](environment_factory.md).
