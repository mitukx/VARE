# Incomplete tool-call retry budget — controller change record

## Triggering evidence

The one authorized Qwen3-4B attempt on Transformers #49458 emitted three edit-tool calls without a closing delimiter. The captured turn lengths were 256, 255, and 256 generated tokens under a 256-token per-turn cap. No call was dispatched, the candidate diff was empty, and the family is consumed. The attempt transcript SHA-256 is recorded in controller-change.json.

## Change

The local runner has an opt-in `--incomplete-tool-retry-tokens` setting. If an incomplete tool call enters the existing no-tool retry path, the next generation may use the configured larger per-turn cap. The request stays bounded by remaining total tokens and available context. The default is unset, so existing frozen protocols keep their original per-turn behavior. The retry reminder reports the next cap, and each turn records its effective generation cap.

The protocol-driven separated evaluator forwards this setting only when the frozen budgets specify it. The evidence collector checks the attempt budget against the protocol and validates per-turn caps, the preceding incomplete-call/reminder evidence, and the expanded-retry turn list. Partial calls remain undispatched. The change does not alter the consumed #49458 attempt, increase the total-token budget, or grant public-test/grader access.

## Validation and limits

Ruff passed on the runner, protocol-driven evaluator, evidence collector, and changed tests. The focused controller, retry-wording compatibility, and evidence-integrity suite passed 77 tests. Exact commands are listed in research/flagship/REPRODUCE.md.

No model inference or grader invocation occurred. The change has not been shown to increase valid tool calls or repair success. A later comparison would require a separate frozen protocol and distinct authorized task; this record does not authorize one.
