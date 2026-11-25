# Public evidence bundle

This bundle contains the frozen Sprint IV v2 protocol, candidate-facing task inputs, public tests, public-safe grader receipts, normalized commands, resource samples, candidate patches, and exact hash-chained raw model/tool event journals. Hidden test source, hidden grader code, hidden stderr/stdout, full task workspaces, and model weights are excluded. They remain in the original local experiment tree. Hidden test identities are represented by SHA-256 only.

The v1 protocol was invalidated before inference because its candidate-visible task JSON exposed the repair operation. It started zero model attempts. Its original protocol hash and prelaunch audit are retained here. V2 uses new task IDs and behavior-only instructions.

## Verify

From this directory run:

```sh
python3 verify_evidence.py
shasum -a 256 -c SHA256SUMS
```

Do not rerun the model on these consumed Development tasks. Raw `events.jsonl` files preserve the model outputs, tool calls, results, and event-chain hashes. The run summaries distinguish model completion from the runner's post-interaction export error. The process failure is not scored as a resource abort.
