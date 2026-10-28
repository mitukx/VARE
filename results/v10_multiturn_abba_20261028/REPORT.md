# v9 vs v10 shared-prefix KV reuse: 8/16-turn ABBA systems measurement (run-001, partial)

Status: **aborted partial run**. 20 of 24 preregistered trajectories completed (blocks 1-5). Block 6 (intervals, 16 turns) was stopped by the operator because the host was needed by another workload. No preregistered stopping rule fired.

Evidence class: a systems measurement with a real local model on a synthetic Development fixture. This is not capability evidence.

## Identities

| Item | Value |
|---|---|
| Base | PR #139 head `df0a360e` (`codex/development-navigation-campaign-v1`) |
| Protocol | `protocol.json`, sha256 `283c01ca9a94cfebe2eb8c12f6d9a9b1c63de302308dd966024cb5b81c03d5f2`. Frozen in commit `577f9225` before inference; implementation commit `1ef3af71` |
| Amendment 001 | `0d665703…`, commit `37137443`. Probe v2 replaces the crashed v1 probe. **Not executed** |
| Model | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, snapshot `019cc73c45c770444708a6dd8690c66243cc5c80`, weights sha256 `56a3d947…060e`, already cached, offline |
| Runtime | Python 3.12.12, mlx 0.32.4, mlx-lm 0.32.2; Apple M1 Pro 32 GiB |
| Decoding | Greedy; 256 tokens per turn; 12288-token context limit |

## Design

- Each block has 4 runs, ordered ABBA or BAAB (A = v9, B = v10). Every trajectory runs in a fresh process.
- Pairs are adjacent runs within a block.
- Tool feedback is real, from the repository's executors (lease store, navigator, syntax check, and an isolated public-test runner). `finish` is offered only on the last turn.
- Completed so far: 6 pairs at 8 turns and 4 pairs at 16 turns.

## Results (paired, v10 vs v9; 95% t interval and bootstrap percentile interval, 10k resamples, seed 20261028)

| Metric | 8 turns (n=6 pairs) | 16 turns (n=4 pairs) |
|---|---|---|
| Trajectory wall reduction | 47.4% (t 41.7–53.1; boot 43.4–51.1) | 58.0% (t 56.2–59.9; boot 57.1–59.1) |
| Mean trajectory wall, v9 → v10 | 121.4 s → 63.8 s | 412.3 s → 173.7 s |
| Prefill seconds reduction | 53.4% (t 49.0–57.7) | 61.0% (t 59.9–62.1) |
| Amortized prefill token reduction | 60.3% (t 58.2–62.5) | 65.5% (t 63.6–67.4) |
| Decode seconds difference | +0.32 s (t −0.19 to +0.83) | −0.14 s (t −0.63 to +0.35) |
| Peak Metal increase | +0.32 GB (t 0.18–0.46) | +0.92 GB (t −0.01 to 1.85) |
| Process peak RSS increase | +0.20 GB (t −0.40 to +0.80) | −0.006 GB (t −0.02 to +0.01) |

Pooled over 10 pairs, the wall reduction was 51.7% (t 46.8–56.6).

Peak Metal ranges:

| Arm | 8 turns | 16 turns |
|---|---|---|
| v9 | 4.9–5.1 GB | 5.06–5.11 GB |
| v10 | 5.1–5.5 GB | 5.45–6.47 GB |

Per-turn amortized prefill stays roughly flat for v10 (≈0.8k → 3.3k tokens). For v9 it grows with history (≈0.8k → 11.7k tokens). Per-turn means are in `frozen-analysis.json`.

### Equivalence, safety and reliability

- **Generated token IDs were identical in all 10 pairs, at every stage and every turn.** Within-arm repeats were also identical in all blocks.
- Prefix audit: 0 violations. v10 reused the exact prefix on 362 of 362 stages, with 0 fallbacks.
- Metal active memory after each turn returned to the post-load level (growth 0 bytes), so no cross-turn retention was observed.
- Parser acceptance was 108/112 turns in both arms. The rejected turns were slot-budget exhaustion, identical in both arms.
- There were no timeouts, crashes, or controller exceptions. Swap growth was at most 217 MiB, system-wide, in a single run.
- Task correctness was 0/10 for v9 and 0/10 for v10. No edit was ever applied (`edits_applied` = 0), so the "changing source" condition was **not exercised**. Tool responses did change across turns: reads, failed edits, and test output.

### Decision

- The frozen analysis decided **`NO_GO_V10_KEEP_V9_DEFAULT`**. Two gates failed:
  1. `kv_correctness`: no probe output existed, because the v1 probe crashed on an instrumentation defect (amendment 001) and probe v2 was not run.
  2. `action_equivalence`: this failure is an artifact. In 4 pairs, a rejected turn is represented by its raw controller output, which embeds the protocol id (`vare.schema-slots.v9` vs `v10`). The post-hoc comparison in `posthoc-action-normalization.json`, which is not preregistered, compares parse status, parsed action, and generated token IDs, and finds all 10 pairs equal.
- The efficiency, memory, reliability, and retention gates passed on the completed pairs.
- **Recommendation: do not promote yet.** Keep v9 as the default and keep v10 as an explicit opt-in. v10 keeps its in-controller exact-prefix fallback, and v9 stays selectable. Promotion needs:
  - the amendment-001 probe v2 run;
  - block 6;
  - a fixture or trajectory in which edits are actually applied;
  - an analysis fix for the protocol-id artifact under a new protocol version.

## Limitations

- The sample is small: 6 and 4 pairs from 3 and 2 variants. Pairs within a variant are not independent.
- Greedy decoding makes repeats deterministic, so the intervals reflect timing noise, not behavioural variance.
- System memory and swap are shared with unrelated processes.
- There is a single machine, a single model, and a synthetic fixture only.
- The run is partial, and the 16-turn horizon lacks the intervals variant.

## Files

| Path | Contents |
|---|---|
| `run-001/` | Raw records: preflight, events, probe failure, and per-run trajectory, guard, grade, and resource samples |
| `run-001/runs/021-*` | Incomplete run 021: logs only |
| `run-001-analysis/frozen-analysis.json` | Output of the frozen `scripts/analyze_v10_multiturn_v1.py` |
| `run-001-analysis/posthoc-action-normalization.json` | Post-hoc comparison (not preregistered) |
| `run-001-analysis/campaign-abort.json` | Abort record |
