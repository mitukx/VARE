# Sprint VI exploratory repeat-exposure campaign

This bundle records the Qwen2.5-Coder-14B paired run for Transformers PR #49842. It is preserved as an exploratory controller diagnostic only. The task family is **not fresh**: Sprint II already used the same MusicGen Melody cached-position bug family under `results/sprint_ii_20261027/external_coding_dev/musicgen_melody_cached_position_v1`. The prior exposure was identified in the retained `invalidation-v1.json` before this bundle was assembled, but was missed by the launch gate. The paired 14B run must not be counted in fresh-task, held-out, or training metrics.

The task was genuinely qualified at the source/grader level: baseline fails public and hidden checks, the pinned known fix passes, and an incorrect mutant fails. This does not undo prior model exposure. Pretraining exposure remains unknown.

## Frozen model and protocol

- Model: `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`, cached snapshot `29efdbab55a161237ab1e432a3abaf6c7ae2b477`, 4-bit, 8,321,093,967 bytes.
- Runtime: MLX 0.32.4, mlx-lm 0.32.2, Python 3.12.
- The invocation specified `--mlx-device cpu` and the event journal reports `device=cpu`; macOS process sampling showed Metal activity. CPU-only execution is therefore **not established**.
- One paired task family, seed 2029, temperature 0.2, top-p 0.95, top-k 20, sampling enabled; at most 12 turns, 1,024 new tokens per turn, 4,096 total new tokens, 8,192 context, 1,800 seconds.
- Autonomous arm searches and reads sources. Localized arm receives a deterministic 80-line AST excerpt selected only from task text and candidate-visible source. Interpret this as an information-value comparison, not a pure coding-skill comparison.
- Resource limits were held fixed: at most 512 MiB swap growth, at least 180 MiB free swap and 25% free memory at runtime. No cap was raised.

## Outcomes

| Arm | Model actions | Tool results | Edit proposals | Applied edits | Public tests | Candidate grade | Status |
|---|---:|---:|---:|---:|---|---|---|
| Autonomous | 12 schema-valid actions; 846 generated tokens | 1 search, 5 reads, 4 syntax rejects, 1 public test failure, 1 file listing | 4 | 0 | Baseline regression failed with `IndexError` | Not run: no patch | Completed 12-turn budget; no patch |
| Localized | 4 schema-valid actions; 1,904 generated tokens | 2 reads, 2 transactional policy rejections | 0 committed | 0 | Not run | Not run: no patch | Resource-aborted at 631.6 seconds when measured swap growth reached 653.5 MiB, above the unchanged 512 MiB cap |

Both candidate target files remained byte-identical to the baseline (`5b27dea9…988b82`); their canonical Git diffs were empty. Autonomous edit proposals failed Python syntax validation at line 110. Localized edit proposals requested `strip_single_python_fence`, but the exact fence contract rejected both with `python_fence_not_exact`. No model attempt is a repair, and the hidden grader was not run on either candidate.

Autonomous resource record: 1,229.9 seconds, minimum free memory 27%, swap growth 427.37 MiB. Localized resource record: 631.6 seconds, minimum free memory 28%, swap growth 653.5 MiB, exit 75 from the monitor. System-reported swap capacity changed during the latter run; the monitor's recorded used-swap delta is retained as the governing abort measurement.

The autonomous agent did run the public regression test against the unchanged baseline and observed the expected failure. That is a useful diagnosis, not a repair result. The localized attempt terminated before public tests or candidate grading.

## Reproducibility and integrity

`protocol.json`, `monitor-protocol.json`, the source-only localization record, exact model snapshot manifest, preflight, command records, event journals, resource samples, monitor summaries, and append-only assessments are retained. `attempt-assessment-v2.json` is a derived reconciliation; the original event journals and model outputs remain unmodified. Event journals use a per-event SHA-256 chain. `sha256-manifest.json` covers every bundle file except itself. Run `python verify_evidence.py` from this directory to validate the artifact hashes, event chains, unchanged candidate sources, and the exposure disqualification.

The candidate-visible baseline source, task prompt, and bounded public test are included. Hidden test source, grader implementation, fixed source, and model weights are excluded. Their source/grader identities and control outcomes are recorded in `qualification-summary.json` and `control-*.json`.

Prior Sprint II records remain under the original VARE worktree and are not copied or altered by this addendum. `append-only-prior-exposure-reconciliation-v1.json` documents the launch-gate miss and disqualifies this run from fresh-task claims. The separate `reconciliation_49900_append_only_v2.json` corrects a different legacy diff-wrapper artifact without changing the #49900 historical files.
