# VARE Sprint IX — research execution report (2026-10-28)

## Executive result

This sprint completed a preregistered, real-model A/B/C feasibility campaign on one newly qualified MLX-LM Development family. It produced **0 independently verified repairs from 1 unique family**. The result is a useful controller diagnosis, not coding-capability improvement: the stagnation guard reduced repeated searches but did not move the model to a useful source range; the localized arm reached the edit tool but exhausted the fixed per-turn output budget before producing an executable patch.

The model ran under the existing memory and swap caps. No attempt was resource-aborted. Public tests were not run on any model candidate because no edit was applied. Each unchanged candidate was independently graded and failed the task checks. No optimizer update, held-out evaluation, or durable long-horizon fault-injection experiment was performed.

The campaign also exposed a concrete implementation defect in PR #139: the duplicate-read guard expected a `sha256` field that the lease reader deliberately removes from its model-facing result. The guard therefore never activated. The fix and three regression tests were pushed to PR #139 at `79be1cae567bd5c3c63e26292a406b21c22e7233`. Local regression tests and lint pass; exact-head GitHub CI was still running at report capture. No mainline merge occurred.

## Task qualification and exposure

| Item | Pinned value |
|---|---|
| Task family | `mlx-lm-prefill-history-dev-01` (one Development family) |
| Upstream | [MLX-LM issue #1773](https://github.com/ml-explore/mlx-lm/issues/1773), [merged PR #1777](https://github.com/ml-explore/mlx-lm/pull/1777) |
| Baseline revision | `acb0240394dfc17ffd6db77a3ccecbdbde522ef9` |
| Fixed revision | `fa6cb55e7e3b21861263df0fc5d235552670dda7` (PR head `b4998c75d48c025103633ab45d95384b2e5a22d8`) |
| Candidate-visible source | `mlx_lm/generate.py` only; candidate workspace had a synthetic baseline commit and no upstream Git history |
| Baseline source SHA-256 | `502aad32fd59ce04cfca7f6e9f1e6b059570e25f64e866291fe4e56cba6a48d2` |
| Known-fix source SHA-256 | `c68732942b6a4653f8c303897d4ef85954fef4c2373b08e7c163baabe77a9599` |
| Incorrect-mutant source SHA-256 | `ef1c4cac3cd7bac55ef82c0267c15ee5b97f4bbdc4f628d531550fbc6c9635d3` |
| Qualification controls | Baseline public/hidden FAIL; known fix public/hidden PASS; incorrect mutant public/hidden FAIL |
| Prior VARE exposure | No matching task or editable-path manifest in 8,958 retained files; this does not exclude manual/external exposure |
| Public-model pretraining exposure | UNKNOWN |

The hidden grader is isolated from candidate tools. Its source and case-level outputs remain in the local-only evidence set. The public evidence bundle contains aggregate receipts only. The task family is now consumed Development data and must not be reused as fresh or blind-held-out evidence.

## Frozen experiment

Canonical protocol SHA-256: `b2cb492c07dd7a52ad5d6edc878841f8898d43c0a86e9f199b160aa5ede4a57e`. The byte-level hash of the formatted `protocol-v3.json` file is `3a0e771bb70a69969c969c497fecfb12f5b1945a6b18ba5574ae8361b6da06a4`; the former is the protocol identity in the sidecar. All 49 pre-inference asset hashes were rechecked with no mismatch.

The three arms used the same checkpoint, greedy decoding, seed `20261028`, task, verifier, 12-turn cap, 384 generated tokens per turn, 1,536 total generated tokens, 8,192 context tokens, 240-second agent wall budget, two-run public-test cap, and unchanged host resource limits. A/B differ only by the planned no-progress guard. C is an information-advantaged localization diagnostic, not a matched-information ability comparison.

| Arm | Interface/controller | Distinct model outputs | Executed tool results | Agent stop |
|---|---|---:|---:|---|
| A — autonomous | Transactional navigation, guard off | 11 schema-valid | 10 | Wall budget exhausted |
| B — stagnation-aware | Same interface, guard on | 8 schema-valid | 7 | Wall budget exhausted |
| C — localized | Frozen 80-line source window, localized interface | 2 schema-valid; third output incomplete | 2 | Edit text slot exhausted the 384-token turn cap |

All three event chains validate. No parser repair or post-hoc tool-call rewriting occurred. A/B’s final schema-valid actions were recorded after their last completed tool result and were not executed before the stop; C’s final output selected `replace_lines_transactional` but did not complete its required text slot.

## Model and resource envelope

- **Checkpoint:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, MLX snapshot `019cc73c45c770444708a6dd8690c66243cc5c80`.
- **Weight identity:** `model.safetensors`, 4,284,346,255 bytes, SHA-256 `56a3d94706833f753e6c6b47ea57af8ef638cd8cc1d74eca1142ba640b26060e`.
- **Tokenizer identities:** `tokenizer.json` SHA-256 `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf`; `tokenizer_config.json` SHA-256 `f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7`. No template override was used.
- **Runtime:** MacBook Pro M1 Pro, 32 GiB unified memory, macOS 27.0.1 arm64, Python 3.12.12, MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.21.0, MLX Metal GPU device.
- **Task grader image:** `localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388`.
- **Frozen caps:** process-tree RSS 10 GiB; maximum swap growth 32 MiB; minimum free swap 512 MiB; minimum available memory 2 GiB; minimum disk free 16 GiB; 300-second outer wall limit. None was relaxed.

| Arm | Generated tokens | Wall seconds | Generation seconds | Peak sampled process-tree RSS | Minimum available memory | Minimum free swap | Swap growth | Resource abort |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 112 | 269.615 | 260.51 | 2,720,235,520 B (2.53 GiB) | 5,723,193,344 B | 1,546,125,312 B | 0 B | None |
| B | 79 | 288.729 | 279.56 | 4,755,259,392 B (4.43 GiB) | 6,557,335,552 B | 1,562,902,528 B | 0 B | None |
| C | 404 | 85.463 | 77.63 | 4,745,805,824 B (4.42 GiB) | 6,803,210,240 B | 1,562,902,528 B | 0 B | None |

A/B elapsed time exceeds the nominal 240-second agent budget because the per-generation wall check cannot preempt an in-flight decode; both remained inside the separately frozen 300-second outer limit. This is execution overhead, not a resource abort.

## Outcomes and diagnosis

| Measure | A | B | C |
|---|---:|---:|---:|
| Unique search queries | 1 | 1 | 0 |
| Search actions / repeated identical actions | 9 / 8 | 2 / 1 | 0 / 0 |
| `read_lines` actions | 1 | 5 (same `1–100` range) | 1 (`364–443`) |
| Syntax checks | 0 | 0 | 1 (unchanged baseline) |
| Edit tool selected | 0 | 0 | 1, but argument slot incomplete |
| Edit attempt executed / applied | 0 / 0 | 0 / 0 | 0 / 0 |
| Public tests | Not run: no edit | Not run: no edit | Not run: no edit |
| Independent hidden grade | FAIL on unchanged baseline | FAIL on unchanged baseline | FAIL on unchanged baseline |
| Verified repairs | 0 | 0 | 0 |

The A/B result supports only a narrow controller observation. The guard changed the action sequence: repeated identical searches fell from eight to one, but B then reread the same first 100 lines five times. It did not improve task completion. Trace inspection showed why: `SourceEditLeaseStoreV1.read_lines` removes its full-file `sha256` from the model-facing result while retaining it internally in the opaque lease. The duplicate-read guard incorrectly required `result['sha256']`; therefore the repeated-read intervention was unreachable.

The localized arm opened the precommitted source window, checked baseline syntax, and chose the transactional edit tool. It then generated 370 tokens of replacement text but hit the 384-token turn cap before satisfying the text slot’s required terminal newline. No partial patch was applied. This is evidence of an edit-output budget/interface failure, not a graded semantic repair failure.

The task-level primary comparison is 0/1 family for A, B, and C, with paired A–B difference 0. One family and one attempt per arm cannot support an uncertainty interval or a generalization claim. The experiment does not establish broad coding ability.

## Controller correction and regression evidence

Commit [`79be1cae567bd5c3c63e26292a406b21c22e7233`](https://github.com/mitukx/VARE/commit/79be1cae567bd5c3c63e26292a406b21c22e7233) changes the duplicate-read key to hash the exact returned range content, path, and line bounds. It does not expose the full-file lease hash, normalize model output, reveal a fix, or replace a model action. The new intervention record is versioned `duplicate_source_read_guard_v2`. Three tests cover hash-less lease responses, changed source/edit progress, and invalid read results.

- Seven relevant Python test modules: **128 passed, 0 failed, 0 errors, 0 skipped**. JUnit: `results/sprint_ix_20261028/engineering/controller-regression-v2.junit.xml`.
- `ruff check scripts/run_local_agent.py tests/test_post_edit_navigation_gate.py`: **passed**.
- The same source commit used in the model runs was `cac34bc5ec2f5580582e5a49825ae3b4a6e5147f`. The guard correction occurred after all three frozen attempts and has **not** been evaluated with a model on a fresh task.

## Repository and PR state

The live `main` before and after the research work remained `f3ab90a36e8724441b6b9555255f8f75a0a4d529`. `main` requires one approving review plus nine status contexts, and administrator enforcement is enabled. The only listed repository collaborator was `mitukx`; no eligible independent reviewer was available in the live collaborator/reviewer data.

At the PR audit snapshot there were **18 open PRs and 9 Drafts**. The machine-readable ledger records each current PR’s base/head SHA, files, check state, review-thread count, stack dependency, disposition, rationale, and next action: `results/sprint_ix_20261028/pr-ledger.json`. Current latest-head checks for the 18 audited PRs were all green at the audit snapshot. The GraphQL history included superseded check contexts; the ledger separately records `gh pr checks` for the current PR head.

| PR | Current relevant state | Outcome this sprint |
|---|---|---|
| [#139](https://github.com/mitukx/VARE/pull/139) | Draft; updated head `79be1cae567bd5c3c63e26292a406b21c22e7233`; latest status view had 14 passing and 6 pending contexts | Added the duplicate-read guard fix and regression tests; no model follow-up on consumed family |
| [#142](https://github.com/mitukx/VARE/pull/142) | Draft evidence PR; audit snapshot head `2251087c41f17220a547f4459be698d34711534b`; Sprint IX evidence commit `b720ed44f0f323652e38fee08639401df799ff93`; review required | Sprint IX safe evidence/report appended; retain Draft pending explicit source-split audit and independent review |
| [#143](https://github.com/mitukx/VARE/pull/143) | Main-based, 19 latest checks green, independent approval required | No merge; no eligible reviewer was available |
| [#140](https://github.com/mitukx/VARE/pull/140) | Main-based, 19 latest checks green, 1 unresolved review thread and approval required | No merge; resolve the thread and secure independent approval |
| [#120](https://github.com/mitukx/VARE/pull/120) | Main-based, 20 latest checks green, 2 unresolved P1 threads | Remains blocked on evaluator completion-forgery findings |
| [#133](https://github.com/mitukx/VARE/pull/133) | Stacked on closed #132 lineage; 2 unresolved P2 threads | Remains blocked on source-index budget and transient rebuild recovery findings |

No PR was merged or closed in this sprint. PR #139 and #142 were updated; all other PR states are unchanged. No historical task, grader, protocol, or result was rewritten. The Sprint IX report and public evidence were appended to evidence-focused PR #142, not merged to `main`.

## Evidence and reproduction

The public-safe bundle is `results/sprint_ix_20261028/mlx_lm_1777_agentic_repair/public-evidence-v1/`. It contains the frozen protocol, model manifest, qualified task prompt and public test, all three raw model trajectories, resource samples, safe aggregate grader receipts, and a SHA-256 manifest. Its verifier command is:

```bash
python3 results/sprint_ix_20261028/mlx_lm_1777_agentic_repair/reproduction/export-public-evidence-v1.py --verify
```

The command passed for 49 payload files; manifest SHA-256 is `d5826bdfade0358087bbbd7268b261390f8cf25acf6f831797252a78487b493b`. Full local raw grader receipts, hidden grader source, qualification control outputs, and their 78-file local-only SHA-256 manifest (`c638f7b75b191d67523f3911e2e70350e95bf89d42d1e8553220291a49fbb47c`) remain under `results/sprint_ix_20261028/mlx_lm_1777_agentic_repair/`; these are excluded from the public export. The prior Sprint IV public evidence verifier was rerun and still passes: 65 files, 8 event chains, 107 events. The exact output is `results/sprint_ix_20261028/engineering/sprint-iv-bundle-verifier.log`. Re-running inference on this consumed family is not a reproduction of a fresh attempt and is prohibited as independent evidence.

## Gate decisions

| Area | Decision | Evidence-based reason |
|---|---|---|
| Real-model tool generation | **RUN — PARTIAL** | Qwen 7B generated 21 strict schema-valid actions from 22 outputs; 19 actions reached a tool result. One edit slot exhausted its frozen token cap. |
| External coding-agent task | **RUN — PARTIAL** | One upstream MLX-LM Development family passed baseline/fix/mutant qualification and received three preregistered attempts. |
| Independent verified repair | **NO-GO** | 0/1 unique families repaired; no patch was applied. |
| Stagnation-aware controller | **PARTIAL** | It reduced repeated search but permitted repeated identical reads because the guard contract was wrong. A code correction now passes unit/regression tests; model effect remains unmeasured. |
| Coding SFT / GRPO | **NO-GO** | No verified coding-reward positive trajectory, no eligible held-out gate, and no reason to perform a formal optimizer demonstration. Updates: 0. |
| Held-out improvement | **NO-GO** | Only one Development family was used; prior exposure/pretraining is not fully auditable. |
| Model-coupled long-horizon RL | **NOT RUN** | No durable continuation, fault injection, optimizer, or checkpoint promotion/recovery was connected to these trajectories. |
| RSI research automation | **NOT RUN** | Task selection and the next intervention were human-selected; no autonomous outer-loop experiment choice or measured next-task improvement occurred. |
| Mainline integration | **NO-GO this sprint** | Branch protection requires an independent approval; none was available. Main SHA did not change. |

## Next highest-value experiment

Qualify a second, disjoint Development source-fix family and prospectively compare the no-progress guard correction against the prior controller at a larger per-turn edit budget, keeping the model, tool schema, and grader fixed. The single decision-changing outcome is whether the corrected controller converts repeated reads into a complete candidate edit and independent pass. Do not rerun the consumed MLX-LM family or claim the code-level guard fix improves repair success until that new model experiment is completed.
