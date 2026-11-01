# VARE Development coding-agent sprint — 2026-10-26 JST

The sprint produced three regression fixes, nine passing task-calibration controls, two completed real-model attempts with independent behavioral grading, two resource-aborted attempts with durable partial evidence, and a verified recovery audit using a retained real-model trajectory. **No model-generated repair succeeded. No optimizer or policy update ran. Learning and capability-improvement claims remain NO-GO.**

The completed attempts are a narrow Development feasibility result on VARE's own source. They do not establish general repository-agent performance, independent held-out improvement, or external task certification.

## Executed results

| Checkpoint / execution | Task | Observed result | Retained generation | Process wall time |
| --- | --- | --- | ---: | ---: |
| Qwen2.5-3B 4-bit, MLX/Metal | Navigation output overflow | Resource abort; no completed generation retained | 0 completed tokens recorded | 4.065 s |
| Qwen2.5-3B 4-bit, MLX/CPU | Nonempty evidence manifest | Completed, graded **FAIL**; invalid JSON at turn one; unchanged candidate | 23 tokens | 4.711 s |
| Qwen2.5-3B 4-bit, MLX/CPU | Navigation rebuild recovery | Completed, graded **FAIL**; invalid JSON at turn one; unchanged candidate | 27 tokens | 4.440 s |
| Qwen3-4B 4-bit, MLX/CPU | Navigation output overflow | Two valid calls (`list_files`, `search_source`), then resource abort; no edit | 36 tokens in two completed generations | 10.458 s |

Only the two completed, graded attempts form the reported coding outcome: **0/2 repairs, 0 applicable edits**. The aborts are not scored as completed attempts. In-flight generation at termination is unknown; retained token totals are not an exact count of all accelerator/CPU work. There were no sampled GRPO groups, so group-relative reward variance was not measured.

The initial six-cell Metal campaign stopped on its first cell. A separately committed CPU protocol admitted two previously unattempted Qwen2.5 cells. Another committed CPU protocol admitted the previously unattempted Qwen3 cells and stopped on its first cell. Remaining cells were not run. No consumed cell was retried, no model output was repaired after generation, and the resource caps were not increased.

Every model call used a cached snapshot, offline flags, a 12-turn / 4,096-token total budget, a 1,024-token per-turn budget, and a 12,288-token context ceiling. The two checkpoint snapshots are pinned in the protocols with complete file hashes. No model weights were downloaded, no paid API or cloud compute was used, and paid spend was zero.

## What changed and what was qualified

The source baseline is PR #137 at `6d95bd9196645ec7ecf7ff5d0c9ec997b24dcab1`. The manifest verifier file was imported explicitly from the separate PR #120 source at `58ecd9c9727a1544a5a08c9c250f1f36dd9a5f8d`; the rest of that PR was not merged.

1. Navigation now trims auxiliary path inventory before dropping useful ranked hits when the serialized response exceeds its budget. It records inventory truncation and preserves an accurate serialized byte count.
2. Navigation retries a bounded index build after a recoverable read/size failure. Restored source becomes searchable; an empty allowlist still fails.
3. The manifest verifier requires a nonempty `files` object or array. Missing and empty inventories cannot produce successful completion; valid formats still verify file bytes and reject corruption.

The three Development tasks represent **two source families**, not three independent repository families. Their candidate-visible inputs are the exact pinned full source files. Model tools can access only the declared source path. Task prompts state the defect and the permitted function body; they do not include the known fix, calibration logs, or hidden check inputs.

For each task the precommitted qualification protocol required baseline failure, known-fix success, and incorrect-mutant rejection. **All nine controls matched their frozen expectations**, and all nine candidates were admitted by the structural gate. Controls contain 8 behavioral checks for each navigation task and 10 for the manifest task. The oracle was authored separately from the candidate source and kept outside the model workspace.

Candidate modules are never imported. The grader verifies that all AST content outside the selected function body is unchanged, checks names/calls/attributes against a restricted repair language, then transplants that function into pinned trusted code. Regression probes reject module-scope completion forgery, `os._exit`, output forgery, reflection, shared-module assignment, and edits to another function. This is a narrow function-contract boundary, not certification of arbitrary hostile Python or general repository execution. It does not resolve PR #120's historical parser grader P1.

These are investigator-authored, same-host controls, **not an independent third-party review**. A retrospective scan of 21 retained model records found no matching task/source exposure; two oversized JSON files and archives were outside that scan. External exposure is unestablished. These tasks are Development only and add zero eligible held-out families.

## Cached template defect

The cached Qwen2.5-3B tokenizer configuration teaches this literal tool-call example:

```text
{{"name": <function-name>, "arguments": <args-json-object>}}
```

Instantiating its placeholders yields invalid JSON. Both completed model outputs repeat doubled braces and fail the unchanged strict parser. The exact checkpoint template was rendered offline; a prospective correction to the literal example renders valid JSON. Tests also cover ambiguous/unrecognized inputs and preserving other tokenizer fields.

The correction is exported to a separate tokenizer configuration. The cached snapshot and both failed trajectories remain unchanged. No corrected-template model run was performed. The repeated pattern suggests a mechanism for the formatting failure, but **a causal effect on coding success has not been measured**.

## Resource and recovery evidence

Each model cell required at least 6 GiB reported available memory and 512 MiB free swap. Limits were 5 GiB sampled process-tree RSS, 6 GiB MLX allocation, 256 MiB host swap growth, and 660 seconds external wall time. Sampling was every 250 ms. RSS is a sampled lower bound on peak process memory; host swap is a system-wide counter and cannot be attributed exclusively to the model.

The Metal attempt observed 628.625 MiB swap growth and was killed. The Qwen3 CPU attempt observed 426.313 MiB growth and was killed. The completed Qwen2.5 CPU cells observed 161.313 MiB and 0 MiB growth, respectively. Their sampled RSS peaks were about 1.87 and 1.86 GiB; MLX reported roughly 2.44 GiB peak allocation. These are different workloads and host states, not a controlled Metal-versus-CPU comparison.

The completed manifest cell exercised a predeclared controller SIGKILL after a durable model rollout and before grading. Recovery checked the pending artifact hashes, retained one model invocation, and graded the same unchanged candidate. The initial parent did not separately persist its observed child return code; the retained evidence is its fault marker, pinned control flow, recovery record, and event chain.

A separately frozen **retained-trajectory recovery audit** makes that boundary independently inspectable: a subprocess durably records the boundary and actually exits with observed return code **-9**. Pinned recovery then grades the retained real-model candidate with **zero new model invocations**. Source tampering fails before grading, and an already completed cell refuses another execution. The recovered semantic result remains FAIL. This is bounded artifact/controller recovery evidence; it is not recovery during generation, uninterrupted long-horizon RL, asynchronous throughput improvement, or a learner crash experiment.

## Live GitHub and integration path

The initial live snapshot had **32 open PRs** and main `f3ab90a36e8724441b6b9555255f8f75a0a4d529`. PR #131 remained Draft at `139de538`; PR #137 was unmerged at `6d95bd91`. Their exact-head `ci`, `cpu-evidence`, and `e2e-system-cpu` workflows had passed. No approving GitHub review was recorded on either selected PR. The two P1 threads on #120 and two P2 threads on #133 were unresolved. Passing CI did not close those findings.

The live-base ancestry audit found missing edges at #132→#131, #114→#99, #108→#107, #107→#106, #103→#102, and #100→#99. The stack must not be merged mechanically.

The execution path is deliberately pinned:

- Current coding navigation and invocation retention: #137 exact source, followed by the local prospective repair commits and their separate campaign protocols.
- Historical 79-runtime-file continuation evidence: #131's exact source. #137 does not contain #131's latest commit as an ancestor. Its V1–V5 recovery predicate already binds the endpoint to the resumed trace, but its inventory lineage differs. Historical evidence must use its own pinned auditor.
- Manifest correction: the standalone #120 verifier source, with explicit provenance. The historical #120 parser grader remains excluded.

The two V5 integration-test failures initially observed in this sprint came from using current HEAD as a historical V5 campaign after runtime source changed. Historical acceptance tests now use the recorded #137 source, and a regression confirms the changed runtime is rejected. The production continuation auditor was not weakened or changed. A new continuation run against the changed runtime would require its own source-inventory protocol; this sprint did not run one.

No PR was merged and main was not modified. New work is reviewable on the isolated `codex/development-navigation-campaign-v1` branch; dependency and human-review gates remain in effect.

## Reconciling three certificates and the reported 10/15

The #129 audit was replayed against the existing upstream MLX-LM Git mirror and returned **AUDIT_PASS: 3 qualified / 1 unqualified**. The three are #1926, #1938, and #1979; #1902's known fix fails.

The later ledger contains six original reserved families and four extensions, for ten source/grader certificate entries:

| Evidence history | Families |
| --- | --- |
| Replayed #129 source/admission/isolation audit | MLX-LM #1926, #1938, #1979 |
| Later original-pool controls | Transformers #48319, #49208; MLX-LM #1921 |
| Prospective extensions | Transformers #49147; MLX-LM #1898, #1877, #1880 |

All 71 path/hash references in that ledger were reconciled: 70 were present in the selected tree; the missing retired #49116 exposure receipt was recovered from its pinned historical Git blob and matched the expected hash. No exposure exclusion was reversed.

This confirms the ledger's identities and arithmetic, not a uniform new certification of all ten. The seven later entries were not rerun through their complete environment and isolation audits in this sprint. #1880 also carries a post-outcome raw-log adjudication of its baseline matcher failure. The ledger itself says fully model-eligible held-out count is **UNKNOWN_NOT_ESTABLISHED**. Thus **10/15 is not accepted as model-training readiness**. At least five more distinct source/grader families would be needed even if all ten clear a common eligibility standard; exposure, isolation and independent-review requirements still apply.

## Decisions

| Gate | Decision and reason |
| --- | --- |
| Prospective regression fixes | GO for technical review; focused checks pass |
| Narrow Development grader calibration | GO within the declared restricted contract; 9/9 controls |
| Qwen2.5 CPU campaign completion | Completed; 0/2 repairs; no coding success |
| Qwen3 campaign completion | NO-GO after resource abort; partial valid tool use only |
| Corrected Qwen2.5 template inference | NOT_RUN; no retrospective rescore |
| SFT, GRPO, optimizer updates and promotion | NO-GO: no positive repair data, no sampled reward groups, no eligible confirmation set, and insufficient current swap admission |
| General real-coding async/long-horizon comparison | NO-GO; only the scoped post-rollout recovery boundary was audited |
| RSI / learned task selection | NOT_ADMITTED; no downstream learning outcome to optimize |

The highest-value next executable model intervention is the exported template correction under a new, independently reviewed protocol with genuinely fresh Development cells and adequate resources. It is not a reason to rerun these failed frozen cells or train on their zero-repair outputs.

## Verification and reproduction

Final focused integration checks: **79 passed, 69 model-dependent continuation cases skipped**; lint and diff checks passed. The full repository suite and new exact-head remote CI are not claimed as passed in this report. The initial integration failures and the rejected CPU v1 syntax setup are retained alongside the passing checks.

Primary files in this bundle:

- `results/vare-development-repair-v1/model-evidence-audit.json`: source-commit, model-file, journal, candidate and grade reconstruction.
- `task-qualification/qualification-summary.json`: nine raw calibration outcomes and input hashes.
- `cpu-model-campaign/`: both complete raw model trajectories, candidate files and grader outputs.
- `model-campaign/` and `qwen3-cpu-model-campaign/`: resource traces and partial invocation journals.
- `retained-trajectory-recovery-audit/`: protocol, observed SIGKILL, recovery and tamper/repeat rejection evidence.
- `qwen25-template-reproduction.json`: original/corrected rendered-example checks and unchanged snapshot hash.
- `readiness-reconciliation.json`, `readiness-hash-audit-with-history.json`, `pr129-replay-audit.json`: task-level readiness limits and retained-record checks.
- `live-github-state.json`, `integration-path.json`: initial live state and dependency analysis.
- `final-integration-tests.xml`, `final-ruff.log`: verification evidence.

From a checkout of this branch, the model evidence audit requires no model loading:

```sh
python scripts/audit_vare_development_campaign_v1.py \
  --outputs results/vare-development-repair-v1 --output /absolute/path/to/new-audit.json
```

Frozen protocols precede their model outcomes at commits `6c2718e1` (Metal), `be48c47e` (Qwen2.5 CPU), and `b10948cd` (Qwen3 CPU). The Git bundle/patch and reproduction notes preserve those identities. Checksums and Git commits are integrity records, not external signatures or independent human approval.
