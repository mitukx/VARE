# Research candidate screen — 2026-10-09

## Decision

**No candidate in this bounded screen supports an original VARE contribution. Stop before implementation.** The strongest new-looking lead, an IW-OPD vLLM sync skipped after resume, is already reported in TRL issue [#7544](https://github.com/huggingface/trl/issues/7544), with the same source predicate, concrete counterexample, and a correct comparison implementation in `GOLDTrainer`. VARE independently checked that the predicate is still present in the pinned current TRL checkout, but this confirms a known issue; it does not make the finding novel.

No external issue or pull request was opened or changed. No fix was copied into VARE. A patch duplicating the public report would add no independent research value and would obscure the actual state of the evidence.

## Repository state and selection constraints

This screen follows [`AGENTS.md`](../AGENTS.md), [`README.md`](../README.md), [`current-gaps.md`](current-gaps.md), and the active decisions in [`next-study-decision-2026-10-09.md`](next-study-decision-2026-10-09.md). The checkout was clean on `main` at `d05e59e` before this report. Prior studies already stop work on the TRL #7206 rollout admission/freshness line, #7249 token normalization, and the top-p support mismatch, and reject the generic partial-audit efficiency direction. Those are not reopened here.

The screen asked whether a recent correctness issue was both consequential and not already disclosed. Public issues and linked development state were inspected before any implementation work. The strongest candidate was then compared against the source at TRL commit `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`, the current local upstream checkout available for inspection. This was a source-level check; no model, vLLM server, or trainer run was available or claimed.

## Candidate screen

| Candidate | Potential impact | Prior-art check | Decision |
| --- | --- | --- | --- |
| IW-OPD resume can skip the first required vLLM weight sync when `distillation_objective="jsd"` and `vllm_sync_frequency > 1` | The sampler can use pre-resume weights until the next periodic sync, making resumed rollouts inconsistent with the restored learner. | TRL [#7544](https://github.com/huggingface/trl/issues/7544) already gives the affected branch, a resumed-step counterexample, and the contrast with `GOLDTrainer`. The issue is open and had no linked PR at inspection. | Confirmed as still present in inspected source, but known. Reject as a VARE novelty claim. |
| AsyncGRPO can omit a reasoning/tool-call turn from the loss after chat-template re-rendering | A multi-turn policy update can train only the final response while excluding the action that invoked the tool. | TRL [#7503](https://github.com/huggingface/trl/issues/7503) includes a Gemma 4 reproducer and traces the mismatch to Transformers' `thinking` versus `reasoning_content` fields; it links Transformers [#49267](https://github.com/huggingface/transformers/issues/49267). | Specific affected model/template and failure path are already public. Reject as known. |
| AsyncGRPO's per-sample stale check can split a forked rollout before group-relative training | Partial groups alter the comparison set used to compute GRPO advantages. | TRL [#7206](https://github.com/huggingface/trl/issues/7206) covers the concern; VARE already has multiple pinned reproducers and stopped candidate fixes in the active decision record. | Known and already investigated; do not reopen. |

These are bounded candidates, not a claim that all RL trainer defects have been searched. The top-p/log-prob support issue [#6789](https://github.com/huggingface/trl/issues/6789) was also checked against VARE's existing stopped source-method audit and excluded as duplicate evidence.

## Strongest candidate: source check and exact counterexample

In the inspected TRL checkout, `IWOPDTrainer` initializes:

```python
self.vllm_sync_frequency = args.vllm_sync_frequency
self._last_vllm_sync_step = -1
```

Before generating with vLLM, it syncs only if:

```python
self.state.global_step != self._last_vllm_sync_step
and self.state.global_step % self.vllm_sync_frequency == 0
```

For `global_step=37`, `_last_vllm_sync_step=-1`, and frequency `5`, the predicate is false because `37 % 5 == 2`. `GOLDTrainer` initializes the last-sync marker to `-frequency` and checks whether at least one full interval has elapsed; for the same state, its predicate is true and it synchronizes before generation.

The source check was run against commit `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`. SHA-256 values were `5d10a9b4a08118f625dc4ecaa056f2362b0f8d656aa1991058f816d22e70fa1a` for `trl/experimental/iw_opd/iw_opd_trainer.py` and `cf20320035b1e7fa895dc058ad5a43ee0bd2b09ff7e212c04c0a10c3e5843a69` for `trl/experimental/gold/gold_trainer.py`. A small independent arithmetic check evaluated the two predicates for the frozen values and returned `IWOPD=False`, `GOLD=True`.

This establishes the control-flow divergence in source and the exact sync decision for one input. It does **not** quantify policy KL, behavior-policy ratio error, loss/gradient change, or task-success impact. The public issue states the remaining reachability condition: this branch is permitted for JSD with vLLM and a sync frequency above one; the IW-OPD objective itself forces frequency one. No production run was attempted.

## Why this is not the requested original result

The reproducer is the same modulo-versus-elapsed-interval counterexample already present in #7544. The proposed repair—use the elapsed-interval predicate and initial marker already used by `GOLDTrainer`, or force a one-time sync before the first resumed generation—is also already implied by that report. Reproducing it locally would be useful as upstream verification but would not satisfy the novelty requirement or establish new scientific knowledge.

The other candidates are likewise already reported or explicitly stopped in VARE. No candidate survived the prior-art gate, so there is no responsible basis for presenting a new mechanism, an upstream-ready original fix, or a model-improvement claim. No trainer code or regression test was added.

## Next decision

**STOP this candidate screen.** Continue the research program only when a concrete issue emerges from a distinct source path and the initial novelty check finds no equivalent paper, issue, PR, or existing VARE record. For any next defect, require a source-pinned path reproducer, an independent objective/gradient oracle, a negative control, and a measured update consequence before proposing a fix. This report is a negative selection result, not evidence of capability gain or a claim of exhaustive literature review.

## Reproduction of the source-level arithmetic check

```python
step, frequency = 37, 5
iwopd_last, gold_last = -1, -frequency
iwopd_sync = step != iwopd_last and step % frequency == 0
gold_sync = step != gold_last and step >= gold_last + frequency
assert (iwopd_sync, gold_sync) == (False, True)
```

The inspected predicates are in the source pinned above; the code block reproduces their boolean decisions but does not instantiate either trainer or synchronize actual weights.
