# Fresh post-training candidate qualification — 2026-10-16

**NO-GO before base inference.** This bounded screen did not identify a novel,
qualified model/task/grader combination. No model was loaded, no optimizer ran,
and no No-update/SFT/RL comparison was started. This is a candidate-selection
decision, not a failed learning experiment or a claim that no useful method exists.

## Repository and prior decisions

The remote main was fetched and pinned to
`467ff55dbf6accf0a59e1f0c7f5741ac53ce21de`. Its
[AGENTS.md research gate](https://github.com/mitukx/VARE/blob/467ff55dbf6accf0a59e1f0c7f5741ac53ce21de/AGENTS.md#L68-L77)
requires a materially distinct candidate before another base-only screen, then
fresh-task success, update/save/reload/resource qualification before a separate
multi-seed comparison. The [2026-10-15 qualification](research-candidate-qualification-2026-10-15.md),
[Stage 0 gate](stage0-research-gate-2026-10-11.md), retirement ledger, README,
roadmap and relevant base-gate reports were reviewed. Their hashes are recorded
in [the capture](../results/research-candidate-screen-2026-10-16/capture-final.json).

The existing toy-model systems result does not qualify a task-success study.
OpenBookQA, GSM8K/ASDiv, exact-integer, SVAMP and earlier code-repair/ticket
pairings remain retired or stopped. Neither their confirmation data nor their
model responses were loaded for this screen. The earlier generic IFBench RLVR
proposal was already [rejected](next-study-decision-2026-10-09.md#current-decision--do-not-run-a-generic-ifbench-rlvr-screen).

## Candidate methods and closest primary work

Searches covered SQL counterexamples/execution rewards, code mutation and
coder/tester co-training, and instruction/prompt robustness, including a query
for October 2026 releases. Paper versions and public upstream source were
checked as of 2026-10-16. Retrieval recency filters are not completeness guarantees.
Prior-art results below are author reports, not independent reproductions.

| Candidate and falsifiable outcome | Exact overlap and remaining distinction | Gate decision |
|---|---|---|
| **SQL counterexample reward.** Give a query reward only when it agrees with the reference on every training-only counterexample database. Test whether this improves protected SQL task success over ordinary single-database RL at equal execution/token budgets. | This is an all-tests execution reward. [CoRE-DB](https://huggingface.co/datasets/thanhdath/CoRE-DB-SQL-R1/tree/d7c1de45b9f38d6c86de3b9ed3dd1503b208ae68) already constructs reward databases from gold-query mutations and separating database instances. [ReViSQL](https://github.com/uiuc-kang-lab/ReViSQL/blob/9fac371aa22019e9912dcbd6572e8fe8194d352a/README.md) already down-weights coincidental execution matches using formal equivalence. [DualSQL](https://arxiv.org/html/2609.18135v1) has REX to handle result-equivalence errors; REX is result matching, **not** counterexample-database generation. Current-policy-conditioned mining would be a different sampling choice, but no distinct estimator or claim beyond existing counterexample selection was established. | **No-go on method qualification.** An unused SQLite task source was found, but its exact independent semantic grader and disjoint training composition are unqualified. A smaller-model port would be a reproduction. |
| **Mutation-conditioned coder/tester RL.** Validate generated tests against reference behavior, reward incremental mutant detection, and use those tests as the solver's training signal. Require higher independently graded full-program pass@1 at equal rollout/test budgets. | [MIST-RL](https://arxiv.org/html/2603.01409v1) already optimizes incremental mutation utility. [TCS](https://arxiv.org/html/2609.03955v1) uses a current-policy buffer and reference-valid counterexamples; importantly, its solver rewards still use dataset tests. Feeding generated tests into solver training is therefore a real distinction from TCS alone, but [CURE](https://arxiv.org/abs/2506.03136v1) and [ZeroCoder](https://arxiv.org/html/2604.07864v1) already co-train coder/tester through execution interactions, with ZeroCoder adding mutation-based discriminativeness and calibration. | **No-go on novelty.** The proposed combination has no established new objective or correction beyond these methods. No independent training/evaluation pack or M1 update budget was qualified. |
| **Prompt-orbit/constraint-aware RL.** Optimize worst-case success across certified equivalent instruction variants; use fixed per-constraint anchors when groups are uniform. Require better success on untouched constraint compositions without clean-task regression. | [PPCL](https://arxiv.org/abs/2402.15833v1) regularizes clean/perturbed losses; [IFBench](https://arxiv.org/abs/2507.02833v1) already studies RLVR generalization to new constraints; [MDP-GRPO](https://arxiv.org/abs/2606.06058v1) uses multi-temperature sampling and dual anchors for low-dispersion groups. [SPrPO](https://arxiv.org/html/2609.34064v1) concerns internal policy perturbations, not certified prompt equivalence, so it does not by itself establish equivalence to this proposal. The remaining combination lacks a distinct justified update/estimand and certified transformation set. | **No-go.** Do not reopen the stopped generic IFBench line or invent prompt variants to obtain a passing signal. A new checkpoint does not resolve method or grader independence. |

[Exploring Pass-Rate Reward](https://arxiv.org/abs/2605.02944v1) is an additional
warning against treating dense partial-test rewards as sufficient evidence of a
useful full-correctness gradient. The small-model
[Qwen3-0.6B/Llama3.2-1B verification-feedback study](https://arxiv.org/abs/2605.30478v1)
also already compares unit tests, lint shaping and combined rewards. Porting
those comparisons is not a novel VARE method. These papers do not prove a new
candidate would fail; they remove the justification for launching a generic port.

## Unused candidates and evaluator qualification

The exact strings `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3-0.6B`,
`BIRD-CRITIC` and `BigCodeBench` have zero occurrences in the pinned main's
tracked docs/protocol/config/requirements/script/benchmark metadata.
This is a repository-scoped screen, **not proof of absence from every private
run, pretraining corpus or sibling project**. No candidate has been selected
for model execution. Published model-card scores are not local base measurements.

| Candidate | Pinned revision / observed metadata | Remaining gate |
|---|---|---|
| SmolLM3-3B | `a07cc9a04f16550a088caea529712d1d335b0ac1`; public, Apache-2.0; `SmolLM3ForCausalLM` | New model/task LoRA or full-update memory, time and save/reload correctness unmeasured. |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca`; public, Apache-2.0; `Qwen3ForCausalLM` | Same unmeasured gates; known small-model RLVR results do not establish a new method or this pairing's base performance. |
| BIRD-Critic-SQLite | `af2b1c3e2f8b3480e5569212198761c770df06d0`; public, CC-BY-SA-4.0; [official source](https://github.com/bird-bench/BIRD-CRITIC-1/tree/f408d6c9db9f16e75b44a1e23f22c04579d1f390) | No task rows/databases were fetched. Need separate training tasks and protected confirmation, correct-fix/wrong-fix controls and exact semantic grader validation. |
| BigCodeBench | `b74c0d0bf70d2c0bc459be537895cca163007f1a`; public, Apache-2.0; [evaluator](https://github.com/bigcode-project/bigcodebench/blob/09dd993f46c3fbf3a799465bb96d524edcb0b199/bigcodebench/evaluate.py) | No task rows were fetched. A file called `train` does not establish a legitimate disjoint training pool; reference/test assets must stay outside optimization and selection. |
| IFBench test | `2e8a48de45ff3bf41242f927254ca81b59ca3ae2`; public; Hub license field absent; [code](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d) | The method line was previously rejected. Dataset licensing, independent training source and transformation certification are not inferred from checker code availability. |

The SQLite evaluator's unordered
[`ex_base` branch](https://github.com/bird-bench/BIRD-CRITIC-1/blob/f408d6c9db9f16e75b44a1e23f22c04579d1f390/evaluation/src/sqlite_test_utils.py#L225-L228)
compares sets of rows. A helper also removes DISTINCT; task-specific tests may
add checks. This is a concrete reason to validate multiplicity, NULL, order,
empty outputs and side effects before calling it an independent exact-semantic
grader. It is **not** an executed benchmark-defect finding: no task-level grader
control was run, and no benchmark outcome is inferred from that branch alone.
The [distilled SQL test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval/tree/e97acc546ecbee8fa27fa8dbf025ef61493a876c)
was inspected as an existing alternative; no replacement grader was implemented.

CoRE-DB is prior art rather than a selected training source. Its public metadata
requires manual access approval; none was requested. ReViSQL's released
Tinker/API training path was not used. No paid service or cloud run was invoked.

## Measurements and decision

The host remains M1 Pro / 32 GiB / 10 cores. The isolated Python 3.11.15
environment has PyTorch 2.8.0 and Transformers 4.56.2; these were checked via
package metadata without loading a model. A functioning installed library is
not an update-cost qualification, and tiny-GPT-2 timings cannot be extrapolated
to either proposed checkpoint.

| Item | Observation |
|---|---|
| First read-only capture | 25 requests, 8.22 s, two retained fetch errors |
| Corrected capture | 34 requests, 9.77 s, zero fetch errors |
| Base performance / mixed full-success groups | **Not measured** |
| Update/save/reload time / model peak memory | **Not measured** |
| Independent grader control pass rate | **Not measured** |
| Model inference / candidate optimizer calls | **0 / 0** |
| No-update/SFT/RL arms / study seeds | **Not started / not selected** |

The first capture used an incorrect IFBench Hub identifier (HTTP 401) and
assumed `main` for the test-suite repository (HTTP 422). The corrected run uses
the official `IFBench_test` identifier and the repository's `master` branch.
Both attempts and their source hashes remain recorded. These are metadata
retrieval corrections, not model failures or performance replications.

**Stop before base inference.** Novelty and independent task/evaluator
qualification have not passed, so starting a nearby base-only screen would
contradict the charter and the requested gate. No outcome-bearing learning
protocol was created. Reopening requires a defined difference from the closest
methods, an unused legal training/dev/confirmation composition, and a validated
independent grader. Only then freeze base success and mixed-reward thresholds
and abort limits before inference. A passing base gate would authorize a
separate bounded update/save/reload smoke; only passing all gates would admit
the preregistered matched No-update/SFT/RL multi-seed comparison.

## Reproduction and retained evidence

The [one-off metadata collector](../reproducers/research_candidate_metadata_20261016.py)
uses only the standard library, refuses an existing output directory, downloads
public metadata/evaluator text without executing it, and never requests weights,
dataset payloads or database files. Full third-party response bodies stay in the
local capture; the public [request ledgers](../results/research-candidate-screen-2026-10-16/)
retain observed status, elapsed retrieval time, response SHA-256, immutable
source revisions, factual model/dataset fields and repository-ledger hashes.
Paper text is not redistributed. Browser review also checked the versioned
CURE, ZeroCoder and MDP-GRPO sources linked above; those are additional primary
references, outside the collector's eight bibliographic captures.

```bash
python3 reproducers/research_candidate_metadata_20261016.py --output /tmp/vare-research-metadata-new
python3 reproducers/research_candidate_metadata_20261016.py --verify /tmp/vare-research-metadata-new
```

New captures may observe changed public metadata; retained response hashes
describe the original observation, not a promise that mutable endpoints stay
byte-identical. The [decision record](../results/research-candidate-screen-2026-10-16/decision.json)
uses `null` for unmeasured model quantities. Hash/inventory verification, a
deliberate modified-byte negative control, Ruff and blob-budget checks passed.
No model regression suite was needed because no model/runtime/trainer changed.
Review follow-up found that a re-sealed capture could change
`repository_revision` and still pass verification. The verifier now binds this
field to the pinned source revision. The re-sealed wrong-revision control failed
in ordinary Python, `python -O`, and `PYTHONOPTIMIZE=1`; the corrected 27-case
integrity suite passes. Baseline and corrected logs are retained under
`results/research-candidate-metadata-review-v1/`.
Frozen prior protocols and results are unchanged. This bounded search does not
establish novelty absence globally, independent external review, task-success
improvement, or new training-system capability.
