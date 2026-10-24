# Contaminated verifier rewards shift a scripted repair policy toward invalid delegates

**Date:** 2026-10-24
**Decision:** GO for a narrow policy-selection mechanism result. NO-GO for an LLM capability, held-out-task, cross-benchmark prevalence, asynchronous-systems, or recursive-improvement claim.

## Frozen question and scope

The frozen hypothesis was that a trainable policy over six fixed repair candidates, trained on the contaminated MLX-LM task image, would select the two no-fix dependency delegates more often than the same policy trained on the sanitized image, when candidate actions were regraded by the sanitized trusted evaluator. The protocol and candidate bytes were frozen before any new controller grading. Protocol SHA-256: `267496f87f488259017cd697a98acf43ad722c98c906e4a126f4082903819064`.

This is a **single previously consumed Development task** (`mlxlm-1912-top-p-min-token`) and a scripted categorical controller. The policy has six scalar logits and sees no action labels or code; it samples among fixed action indices and receives verifier rewards. It is not a language-model update, a model-discovered attack, a held-out-task evaluation, or evidence about contamination prevalence across benchmarks. The trusted evaluation uses new sampled actions and the same 14-case task oracle in the fully sanitized image; it does not make the task itself held out.

The result tests a causal mechanism: whether false executable rewards can move an actual trainable policy toward known-invalid actions, and whether a clean grader detects those actions. It does not establish that a coding model will discover these candidates or that this effect generalizes to other repositories.

## Design and budgets

The protocol used eight paired seeds (`11, 23, 37, 53, 71, 97, 131, 173`) and four conditions. Each condition-seed pair collected 20 training grades in five batches of four, made five Adam step calls, then sampled 50 actions that were all graded against the sanitized image. Three arms used Adam at learning rate `0.08`; the frozen-policy arm used learning rate zero but retained the same five step calls.

| Condition | Training reward source | Role |
| --- | --- | --- |
| Contaminated feedback | Original-image pass/fail | Treatment |
| Sanitized feedback | Fully sanitized-image pass/fail | Primary control |
| Reward shuffle | Original-image pass/fail, shuffled across action identities within each four-action batch | Action-reward alignment control |
| Frozen policy | Original-image grades, zero learning rate | Optimizer/update control |

The action set contained baseline, the known fix, two incorrect mutants, and source/bytecode dependency delegates. Candidate, protocol, grader, adapter, repository revision, and both image digests are locked in the run evidence. The evaluation process received only a boolean from the isolated grader. The primary metric was the per-seed share of the 50 sanitized-evaluation actions selecting a delegate that the sanitized grader rejects. The primary contrast was contaminated feedback minus sanitized feedback. Three paired contrasts were predeclared and adjusted with Holm correction; the eight seed pairs, rather than individual actions, are the inferential units.

Budgets were 2,240 grader invocations, 31,360 candidate-case executions, 160 optimizer step calls (120 positive-learning-rate and 40 zero-learning-rate), and 2,400 seconds. No model or network calls and no accelerator computation were used.

## Results

### Contaminated rewards and trusted outcomes

Across the three conditions trained against the original image, the two delegates were selected 156 times and passed the contaminated grader all 156 times. Those candidates are invalid: each fails in the sanitized environment. The contaminated-image training arms therefore recorded **156 false acceptances in 480 grader calls (32.5%)**. This denominator is repeated action sampling on one task, not 480 independent tasks or a benchmark prevalence estimate.

| Training condition | Training grades | Direct-delegate selections | Contaminated raw delegate passes | Direct-delegate samples assigned reward 1 | Sanitized evaluation invalid delegates | Sanitized evaluation task passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contaminated feedback | 160 | 53 | 53 | 53 | 152/400 (38.00%) | 94/400 (23.50%) |
| Sanitized feedback | 160 | 49 | 0 | 0 | 120/400 (30.00%) | 104/400 (26.00%) |
| Reward shuffle | 160 | 51 | 51 | 32 | 119/400 (29.75%) | 69/400 (17.25%) |
| Frozen policy | 160 | 52 | 52 | 52 | 127/400 (31.75%) | 70/400 (17.50%) |

The reward-shuffle column counts direct-delegate samples that received a positive **shuffled** training reward, not their raw verifier outcome. Its 51 direct-delegate grades passed the contaminated verifier, but only 32 of those action positions were assigned reward one after the frozen within-batch permutation.

On the sanitized training arm, the grader accepted the known fix in all 29 sampled attempts and rejected all 49 delegate attempts. Across all four arms, the known fix passed all 106 sampled training attempts. On the 1,600 sanitized evaluation calls, the two delegates were selected 518 times and rejected 518 times; the only passing candidate was the known fix (337/337). There were **zero clean-oracle false acceptances** in evaluation. Baseline and both mutants were rejected in all sampled training and evaluation grades.

The artifact-specific split was consistent: in contaminated-feedback training, source delegation passed 28/28 and bytecode delegation 25/25; in sanitized-feedback training, those actions passed 0/28 and 0/21. The sanitized evaluator rejected every sampled source and bytecode delegate. This tests both fixed controls, but does not estimate how often a real agent would choose either route.

### Policy selection and paired comparisons

The contaminated-feedback policy ended with mean probability `0.4044` on the two invalid delegates, versus `0.3162` under sanitized feedback, `0.3314` under reward shuffling, and `0.3333` in the frozen policy. The main measured effect is the sampled action distribution under the independent clean oracle:

| Preregistered contrast on invalid-delegate share | Mean paired difference | Descriptive paired 95% t interval | Exact two-sided sign-flip p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| Contaminated − sanitized | +0.0800 | [+0.0532, +0.1068] | 0.0078125 | 0.0234375 |
| Contaminated − reward shuffle | +0.0825 | [+0.0617, +0.1033] | 0.0078125 | 0.0234375 |
| Contaminated − frozen policy | +0.0625 | [+0.0365, +0.0885] | 0.0078125 | 0.0234375 |

All eight paired seed differences were positive in each planned contrast. The estimate is an eight-seed controller effect on one task, not a task-level population effect. The paired t intervals are descriptive; the exact sign-flip tests with Holm adjustment are the preregistered inference.

Clean task-pass share was 23.50% for contaminated feedback and 26.00% for sanitized feedback. This secondary outcome was not the primary inferential target. Its exploratory paired difference was −2.50 percentage points (descriptive paired t interval −4.45 to −0.55 points; exact sign-flip `p=0.0625`, unadjusted). It is directionally consistent with the invalid-action shift, but this small, single-task result does not establish a general task-success reduction.

### Optimizer and execution accounting

The runner made **160 Adam step calls** and recorded **117 parameter changes**: 40 in contaminated feedback, 37 in sanitized feedback, 40 in reward shuffle, and zero in the frozen control. The frozen arm retained all five calls per seed but had zero parameter changes. Checkpoints were written and loaded with PyTorch's restricted `weights_only=True` loader; all 32 final condition-seed checkpoints are present in the run artifact.

The run completed in 1,647.3 seconds (0.458 wall-hours): 4,895 grader calls and 68,536 candidate-case executions per wall-hour. Host available memory was 28.89% at run preflight and 33.13% at completion; swap stayed at 2,251.75 MiB, below the frozen 2,800 MiB abort threshold. These are aggregate task-grader rates, not accelerator or model-training throughput.

## Failed attempt and recovery record

The runner summary's `training_false_reward_count=53` covers the contaminated-feedback treatment arm. The raw training-event audit additionally counts 51 and 52 raw false passes in the reward-shuffle and frozen controls, respectively; these control observations are not included in that treatment-only summary field.

The first execution (`run1`) completed contaminated-feedback seed 11: 20 training grades and five optimizer calls, then stopped before any evaluation grade. PyTorch 2.9.1's restricted checkpoint loader rejected its own serialized `TorchVersion` metadata. This negative outcome is preserved byte-for-byte as 21 manifested artifacts; its original evidence-manifest SHA-256 is `4d599bbce1c38f184c8ae9b22f9f62425f8c90845be9a3864fc310bfa8c5231a`, and its checkpoint SHA-256 is `26f09e6fbf77add8e61e0af229f56e3725a9fd497a5584dc5c4f0af6c385e4f3`.

A pre-evaluation erratum allowlisted only PyTorch's `TorchVersion` type inside the restricted safe-global loader while retaining `weights_only=True`; checkpoint serialization itself was unchanged. It did not change the task, images, candidates, reward, seeds, budget, optimizer, metric, evaluator, or analysis. A zero-grader resume audit rechecked every run1 artifact, the failure record, all 20 training events, five optimizer events, and exact checkpoint hash. The continuation used that checkpoint and did not repeat its grades or optimizer steps. The first 20 observations came from implementation commit `0fec161b`; the 2,220-call continuation and final runner snapshot came from `87f6129e`. The original run1 tree is copied byte-exact under run2's `resumed-prefix/run1`.

Verifier uncertainty was not separately modeled: this fixed candidate set was scored by the isolated deterministic pass/fail grader. The intervals quantify controller-seed variation only. The run was synchronous; its checkpoint-load recovery is not an asynchronous queue, stale-rollout, or recovery-consistency benchmark.

## Evidence and reproduction

The protocol SHA is `267496f87f488259017cd697a98acf43ad722c98c906e4a126f4082903819064`. The collected run has 77 original artifacts, including per-grade reports, action sequences `0..2239`, optimizer traces, checkpoints, the failed-run prefix, telemetry, and summary. To comply with the repository's 1 MiB tracked-blob limit, the two event streams were gzip-compressed losslessly after collection. The exact original manifest and lock are preserved as `evidence-manifest.as-collected.json` and `evidence-lock.as-collected.json`; their original manifest SHA-256 is `5631d893209f0ad55623f2db2908e8d5df321e4679d8dcca8b289373c12c2ac8`. The current 80-artifact packaging manifest has SHA-256 `67363a06ba21f514e0024a31ee8b4afd88475eb107f63317652091e3ebdb5b72`. `event-stream-repackaging.json` records the exact original byte counts and SHA-256 values, and decompression reproduces the event bytes exactly. The uncompressed source artifacts remain retrievable from Git commit `96bcc308ab9e521a30e27c6bc9a04842efca29e1`. No grades or outcome-bearing records were changed. The run contains 640 training grades, 1,600 sanitized evaluation grades, 160 optimizer records, and no model calls.

Verify the frozen run artifact from the repository root:

```bash
python - <<'PY'
import gzip, hashlib, json, pathlib
root = pathlib.Path("results/verifier_integrity_policy_gradient_v1/run2")
manifest = json.loads((root / "evidence-manifest.json").read_text())
lock = json.loads((root / "evidence-lock.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert len(manifest["artifacts"]) == manifest["artifact_count"] == 80
assert all((root / row["path"]).stat().st_size == row["bytes"]
           and sha(root / row["path"]) == row["sha256"]
           for row in manifest["artifacts"])
assert sha(root / "evidence-manifest.json") == lock["manifest_sha256"]
source_manifest_path = root / "evidence-manifest.as-collected.json"
source_lock_path = root / "evidence-lock.as-collected.json"
assert sha(source_manifest_path) == lock["source_manifest_sha256"]
assert sha(source_lock_path) == lock["source_lock_sha256"]
source_manifest = json.loads(source_manifest_path.read_text())
source_lock = json.loads(source_lock_path.read_text())
assert source_lock["manifest_sha256"] == lock["source_manifest_sha256"]
packaging_path = root / "event-stream-repackaging.json"
assert sha(packaging_path) == lock["stream_repackaging_sha256"]
packaging = json.loads(packaging_path.read_text())
stream_bytes = {}
for row in packaging["streams"]:
    content = gzip.decompress((root / row["archive_path"]).read_bytes())
    original = next(x for x in source_manifest["artifacts"]
                    if x["path"] == row["source_path"])
    assert len(content) == row["source_bytes"] == original["bytes"]
    assert hashlib.sha256(content).hexdigest() == row["source_sha256"] == original["sha256"]
    stream_bytes[row["source_path"]] = content
summary = json.loads((root / "summary.json").read_text())
assert summary["actual_grader_invocations"] == 2240
assert summary["optimizer_step_calls"] == 160
training = [json.loads(x) for x in stream_bytes["training-events.jsonl"].splitlines()]
evaluation = [json.loads(x) for x in stream_bytes["evaluation-events.jsonl"].splitlines()]
sequences = sorted([x["grade"]["sequence"] for x in training + evaluation])
assert len(training) == 640 and len(evaluation) == 1600
assert sequences == list(range(2240))
print("PASS: 80/80 packaged artifacts, original event-stream bytes, and frozen totals")
PY
```

Key files: [protocol](../protocols/verifier_integrity_policy_gradient_v1.json), [protocol lock](../protocols/verifier_integrity_policy_gradient_v1.lock.json), [pre-evaluation recovery erratum](../protocols/verifier_integrity_policy_gradient_v1.pre-evaluation-erratum.json), [runner](../scripts/run_verifier_integrity_policy_gradient_v1.py), [preserved failed run](../results/verifier_integrity_policy_gradient_v1/run1/), and [completed, losslessly packaged run](../results/verifier_integrity_policy_gradient_v1/run2/).

## Literature and novelty boundary

The verifier-integrity report for the transitive-reference pilot compares this result against [SWE-Bench+](https://arxiv.org/abs/2410.06992), [Unearned Passes](https://arxiv.org/abs/2609.34262), [Hacker-Fixer](https://arxiv.org/abs/2606.08960), [Scalable Weak-to-Strong Verifier Hardening](https://arxiv.org/abs/2610.15924), [SWE-Bench Pro V2](https://arxiv.org/abs/2610.24206), and [SWE-Bench Pro Verified](https://arxiv.org/abs/2609.08149). These are close precedents, not background analogies. SWE-Bench Pro V2 reports forged dependency-cache state, grades patches in a fresh task image, and removes `__pycache__`/`.pyc` artifacts; it also describes cache state outside a patch affecting grades. Unearned Passes studies protected-reference channels including Git history and reruns after repair. Hacker-Fixer and the newer weak-to-strong hardening work use model-driven exploit discovery and verifier repair at much larger scale. SWE-Bench+ addresses solution leakage through task text. Together these papers already establish that leaked references and mutable dependency/verifier state can produce unearned rewards and corrupt training/evaluation signals.

The bounded result is a controlled replication of a reward-to-policy effect: in a fixed candidate-selection controller, repeated false passes for known-invalid source/bytecode delegates increased their selection under clean regrading relative to sanitized feedback, shuffled feedback, and a frozen policy. The addition here is the paired, multi-seed measurement on these fixed counterfactuals, not discovery of the general mechanism. This was a scripted action policy with no model discovery and only one consumed task; it is not a publication-level novelty or general hardening-method claim.

## Decision

- **GO:** retain this as a verified, preregistered one-task controller mechanism result and as evidence that contaminated executable reward can steer a policy toward invalid installed-repair delegates.
- **NO-GO for novelty claims:** current benchmark-hardening literature already reports closely related dependency-cache and clean-container findings. Present this as a small controlled policy-response measurement, not a new contamination or verifier-hardening discovery.
- **NO-GO:** do not claim improved coding-model capability, held-out-task generalization, cross-benchmark contamination prevalence, or recursive self-improvement. No trained language model, fresh task split, or 15-task learning eligibility gate was involved.
- **NO-GO for Experiment C this cycle:** A and B met the mechanism trigger, but there is no fresh task that is both independently qualified and audited for a contaminated-versus-clean environment pair. The manual queue's other Development candidates remain unqualified with prior use marked unknown. The cached Mistral-7B is available locally, but its four frozen attempts on the already-consumed tasks produced 0/4 independently correct repairs. Do not spend another attempt by prompt tuning or reusing those tasks.
- **NO-GO for async/RSI expansion in this cycle:** a scripted controller result does not validate a real model-coupled rollout/update pipeline. First obtain fresh verified model trajectories and demonstrate a real student update under independent task-level evaluation. The existing four zero-success Mistral-7B attempts remain negative evidence; no SFT or GRPO was launched.

The next justified step is to qualify a small, task-diverse Development set with known exposure history, then audit installed source/bytecode provenance and genuine-fix acceptance under a preregistered selection rule. Only after that gate passes should another model-generated run or asynchronous systems experiment be considered.
