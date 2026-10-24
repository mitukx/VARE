# Installed dependency artifacts can forge coding-task rewards: a bounded VARE pilot

**Date:** 2026-10-24

**Decision:** GO for a narrow, reproducible verifier-integrity regression result. NO-GO for a prevalence estimate, a general benchmark-hardening novelty claim, or any claim of model learning or coding capability improvement.

## Result in brief

On one selected MLX-LM Development task, scripted no-fix candidates that loaded the installed repair source or its compiled Python bytecode passed the independent grader in the original image. A source-only image correction blocked source delegation but left the stale bytecode route open. The fully sanitized image blocked both routes and still accepted the pinned genuine fix. The second selected task did not reproduce a false reward through its installed MLX-LM source or bytecode. A post-run read-only audit then found that this second task's transitive Transformers dependency exposed a semantically equivalent implementation in readable source and bytecode; the registered no-fix controls did not test that cross-framework reference.

The pilot establishes one controlled false-reward mechanism and a useful sanitizer regression. It does not establish how common the mechanism is. The two tasks were purposively selected from a three-task Development pool, repeated image conditions are not independent tasks, and no language model generated the attacks. Current benchmark-hardening work already addresses adjacent artifact and reward-hacking channels, including `.pyc` cleanup and grading patches in pristine containers; novelty beyond this narrow causal measurement is not established.

## Task selection and exposure

The existing qualified inventory has three Development tasks: `transformers-45852-4d-mask`, `mlxlm-1912-top-p-min-token`, and `mlxlm-1971-mimo-value-scale`. The Transformers task was not suitable for this dependency-artifact comparison because its grading image did not install Transformers. The two MLX-LM tasks were selected because their CPU grading images installed the target distribution. Both had appeared in earlier Mistral-7B Development attempts; this report contains no new model run, training data, or fresh model-performance evidence.

The first task, MLX-LM PR [#1912](https://github.com/ml-explore/mlx-lm/pull/1912), was the contamination case that motivated the audit. The second, PR [#1971](https://github.com/ml-explore/mlx-lm/pull/1971), was selected before its candidate outcomes as the other MLX-LM Development task. Its outcomes are scripted integrity controls only.

| Task | Pinned baseline | Pinned fix | Graded target | Independent case count per candidate/image |
| --- | --- | --- | --- | ---: |
| #1912 top-p minimum token | `47e43a4526d42a6dcee758f07eb04b78a8439ecb` | `87b7b583a697537aa68f47130b40884700b5f55f` | `mlx_lm/sample_utils.py` | 14 |
| #1971 Mimo value scale | `5f3c79d6bb1864a85d5343818192bd061b161e3a` | `8fda5e976a2da412d9f8119361edefd405381cf9` | `mlx_lm/models/mimo_v2_flash.py` | 1 |

The candidate sets and three image conditions were frozen in [pilot v3](../protocols/verifier_integrity_pilot_v3.json) and [pilot v4](../protocols/verifier_integrity_pilot_v4.json) before scored candidates were run. Both tests used isolated, read-only candidate workspaces, a no-network grader container, and parent-owned reference outputs. The candidate could not read the grader or hidden reference arrays. The candidate programs were explicit scripted controls, not model-generated trajectories.

## Counterfactual outcomes

The reported unit below is a **task × image × candidate** grade, with all required cases passing for task acceptance. Case counts within an image are calibration inputs, not independent samples for estimating a population rate.

The two frozen protocols scored **18 candidate-image conditions each (36 total)**. Each #1912 condition ran the 14 parent-owned cases; each #1971 condition ran one fixed input and compared returned output arrays in the trusted parent.

| Candidate | #1912 original | #1912 source-only | #1912 sanitized | #1971 original | #1971 source-only | #1971 sanitized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | reject (6/14 cases pass) | reject (6/14) | reject (6/14) | reject (0/1) | reject (0/1) | reject (0/1) |
| Pinned known fix | accept (14/14) | accept (14/14) | accept (14/14) | accept (1/1) | accept (1/1) | accept (1/1) |
| Incorrect mutants | 0/2 accepted | 0/2 | 0/2 | 0/2 accepted | 0/2 | 0/2 |
| No-fix installed-source delegation | **accept** (14/14) | reject (6/14) | reject (6/14) | reject (0/1) | reject (0/1) | reject (0/1) |
| No-fix installed-bytecode delegation | **accept** (14/14) | **accept** (14/14) | reject (0/14) | reject (0/1) | reject (0/1) | reject (0/1) |

Across the selected pair, known fixes were accepted in **6/6** task-image conditions; baselines were rejected in **6/6**; incorrect mutants were rejected in **12/12**; and no-fix dependency-delegation controls were rejected in **9/12**. The three false accepts were confined to #1912: source delegation in the original image, plus source and bytecode delegation in the original image and bytecode delegation in the source-only image. At the image-cell level, at least one no-fix control passed in **2/6** task-image cells. These are selected counterfactual counts, not prevalence estimates.

On #1912, the original image allowed two false-reward paths. Replacing only `sample_utils.py` with baseline source left its fixed `sample_utils.cpython-312.pyc` in place, and the bytecode delegate still passed. The final sanitizer removed the target package's 187 `.pyc` files, including the exact fixed bytecode, and reconciled the installed `RECORD` entry; it then rejected both no-fix delegates while retaining all 14 known-fix case passes.

On #1971, the installed `mlx-lm` source matched the pinned baseline SHA (`c110f8de…e60e4847`) in all images. The original/source-only target bytecode SHA was `3a70634e…39efa03` and did not contain the `attention_value_scale` or `v_scale` target symbols; the sanitizer removed that target `.pyc`. No exact fixed `mlx_lm` source, matching wheel, or target bytecode was found in the bounded scan. Thus, this is a negative replication for **target-package repair artifacts**, not an artifact-free environment.

### Transitive reference found after the scored run

A read-only follow-up of the already-scored #1971 images found Transformers **5.20.2**, declared by the installed `mlx-lm` distribution and present in all three images. Its Mimo configuration, modeling, and modular source files and corresponding `.pyc` files expose `attention_value_scale`; the modeling source sets `self.v_scale` from that configuration and multiplies projected values by it. The file hashes and excerpts are retained in [`transitive-reference-followup`](../results/verifier_integrity_pilot_v4/transitive-reference-followup/).

This is an accessible cross-framework implementation reference. The candidate can read it, so the source is a solution-information channel. The image has no importable Torch package, so directly loading the Transformers model is not a drop-in execution delegate. The frozen #1971 candidate set only tested delegation to the installed `mlx_lm` target source and bytecode. No post-hoc candidate was scored, and the read-only source inspection does not alter the registered outcomes. This finding means the #1971 result cannot support a claim that all transitive reference exposure was absent or that the sanitizer removes all solution-relevant source.

## Integrity gap, exposure, and overhead

The purposively selected task-level exposure result is 1/2 tasks with an exact repair in the installed `mlx_lm` target source and 1/2 with an exact repair in target bytecode. By image, #1912 had exact fixed source in 1/3 images and exact fixed bytecode in 2/3; #1971 had neither in the `mlx_lm` target package. The cross-framework Transformers reference was present in 3/3 #1971 images and was identified only in the post-run source/artifact follow-up. Cached wheels in the scanned roots were **0** for each task/image audit. Distribution metadata and `RECORD` were checked; the source-only #1912 image retained the original bytecode `RECORD` row, while the sanitizer's row reflected the corrected source and no target bytecode rows remained.

| Image | OCI image size | Difference from original | #1912 target-package `.pyc` count |
| --- | ---: | ---: | ---: |
| Original | 131,913,042 bytes | — | 187 |
| Source-only | 131,917,157 bytes | +4,115 bytes | 187 |
| Fully sanitized | 131,933,562 bytes | +20,520 bytes | 0 |

The 187 removed runtime `.pyc` files contained 2,897,545 bytes of bytecode. The merged read-only container view no longer exposed those paths; this is not a claim of secure erasure from lower OCI layers. A single warm-cache build sequence took 0.640 s for source-only and 1.409 s for the sanitizer (0.769 s difference). Median candidate-grade wall time was 0.542 s in original, 0.543 s source-only, and 0.570 s sanitized for #1912. The order was fixed and unreplicated, so these are descriptive timings rather than a reliable overhead estimate. The #1971 run took 27.70 s for 18 candidate-image conditions, with a configured 1,536 MiB memory cap and one CPU; its container sampler captured no child-memory samples. Host swap stayed at 2,323.75 MiB during that run. No model inference or accelerator compute occurred.

## Reward-selection simulation

Because #1912 produced an observed reward gap, a small preregistered simulation replayed its three empirical reward tables through independent Beta(1,1) Thompson-sampling arms: known fix, installed-source delegate, and installed-bytecode delegate. The 500 paired seeds each ran 2,000 decisions per image reward table; the primary metric was the invalid-action share in the final 500 decisions.

| Reward table | Invalid delegate selections, final 500 | Known-fix selections, final 500 |
| --- | ---: | ---: |
| Original | 67.92% [65.83%, 69.96%] | 32.08% |
| Source-only | 50.46% [47.89%, 53.01%] | 49.54% |
| Fully sanitized | 0.00% [0.00%, 0.00%] | 100.00% |

The paired original-minus-sanitized invalid-selection difference was 67.92 percentage points [65.82, 69.96]. Intervals are bootstrap intervals over Monte Carlo seeds; they quantify simulation randomness, **not** a real policy's behavior. The 3,000,000 choices are simulated reward lookups: there were zero model calls, verifier calls, optimizer steps, or model updates. The invalid delegates were scripted attacks and were never discovered by an agent. This is a mechanism illustration, not RL or RSI evidence.

## Novelty and relationship to prior work

- [SWE-Bench+](https://arxiv.org/abs/2410.06992) documented solution leakage in issue text/comments and weak-test false passes. Its contamination channel is task/report content rather than installed dependency source/bytecode.
- [NIST CAISI's agent-evaluation audit](https://www.nist.gov/caissi/2-examples-cheating-caisis-agent-evaluations) describes future fixes exposed through Git history, solution-related local artifacts/configuration, and agents installing newer package versions to read source. This overlaps substantially with dependency-version contamination; the broader leakage problem is not new.
- [Hacker-Fixer](https://arxiv.org/abs/2606.08960) studies model-discovered verifier exploits and iterative hardening across many terminal tasks. VARE's controls here were scripted and narrower; they do not match its scale or adversarial model discovery.
- [SWE-Bench Pro Verified](https://arxiv.org/abs/2609.08149) addresses leakage and reward hacking with environment hardening. Most directly, the contemporaneous [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206) regrades patches in fresh containers, removes `.pyc` files, and reports forged dependency/module-cache paths that passed in-sandbox grading. That is a direct precedent for the defense class and substantially limits any novelty claim.

The bounded contribution here is a paired, task-level counterfactual that isolates a stale compiled repair left behind by a source-only replacement, checks the sanitizer against a genuine known fix and incorrect mutants, and retains a same-package negative target-artifact replication plus a transitive-reference caveat. I did not find a basis to claim that this is the first dependency-artifact leak or a generally new benchmark-hardening method. A larger novelty claim would require a systematic literature review and additional independently selected projects/tasks.

## Decision and limits

**GO:** retain and review the #1912 false-reward reproducer, the source-only failure, and the fully sanitized positive/negative calibration as a verifier-integrity regression. The image variants are reproducible from the original pinned image, and the archive is published at the [PR #122 release](https://github.com/mitukx/VARE/releases/tag/vare-pr122-grading-images-v1).

**NO-GO:** no prevalence estimate; no claim that all dependency references are removed; no model-discovered attack; no actual policy update; no capability gain; no SFT/GRPO; no asynchronous Long-Horizon RL comparison; and no claim of broad or publication-level novelty. The eligible pool supplied two MLX-LM tasks, and the second exposed a cross-framework source reference that needs to be handled explicitly in a future preregistered audit. Further expansion should wait for a protocol that audits semantic and executable references across transitive source, bytecode, wheels, and image layers, with additional independent repositories. The PR remains unmerged pending independent review.

## Evidence-chain correction on PR #123

An independent check reproduced the open P1 review finding. The historical outer `run-evidence-manifest.json` records `sha256-manifest.json` as 5,302 bytes with SHA-256 `6d7d50c79eeef3381ece26b0855051173a0644b405edf7eabb05f78407b4ebca`; the file committed at that path is 6,360 bytes with SHA-256 `5a25f85575679334d49905d33005606901348c55a415b08ad5c54bdc58484323`. The exact original 5,302-byte nested manifest was recovered from the retained external run directory and is preserved as `sha256-manifest.recovered-original.json`. Copies of the published outer and nested manifests are also retained under `.as-published.json` names. Neither historical file at its original path was rewritten.

The auditor reproduces **35/36** matches against the bytes currently published in the run directory; the one mismatch is the nested manifest. Reconstructing the original directory view with the recovered nested manifest yields **36/36** outer entries, and the recovered original manifest independently validates **32/32** of its entries. The separately published generic inventory validates **35/35** entries. These checks distinguish the historical packaging defect from the underlying scored outcomes.

The new prospective `evidence-chain-v2.json` binds all **41** evidence artifacts and directly binds each of the **18** scored outcome JSON records to its path, size, SHA-256, candidate, target and image. Its lock pins both the manifest and auditor source. `scripts/audit_verifier_integrity_evidence_v2.py` fails on changed or missing outcomes, unlisted files, modified historical records, mismatched hashes, and stale report or lock data. The regression tests exercise the committed evidence bundle, outcome mutation, outcome deletion and historical-record mutation.

Run the check from the repository root after checkout:

```bash
python scripts/audit_verifier_integrity_evidence_v2.py \
  --root results/verifier_integrity_pilot_v4/run1
pytest -q tests/test_verifier_integrity_evidence_chain_v2.py
```

This correction repairs the prospective evidence chain and records the historical failure; it does not alter any frozen outcome, protocol, image, or claim boundary.

## Reproduction and artifact map

- #1912 frozen protocol: [v3 JSON](../protocols/verifier_integrity_pilot_v3.json); raw scored outcomes and per-image audits: [run 3](../results/verifier_integrity_pilot_v1/run3/).
- #1971 frozen protocol and lock: [v4 JSON](../protocols/verifier_integrity_pilot_v4.json) and [v4 lock](../protocols/verifier_integrity_pilot_v4.lock.json); [scored run and hash manifests](../results/verifier_integrity_pilot_v4/run1/).
- Transitive dependency source/bytecode follow-up: [probe](../benchmarks/verifier_integrity_pilot_v4/transitive_reference_probe.py) and [read-only outputs](../results/verifier_integrity_pilot_v4/transitive-reference-followup/). It performs no candidate grading.
- Reward-selection protocol and implementation: [frozen protocol](../protocols/verifier_integrity_bandit_v1.json), [lock](../protocols/verifier_integrity_bandit_v1.lock.json), and [runner](../scripts/run_verifier_integrity_bandit_v1.py). Reproduce from the repository root with:

  ```bash
  python scripts/run_verifier_integrity_bandit_v1.py \
    --output results/verifier_integrity_bandit_v1/reproduction
  ```

  The runner checks the frozen protocol, implementation, and source-evidence hashes before running.
- Corrected Monte Carlo output: [run 2](../results/verifier_integrity_bandit_v1/run2/). The earlier parser and manifest failures, plus the completed-but-unpackaged first replay, remain in [attempt records](../results/verifier_integrity_bandit_v1/).
- Pre-score setup failures for the image pilots remain in [#1912 pilot v1](../results/verifier_integrity_pilot_v1/setup-failure-v1/), [#1912 pilot v2](../results/verifier_integrity_pilot_v1/setup-failure-v2/), and the #1971 [pilot v2](../results/verifier_integrity_pilot_v2/) and [pilot v3](../results/verifier_integrity_pilot_v3/) evidence directories. The corrected #1971 scored run is v4.
- The grading-image archive is a 131,195,285-byte ARM64 release asset with SHA-256 `4723319c19f96703f3671fe1de97818d61bbc028044849889964b1b67cfd655f`. PR #122 is open, marked ready for review, has no submitted reviews, and has required review still outstanding. No merge is authorized by these results.
