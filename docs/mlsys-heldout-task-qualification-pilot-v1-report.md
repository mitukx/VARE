# Held-out task qualification pilots — 2026-10-25

## Decision

**NO-GO for model learning and held-out capability evaluation.** Three original held-out task families now have complete grader certificates within the audited VARE scope. One additional family failed qualification. The existing gate still requires 15 eligible independent held-out tasks; 12 more are needed. No model inference, optimizer update, or training was run in this cycle.

This is an E1 task-readiness result. It establishes bounded, isolated regression checks for three source-pinned tasks. It does not establish agent performance, model generalization, a newly blind 15-task pool, or a capability improvement.

## Frozen pool results

The original task allocation and historical records are unchanged. The three successful tasks were selected from that allocation using source/test feasibility and exposure records, before any candidate model or learning agent accessed them. Known fixes were grader calibration controls only.

| Original held-out family | Status | Baseline | Known fix | Incorrect mutant | Current blocker or evidence |
| --- | --- | --- | --- | --- | --- |
| `transformers-48319-stale-best-checkpoint` | `UNASSESSED` | — | — | — | No complete certificate in this cycle. |
| `transformers-49208-resume-metrics` | `UNASSESSED` | — | — | — | No complete certificate in this cycle. |
| `transformers-49412-fusion-idempotence` | `RETIRED_EXPOSED` | — | — | — | Nine retained Qwen3-4B Development attempts; not eligible for fresh held-out use. |
| `mlxlm-1902-continuation-score-truncation` | `UNQUALIFIED` | FAIL | FAIL | FAIL | All three variants missed the parent-owned checks; the known-fix false negative is unresolved and was not tuned or rerun. |
| `mlxlm-1921-padded-logprobs` | `UNASSESSED` | — | — | — | No complete certificate in this cycle. |
| `mlxlm-1926-falcon-cache-gradient` | `CERTIFIED_WITHIN_AUDITED_VARE_SCOPE` | FAIL | PASS | FAIL | Baseline, exact target-only fix, mutant, patch-admission controls, and shared isolation boundary passed their declared gates. |
| `mlxlm-1935-zero-max-tokens` | `UNQUALIFIED_NOT_RUN` | — | — | — | Static preflight found the pinned fixture would download model weights; no task grader was run. |
| `mlxlm-1938-phi3small-logits-mask` | `CERTIFIED_WITHIN_AUDITED_VARE_SCOPE` | FAIL | PASS | FAIL | Same declared gates passed. Two completed variants from an interrupted attempt were hash-checked and reused; the failed attempt remains preserved. |
| `mlxlm-1979-ssm-padding-mask` | `CERTIFIED_WITHIN_AUDITED_VARE_SCOPE` | FAIL | PASS | FAIL | Right-padded and unpadded valid-token outputs and final state matched under the pinned 1e-4 tolerance. |

Across four fully attempted task families, **3/4 certificate gates passed** and **1/4 failed**. All three successful tasks had baseline failure, known-fix acceptance, and incorrect-mutant rejection. This is task qualification, not agent success data. Of the eight initially surviving reserved families, three are now certified, one was already retired for exposure, and five remain unqualified or unassessed. At least seven additional independent families outside the original pool are still needed even if all five remaining original candidates qualify.

The 24-item manual queue remains a shortlist, not an eligible inventory. This cycle added **zero** newly split families from that queue. No Development task was reclassified as Held-out.

## Protocol and isolation

Three hash-locked protocols preserve the original task split and source catalog:

- Pilot v1 selected #1902 and #1926. It retained two no-grader setup failures (incomplete clone blobs and an adapter path-prefix error), then recorded all task variants. The #1902 known fix failed all three checks, so that task remains ineligible.
- Pilot v2 selected #1938. A staging guard initially rejected a mutant identical to the baseline. The original failure and already-completed baseline/fix records were retained; a narrow resume verified their hashes and ran only the mutant and boundary control. It did not repeat prior model or task outcomes.
- Pilot v3 selected #1979. It uses deterministic synthetic arrays and the upstream right-padding equivalence semantics. The incorrect mutant removes the mask from the state-decay term.

Candidate containers run the pinned source-only MLX-LM tree and a call adapter, with no Git history, test file, grader, expected values, task prompt, or network. The container is read-only, uses UID 65534, one CPU, a 1,536 MiB memory cap, and a 30-second call timeout. The shared boundary probe confirmed the grader and Git history were absent, the subject mount was read-only, and network access failed.

Pinned execution image: `vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7` (Linux/ARM64, 131,913,042 bytes). It already contained Python 3.12.15, MLX 0.32.4, NumPy 2.5.4, and tqdm 4.70.1. `pytest` and `lm-eval` were absent; no packages, model weights, or datasets were downloaded. The tests use synthetic inputs and do not need either package.

## Exposure search and adjudication

The corrected append-only VARE exposure audit examined **146 retained Git refs** and **5,796 local evidence files** in identified model/agent artifact paths. There were no candidate artifact paths in the scanned Git refs and no task-specific matches in the corrected local search. External model pretraining is unknown and is not covered by this audit.

The first #1979 search used an overly broad “SSM near mask” expression. It found four worktree copies of one retained #1978 RotatingKVCache agent record, all with SHA-256 `9d864c781ce4c417db2caaca9e1c317bd2f4e06708a943134ceb395f1ed6b986`. Manual inspection showed the prompt only named `mlx_lm/models/cache.py`; the transcript read that file, made no edit, and ran no tests. It did not mention SSM or padding and did not include #1979’s prompt, test, or fix. The broad scan and its four identical matches are preserved. The corrected matcher is covered by a regression that rejects this adjacent-task false positive and matches explicit #1979/SSM-padding references. The adjudication record is [here](../results/mlsys-heldout-task-qualification-pilot-v3/exposure-adjudication-v2.json); it supersedes the unchanged v1 record, which had an inaccurate timestamp. The audit therefore clears #1979 only within the documented local VARE scope; it does not establish global absence or absence from model pretraining.

## Resource use and reproducibility

| Run | Task graders + boundary calls | Elapsed / runner wall | Host CPU (user + system) | Host peak RSS | Host swap growth | Candidate memory |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot v1, #1902 + #1926 | 13 | 12.88 s | 0.411 + 1.337 s | 32.85 MB | 0 B | 1 sample reported `0 B / 0 B`; unusable. 1,536 MiB enforced cap. |
| Pilot v2, #1938 | 4 unique calls | 129.58 s elapsed; 4.39 s resume-process wall | 0.175 + 0.361 s in resume process | 30.70 MB in resume process | 0 B | No samples; unusable. 1,536 MiB enforced cap. |
| Pilot v3, #1979 | 4 | 5.36 s | 0.211 + 0.526 s | 31.03 MB | 0 B | 1 sample reported `0 B / 0 B`; unusable. 1,536 MiB enforced cap. |
| Corrected exposure audit | — | 141.48 s | 122.607 + 4.235 s | 81.89 MB | Not applicable | Host process measurement. |

The container memory sampler did not provide a valid peak in any pilot; `0 B` is not interpreted as zero consumption. Candidate peak memory is **unknown**, with the 1,536 MiB cgroup limit as the only measured containment bound. Pilot v2 elapsed time includes the preserved interruption and correction interval; its 4.39-second runner wall time covers only the resume process. No swap increase occurred during any pilot.

The independent same-host audit re-read pinned source/test/license blobs, exposure records, task decisions, run manifests, protocol locks, and the original allocation. It returned `AUDIT_PASS`: three qualified, one unqualified. Source, prompt, grader, license, protocol, and raw-record hashes are retained. The audit is reproducible, but it is not an external third-party implementation. The VARE exposure inventory was local to this host; the committed adjudication preserves the #1978 artifact identity and relevant fields, but another host cannot reconstruct the full 5,796-file inventory from this PR alone.

The execution image was already present locally and is pinned by its Docker image ID and digest. This PR does not include an image archive or a tested, digest-pinned build recipe, so exact container execution requires separately obtaining that image. The report's dependency and Docker inspection records identify the executable environment; they do not make the image independently rebuildable.

To replay source, protocol, run-manifest, and gate checks from a full checkout with the pinned upstream commits and image available:

```sh
python scripts/audit_mlsys_heldout_task_qualification_pilots_v1.py \
  --output results/mlsys-heldout-task-qualification-pilot-v3/replay-audit.json
```

That audit output is append-only. To rerun a task grader, use the corresponding frozen protocol and lock with a **new** output directory, for example:

```sh
python scripts/qualify_mlsys_heldout_task_pilot_v1.py \
  --source-repo source/mlx-lm.git \
  --exposure-audit results/mlsys-heldout-task-qualification-pilot-v4-exposure-audit.json \
  --output /tmp/mlxlm-1979-rerun \
  --protocol protocols/mlsys_heldout_task_qualification_pilot_v3.json \
  --protocol-lock protocols/mlsys_heldout_task_qualification_pilot_v3.lock.json
```

## Development-only model evidence

Existing Development evidence does not clear the learning gate. The user-provided four Mistral-7B attempts had zero successful repairs. Earlier Qwen task-agent runs and the Qwen2.5-0.5B LoRA / RVL optimizer smokes establish only previously exposed Development feasibility or update-path behavior; they are not held-out comparisons. No new inference or training was justified by the present 3/15 task inventory. See the retained [7B evaluation](7b-mlsys-independent-evaluation.md), [actual model update readiness report](mlsys-actual-model-update-readiness-v2-report.md), and [RVL CPU update-path report](rvl-cpu-real-model-update-path-v3-report.md).

## PR stack and gates

PR #127 remains Draft and unmerged at `9d471872fdc6449fed8586fcdfe081b161c9e0cd`; it targets the #126 head `c728b9f2ece5dbd1f08878520b0efe5e17ca3b48`. PR #128 remains Draft and unmerged at `7f7d5a4e3963adef298dee7520bcfc36971639d9`, based on #127’s head. The recovery-auditor correction remains in the chain. The latest PR review-thread snapshots show the addressed P1 findings resolved, but no required human approval is recorded. No merge is proposed.

## Evidence index

- [Pilot v1 protocol and lock](../protocols/mlsys_heldout_task_qualification_pilot_v1.json) · [raw run](../results/mlsys-heldout-task-qualification-pilot-v1/run-3/) · [independent audit](../results/mlsys-heldout-task-qualification-pilot-v1/evidence-audit-v3.json)
- [Pilot v2 protocol and lock](../protocols/mlsys_heldout_task_qualification_pilot_v2.json) · [raw run](../results/mlsys-heldout-task-qualification-pilot-v2/run-1/)
- [Pilot v3 protocol and lock](../protocols/mlsys_heldout_task_qualification_pilot_v3.json) · [raw run](../results/mlsys-heldout-task-qualification-pilot-v3/run-1/) · [independent audit](../results/mlsys-heldout-task-qualification-pilot-v3/evidence-audit-v2.json)
- [Initial broad-pattern exposure scan](../results/mlsys-heldout-task-qualification-pilot-v3-exposure-audit.json) · [corrected exposure scan](../results/mlsys-heldout-task-qualification-pilot-v4-exposure-audit.json) · [adjudication record](../results/mlsys-heldout-task-qualification-pilot-v3/exposure-adjudication-v2.json)

## Next decision

**NO-GO for SFT/GRPO, new held-out model evaluation, or an async learning comparison.** The immediate data gate is 3/15; 12 eligible families remain. Continue source/exposure/environment qualification of the unassessed original tasks and prospective candidates from the existing queue. Do not consume any of them with model inference until their certificates, split membership, and learning protocol are complete. The failed #1902 calibration and all earlier failed staging/audit attempts remain part of the record.
