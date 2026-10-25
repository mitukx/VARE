# Coding reward provenance replay v1

**Run date:** 2026-10-25
**Decision:** **GO** for a preregistered, cross-repository reward-provenance consistency study; **NO-GO** for prevalence, coding capability, model-learning, GRPO, or general hardening claims from this result.

## Finding

A coding reward changed while the scoring code and test bundle stayed fixed. On MLX-LM PR [#1912](https://github.com/ml-explore/mlx-lm/pull/1912), an installed-source delegate and an explicit compiled-bytecode delegate each passed all 14 checks in the original grading image despite containing no repair. Replacing only the installed source stopped the source delegate but left the fixed `sample_utils.pyc` available; the compiled delegate still passed. The sanitized image removed target-package bytecode and preserved the known repair's 14/14 acceptance while rejecting both delegates.

The original, source-only, and sanitized images have different executable-environment identities, but the same grader source, adapter, task prompt, and executed test bundle. A cached reward keyed only by scoring-code identity therefore admitted two false positive reward records. Requiring an environment-aware identity and regrading retained the genuine fix and rejected the two delegates.

This is one selected Development task with scripted controls. It measures the mechanism and the reward-admission consequence; it is not a task-population estimate, a model rollout, or a new on-policy run.

## Threat model and boundary

The candidate is expected to implement the task against the pinned baseline. The grading image originally contained the future fixed `mlx_lm/sample_utils.py` source and bytecode. The controls explicitly loaded either installed source or `.pyc` instead of carrying the repair in the submitted target. The candidate container had no network, no grader or expected-output mount, a read-only root, one CPU, 1,536 MiB memory, 64 pids, and UID 65534. Hidden tests were not mounted.

The sanitized image is a trusted **per-target reference environment** for this counterfactual: it carries pinned baseline source, no `mlx_lm` bytecode, a corrected distribution `RECORD`, and no pip/uv wheel cache. It is not a certificate that all transitive packages or lower image layers are free of solution-related material. The same 14-case parent-owned grader was used in all three images, so the treatment is the executable environment, not a change to the scoring rule.

The candidate set was fixed before the original scored run: known failing baseline, pinned upstream fix, two incorrect mutants, source delegation, and bytecode delegation. The five non-fix candidates form negative controls. The delegates are scripted integrity controls, not attacks discovered by a model.

## Independent reproduction from the published base image

I downloaded the PR #122 ARM64 OCI archive and verified its SHA-256 before loading it:

| Artifact | Identity |
| --- | --- |
| Published archive | 131,195,285 bytes; SHA-256 `4723319c19f96703f3671fe1de97818d61bbc028044849889964b1b67cfd655f` |
| Original MLX-LM 0.32.0 image | `sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7` |
| Source-only rebuild | `sha256:2dcf69441110ee643a6f5705d63e96980adbb4e73769afccc505aa6df8b4762e` |
| Fully sanitized rebuild | `sha256:7b00ef0fe1d1912317a88d04ed9bdf00721e65285eca65a892fca48f70fcff25` |

The release archive did not contain the exact source-only and fully sanitized manifests used by the v1 pilot. Both were rebuilt offline from the archive's exact original digest using the frozen Dockerfiles. Their resulting digests exactly matched the prior run. The canonical SHA-256 of the three image-audit objects also matched the frozen run's audit (`bcb741ef34b132e20d2f32d22ee9af1563dfddf1cc5a327bda71707f248cb433`).

All **18 of 18 candidate × image outcomes** matched the earlier run's acceptance decision and per-candidate case count. Case timings are run-specific and are not claimed to match. This reproduction scored 252 fixed candidate-case checks. The candidate and image records are retained under [`results/verifier_integrity_provenance_v1/reproduction-run-1`](../results/verifier_integrity_provenance_v1/reproduction-run-1/).

| Candidate | Original image | Source-only | Fully sanitized |
| --- | ---: | ---: | ---: |
| Baseline | reject (6/14 cases) | reject (6/14) | reject (6/14) |
| Pinned genuine fix | **accept (14/14)** | **accept (14/14)** | **accept (14/14)** |
| Incorrect mutants (2) | 0/2 accepted | 0/2 | 0/2 |
| Installed-source delegate | **false accept (14/14)** | reject (6/14) | reject (6/14) |
| Installed-bytecode delegate | **false accept (14/14)** | **false accept (14/14)** | reject (0/14) |

The source-only audit had the baseline target source SHA but retained the same target `.pyc` SHA as the original image. The probe found the compiled `apply_top_p` implementation, and the fixed bytecode delegate passed all 14 checks. In the sanitized image `mlx_lm_pyc_files` was zero; the bytecode delegate returned no passing cases. The bytecode execution here is explicit: the candidate unmarshals the installed `.pyc`; this does not claim that ordinary Python import would select stale bytecode after a source timestamp change.

## Reward identity and stale admission

Each of 18 retained reward records binds the task baseline commit, candidate target/source digest, grader and adapter digests, an executed-test-bundle digest, image manifest digest, installed source/bytecode audit digest, verifier epoch, per-case decisions, invocation ID, run-completion timestamp, and raw outcome receipt SHA-256. The test bundle digest is computed from the code that defined the 14 actually executed parent-owned cases, adapter, task protocol and task prompt. The historical v4 protocol's separate upstream test-file digest mismatch remains documented and was not used as the scoring oracle.

The verifier epoch is the canonical hash of scorer identity, test-bundle identity, image manifest digest and installed-provenance audit. Scorer and test identities stayed constant; image and provenance identities changed, so the old records became stale. Distribution version strings alone would not detect this transition. The 18 records are in [`reward-provenance-records.jsonl`](../results/verifier_integrity_provenance_v1/replay-analysis-v1/reward-provenance-records.jsonl), with their summary and manifest in the same directory.

The trusted reference score vector for candidate order `[baseline, fix, keep-all, drop-all, source-delegate, bytecode-delegate]` is `[0, 1, 0, 0, 0, 0]`. The contaminated original-image vector is `[0, 1, 0, 0, 1, 1]`.

| Admission rule | New candidate grades | Positive records retained | False positive rewards admitted | Genuine fix falsely rejected | Positive labels sent to optimizer replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reuse by scorer identity only | 0 | 3 | **2/5 invalid candidates (40%)** | 0/1 | 3, including 2 invalid delegates |
| Reject all records after any epoch change | 0 | 0 | 0 | **1/1** | 0 |
| Match provenance and regrade in sanitized image | 6 candidate grades / 84 cases | 1 | **0/5** | 0/1 | 1 genuine fix |

The denominator is the five fixed non-fix candidate controls on this one task, not a sample of coding tasks. Blanket rejection is safe from false admission but discards the only valid positive. Provenance-aware regrading preserves that positive while rejecting the two environment-dependent shortcuts.

### Optimizer boundary and costs

The analysis performs a one-step CPU `torch.optim.SGD` call on a six-logit categorical selector using only the six scalar pass/fail records. No candidate source, prompt, or gold patch text enters that selector. With the stale contaminated vector, the two delegate actions' combined probability mass moved from 33.33% to 36.10% (+2.77 percentage points). With the trusted regraded vector it moved to 32.36% (−0.98 points). Blanket rejection made no optimizer call. This is an offline scripted reward replay using a real PyTorch optimizer API; it is not a VARE/RVL trainer, language-model update, on-policy trajectory, or evidence of learned coding behavior.

The six sanitized candidate-grade invocations took **4.958 seconds summed wall time** (median 0.749 s each) in this warm-cache run. Each was capped at one CPU, so 4.958 candidate-CPU seconds is an upper bound; actual container cgroup CPU time was not retained. The 18-condition runner took 21.19 s wall time including image probes/build/push and orchestration. The two offline image builds took 0.059 s and 0.099 s with local caches warm; that is not a portable build-cost estimate. No model inference or hidden-test execution occurred.

No fixed candidate has behavior-policy token log-probabilities or model/checkpoint identity. Thus, **zero GRPO/on-policy groups are eligible**. The historical fixed repair was graded, but never used as model-visible training text.

## Generalization gate and prior work

This finding is narrower than the established benchmark-integrity literature:

- [SWE-Bench+](https://arxiv.org/abs/2410.06992) measured solution leakage in issue text/comments and weak-test false passes. It does not establish this installed-bytecode mechanism.
- [NIST CAISI's agent-evaluation audit](https://www.nist.gov/caissi/2-examples-cheating-caisis-agent-evaluations) documents solution access through future Git history, local artifacts/configuration, and fetching newer packages to read source. That makes the broad environment-leakage concern prior art.
- [Hacker-Fixer](https://arxiv.org/abs/2606.08960) studies model-discovered verifier exploits and iterative hardening at much greater task scale. VARE's candidate controls were scripted and did not discover exploits.
- A strong independent precedent is [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full): its official report says pristine-image regrading caught a forged Go module checksum and Go module-cache edits that passed in-place but failed on a clean image. Its [public V2 release](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md) publishes per-task verifiers/reference solutions, pinned image identities, a full SHA256 inventory and the fresh-image regrader. I did not rerun those external task trajectories here, so they are not included in VARE's denominators.

The narrow contribution here is the repeated-bytecode counterfactual under a fixed scoring implementation, its genuine-fix and incorrect-mutant controls, and an explicit stale-reward admission replay. Fresh-image regrading and dependency-cache attacks are already described in Pro V2, so this report makes no first-ever, broad-hardening, or publication-level novelty claim.

The external Pro V2 results provide an independently documented second real-repository example with public task/test/image provenance. That clears the bar to justify a larger, preregistered **cross-repository** consistency study, but not a benchmark-wide prevalence claim. That follow-up should reproduce one specific Go module-cache case and compare scorer-only reuse, blanket rejection and provenance-aware regrade under the published pristine-image protocol.

## Artifact integrity and availability

The PR #122 GitHub release archive was externally retrievable and its downloaded bytes matched the pinned digest above. The GitHub API currently reports that release as `immutable: false`; its tag and assets remain mutable references. Enable GitHub release immutability before publishing a new study release: GitHub says immutable releases lock the tag/assets and generate a release attestation, but the setting affects future releases only. For an independent verification path, publish the exact archive and evidence manifest to a versioned archival record (for example, Zenodo), record the version DOI, and sign the archive/OCI digests with a Sigstore bundle. A specific Zenodo record version has its own persistent identifier and fixed files; a new version should be used for later content changes. See [GitHub's immutable-release documentation](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases), [Sigstore blob signing](https://docs.sigstore.dev/cosign/signing/signing_with_blobs/), and [Zenodo versioning](https://help.zenodo.org/docs/deposit/manage-versions/).

For this task, the original contaminated image is independently reconstructable from the published OCI archive; the derived v1 source-only and sanitized images were rebuilt from that image and matched their original digests. The older #1912 and #1971 scored records are retained in the PR source tree and can be checked against the source locks. The historical v7.3 async run directories under `results/verifier_consistent_async_grpo_v7/` remain untracked local evidence on this Mac and are not made public by this change; they were neither modified nor rerun.

The new reproduction's individual result JSON and image audit files are retained under [`reproduction-run-1`](../results/verifier_integrity_provenance_v1/reproduction-run-1/). The reward records and admission output are retained under [`replay-analysis-v1`](../results/verifier_integrity_provenance_v1/replay-analysis-v1/). Hash manifests are integrity checks, not signatures: they do not authenticate who produced a file.

## PR stack state

At the time of this report, the existing stack is open and unmerged, in this order:

| PR | Base → head branch | Exact head | State |
| --- | --- | --- | --- |
| #122 | `main` → `codex/flagship-teacher-gate` | `37b31519818b5dfbf245075cc541d6c8a38b4099` | Ready for review; required review outstanding |
| #123 | `#122` head → `codex/dependency-artifact-pilot` | `a02d6cb64f2531cbe6152266b431d8a466f0bdbc` | Open, checks clean |
| #124 | `#123` head → `codex/transitive-reference-exposure-v1` | `af4bc1f862f2b8b2d0564cd8731b81129c1195ec` | Open, checks clean |
| #125 | `#124` head → `codex/real-agent-interface-v1` | `a476feeed7ee5d711b3741b0a0ffa89f7c29cd76` | Open, checks clean |
| #126 | `#125` head → `codex/verifier-consistent-async-grpo-v1` | `c728b9f2ece5dbd1f08878520b0efe5e17ca3b48` | Open, checks clean |
| #127 | `#126` head → `codex/verifier-drift-multiscenario-v1` | `ecc5570a4e2275ecb78d8367f2bd383fbd197cac` | Draft, checks clean |

The dependency chain is intact. Integration must proceed bottom-up after independent review; #122 still blocks it, and #127 remains draft. This work adds a small follow-up branch based on #127 and does not change those heads or their review state.

## Reproduction

Run on ARM64 with Docker/Colima, the exact source history, and the locked VARE tree. The candidate calls run with network disabled. First download and verify the published image archive:

```bash
mkdir -p work/pr122-published-artifact
gh release download vare-pr122-grading-images-v1 --repo mitukx/VARE \
  --pattern 'grading-images-arm64.tar.gz' \
  --pattern 'grading-images-arm64.tar.gz.sha256' \
  --dir work/pr122-published-artifact
printf '%s  %s\n' \
  4723319c19f96703f3671fe1de97818d61bbc028044849889964b1b67cfd655f \
  work/pr122-published-artifact/grading-images-arm64.tar.gz | shasum -a 256 -c
docker --context colima load --input work/pr122-published-artifact/grading-images-arm64.tar.gz
```

Fetch upstream source history, make sure the Colima Docker context has a local registry at `localhost:5000` for the frozen runner's derived-image pushes, reproduce the six-candidate comparison, then run the offline reward replay:

```bash
git clone https://github.com/ml-explore/mlx-lm.git work/upstream-mlx-lm
git -C work/upstream-mlx-lm fetch --depth=40 origin 87b7b583a697537aa68f47130b40884700b5f55f
python benchmarks/verifier_integrity_pilot_v1/run_pilot.py \
  --protocol-sha256 c209985ec5ae31be26053aacd479f43c31f83349555777b7053e91bae04ee6fb \
  --source-repo work/upstream-mlx-lm \
  --output results/verifier_integrity_provenance_v1/reproduction-run-2
python scripts/run_coding_reward_provenance_replay_v1.py \
  --run-dir results/verifier_integrity_provenance_v1/reproduction-run-2 \
  --historical-dir results/verifier_integrity_pilot_v1/run3 \
  --published-archive work/pr122-published-artifact/grading-images-arm64.tar.gz \
  --output results/verifier_integrity_provenance_v1/replay-analysis-v2
```

The replay script verifies the protocol/implementation locks, published archive checksum, task commits and source hashes, all 18 candidate-image outcomes against frozen results, image audits, and expected clean/contaminated reward vectors before making its single-step offline optimizer calls. Output directories are non-overwriting. Current result hashes are listed in [`replay-analysis-v1/manifest.json`](../results/verifier_integrity_provenance_v1/replay-analysis-v1/manifest.json).
