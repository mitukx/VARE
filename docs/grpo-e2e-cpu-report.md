# Real-model multi-completion GRPO correctness — CPU

**PASS for the final v5 systems protocol:** 3 seeds × 12 rounds, 36 actual candidate GRPO updates, 27 diagnostic promotions and 9 deliberate rejections. Nine additional optimizer calls exercised post-step rollback and save/restore consistency. This is not a task-capability study, a new learning method, or evidence of recursive improvement.

## What changed

Investigation started from main `467ff55`; PRs #69 and #70 remain open. Existing VARE replay, `CapabilityLoop`, RVL hooks, `HFLocalBackend`, and `HFCausalLMGRPOTrainer` were reused. The older real-trainer group-boundary experiment had stopped before its optimizer gate; its inputs/results were not reused.

| Matched correctness control | Before | Correction / after |
| --- | --- | --- |
| Rollout policy version / queued publication, and verified response versus token generation | All 6 targeted regressions failed on main; a real-model corrupted response reached an optimizer update. | Bind policy ID/version under the model lock; reject response mismatch before training. All 6 pass. |
| Full replay occupied by stale, high-priority groups | A fresh lower-priority group could not enter; matched fixture skipped training. An eligible high-priority control still trained. | Rescreen before admission, remove ineligible groups whole, refresh eligible priorities; preserve use-time rechecking. Both fixture arms pass. |
| Behavior-score normalization | Captured real-model scores normalized on a strided vocabulary axis differed from the learner's normalization. | Small **unmerged RVL dependency patch**, applied to pinned source, normalizes each vocabulary row like the existing trainer. No trainer replacement. |
| Failure evidence / runner accounting | Early preflight failures left no artifact; earlier no-update rounds overwrote a preceding record. | Atomic output ownership and retained rejection; require one new optimizer record per round. Preflight baseline: 2 fail / 1 control pass; fixed: 3 pass. |

The numerical reference forms the complete group-mean loss graph and performs one backward, independently of RVL's per-response backward/accumulation. It compares explicit VARE-group advantages, loss, clipped gradients, Adam state and post-update parameters. Two groups share a prompt ID; one has constant rewards. Merging those groups deliberately changes the reference gradient. Perturbed behavior scores fail the alignment control.

## Fixed conditions and measurement

Final [protocol](../benchmarks/grpo_e2e_cpu_v5/protocol.json), measured learner source `eac97b22f42c4c9efa55939d50abba5def00a346`. Apple M1 Pro / 32 GiB; CPU fp32, one Torch thread; Python 3.11.15, Torch 2.8.0, Transformers 4.56.2. Public `roneneldan/TinyStories-1M` revision `77f1b168e219585646439073245fe87e56b3023e`; **131,200 trainable parameters**, position embeddings and final layer norm. RVL base `e381c4b3b8c2ad8e5dc4c4b1a33b7922531c3e33` plus the [exact portable patch](../patches/rvl-contiguous-transition-logprobs-v1.patch). Model/dependency/source hashes are retained.

Seeds `65031/65032/65033`; eight completions per round in two intact groups of four, at most three new tokens each; four disjoint diagnostic evaluation prompts per checkpoint. Reward uses a text-byte checksum; evaluation uses a different SHA-based synthetic criterion. A separate checkpoint consumer reloads saved tensors and independently performs greedy decoding to check RVL's evaluation reports. These criteria measure system behavior, **not semantic task success**. Ties are permitted for diagnostic promotion; every fourth round instead requires an impossible 1.1 gain to exercise rejection.

Throughput includes reference computation, checkpoint IO, and independent evaluation comparison. The numerator counts completed rollout samples/completions and their generated tokens, not unique prompt IDs, evaluation outputs, or reference forward tokens. Model loading and post-campaign fault/recovery controls are excluded. Peak RSS is the whole-process highwater, including setup/fault controls. No discarded warmup or simultaneous local model test ran in the final campaign. Mean ± sample SD across the three declared workload/sampling seeds:

| Seed | Candidate updates | tasks/s | output tokens/s | Peak RSS MiB | CPU core % | Reload → next candidate ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65031 | 12 | 11.272 | 33.817 | 615.313 | 95.41 | 46.131 |
| 65032 | 12 | 12.465 | 37.394 | 638.219 | 98.61 | 44.690 |
| 65033 | 12 | 12.053 | 36.158 | 640.656 | 97.93 | 45.958 |
| Mean ± SD | 12 | **11.930 ± 0.606** | **35.790 ± 1.817** | max 640.656 | 97.32 ± 1.68 | **45.593 ± 0.787** |

Pooled main-run latency samples, milliseconds; p50 is the upper middle order statistic, p95 the nearest-rank statistic. Rollout latency includes queue/lock waiting with four concurrent requests and one serialized model. The validation probe is an additional audit call, not an estimate of all guard overhead. Stages overlap and must not be summed as exclusive time.

| Stage | n | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Rollout | 288 | 101.576 | 131.540 |
| Verification function | 288 | 0.00317 | 0.00654 |
| Freshness pruning | 36 | 0.0434 | 0.0821 |
| Validation probe | 36 | 0.203 | 0.247 |
| Transactional training hook | 36 | 65.125 | 79.757 |
| RVL checkpoint evaluation | 72 | 107.472 | 120.736 |
| Whole round, including reference/IO/comparison | 36 | 618.522 | 793.561 |

Same captured score tensors, CPU one thread, three alternating paired batches of 100 calls after 20 warmups: legacy normalization **0.9803 ± 0.0040 ms**, patched normalization **0.2015 ± 0.0019 ms**. Maximum error against FP64 normalization of these same tensors was `1.4521e-4` versus `6.9628e-5`. This is a component microcost/precision witness, not an end-to-end speedup or GPU result; [source and raw measurements](../results/grpo-e2e-cpu-v5/grpo-transition-cost.json) are retained. Total protection overhead against an unguarded full-learning baseline was not measured here.

## Correctness and retained failures

Maximum absolute reference discrepancies: loss `2.61e-8`, gradient `2.98e-8`, updated parameters `2.98e-8`, optimizer tensors `3.73e-9`; each stays below the fixed `2e-6` limit. Maximum behavior-logprob discrepancy is `3.05e-5`, below the **explicit v3–v5 `1e-4` fp32 alignment bound**. Token IDs/order are preserved exactly; probabilities are not claimed bitwise exact.

A supplemental read-only replay exposed a validation blindspot: all 33 nonzero gradients were clipped, so an incorrect batch-averaging factor could pass the clipped-gradient comparison (maximum counterfactual difference `5.96e-8`). Replaying the existing full-graph reference across all 36 saved rounds now checks the recorded preclip norm and reconstructs the native preclip vectors from the saved clipped vectors and Torch's clipping coefficient. Maximum reconstruction error was `4.77e-7`; all 33 wrong-factor controls were distinguishable before clipping. The old auditor accepted a rehashed wrong-norm trace; the model-backed auditor rejects it. This is an improved correctness gate, **not an observed RVL accumulation bug**. Actual preclip vectors were reconstructed, not directly captured. No new rollouts, native trainer runs, or changes to primary measurements/protocols were made. [Supplemental measurements and scope](../results/grpo-e2e-cpu-v5/preclip-validation.json) and a separate release archive retain the evidence.

Each seed pruned 88 of 104 cached rows inspected before admission (**84.62%**, deliberate zero-lag workload); selected batches had zero policy lag. Actual loaded model hashes/versions were recorded for every rollout and checked against saved checkpoints. Every seed rejected all eight controls: wrong rollout version, response mismatch, incomplete group, mixed policy, stale data, token/logprob length mismatch, nonfinite logprob, and a failure after a real optimizer step. All tested rollback paths restored model/optimizer/CPU RNG. Missing evaluation rows were also rejected by the existing promotion gate.

The all-zero-advantage final round still changes weights with warm Adam moments and historical weight decay `0.01`; it matches the reference optimizer and is not presented as a no-op or learning signal. Existing RVL snapshot/load logic reproduced the next candidate exactly. The reported 45.6 ms covers same-process checkpoint loading through the next actual candidate update, **not crash/restart**. The resume input and executed equality result are retained; the two compared resume-output states are not separate artifacts.

Failures are preserved in the history archive. V1's apparent PASS is explicitly invalidated: only 28 candidate updates occurred in 36 rounds, and no-op rounds overwrote earlier records. One preliminary performance campaign overlapped a local regression and is excluded. Width-2 tiny GPT-2 also failed a strict alignment diagnostic. V2 stopped after one completed round when the next comparison differed by `2.24e-5`; same-logit fp32/FP64 normalization differed by `3.80e-5`. Its `2e-5` gate was not rewritten. V3 separately fixed the fp32 bound and passed; v4 fixed reviewed preflight retention; v5 retained every latency sample and isolated post-campaign canaries. Earlier protocols/results remain unchanged; **only v5 supplies the headline measurements**.

Local full suite: **939 passed, 8 skipped** with the optional math grader excluded; the final GRPO/auditor regression has **4 passed**, including a rehashed missing-round negative control. An initial incorrectly configured invocation had 22 failures from legacy `python3`/subprocess import paths; its log is retained. The configured CI includes the isolated math grader and actual-model CPU regressions. The [initial independent automated review](https://github.com/mitukx/VARE/pull/71#discussion_r4291785712) found the preflight defect; its regression and fix are recorded, and the thread is resolved. An additional review request hit account quota; no extra credits/API were used. This is not outside human reproduction or a fresh independent review of the final head.

The default read-only auditor rehashed **129 artifacts**, recomputed **792 latency samples**, checked saved gradient/parameter vectors, checkpoint RNG/counters, evaluation-pair coverage and rollout/checkpoint identities. Optional model-backed mode also replays the existing full-graph loss/optimizer reference and checks preclip norm, reconstructed gradients and updated parameters. Resume-output equality remains an online runner check. Inspect [metrics, numerical comparisons, logs and validation](../results/grpo-e2e-cpu-v5/validation.json), [CSV](../results/grpo-e2e-cpu-v5/comparison.csv), [artifact manifest](../results/grpo-e2e-cpu-v5/artifact-manifest.json), and [raw release assets](https://github.com/mitukx/VARE/releases/tag/grpo-e2e-cpu-20261017).

## Reproduce

Use a fresh Python 3.11 virtual environment, install `requirements/e2e-rl-real-model-v1.txt` and `pip install -e '.[dev]'`. Activate it so subprocess `python3` also uses that environment. All output directories must be new.

```bash
git init /tmp/vare-rvl-grpo-new
git -C /tmp/vare-rvl-grpo-new fetch --depth 1 https://github.com/mitukx/Recursive-Verification-Lag.git e381c4b3b8c2ad8e5dc4c4b1a33b7922531c3e33
git -C /tmp/vare-rvl-grpo-new checkout --detach FETCH_HEAD
git -C /tmp/vare-rvl-grpo-new apply "$PWD/patches/rvl-contiguous-transition-logprobs-v1.patch"
python -c "from huggingface_hub import snapshot_download; snapshot_download('roneneldan/TinyStories-1M', revision='77f1b168e219585646439073245fe87e56b3023e', local_dir='/tmp/vare-grpo-model-new', allow_patterns=['*.json','*.txt','pytorch_model.bin'])"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
for seed in 65031 65032 65033; do
  python scripts/validate_grpo_e2e_cpu.py --model-dir /tmp/vare-grpo-model-new --rvl-source /tmp/vare-rvl-grpo-new --seed "$seed" --output "/tmp/vare-grpo-new-$seed"
  python scripts/audit_grpo_e2e_cpu.py "/tmp/vare-grpo-new-$seed"
done
VARE_GRPO_MODEL_DIR=/tmp/vare-grpo-model-new VARE_E2E_RVL_DIR=/tmp/vare-rvl-grpo-new python -m pytest tests/test_grpo_e2e_cpu.py
```

Audit retained data without inference/training (Torch still required):

```bash
mkdir /tmp/vare-grpo-retained-new
gh release download grpo-e2e-cpu-20261017 --repo mitukx/VARE --pattern grpo-e2e-cpu-main.tar.xz --dir /tmp/vare-grpo-retained-new
echo '539b8476201699e392cf77a2fca98810d98b774b80d3796f83752f3b070a28b0  /tmp/vare-grpo-retained-new/grpo-e2e-cpu-main.tar.xz' | shasum -a 256 -c -
tar -xJf /tmp/vare-grpo-retained-new/grpo-e2e-cpu-main.tar.xz -C /tmp/vare-grpo-retained-new
for seed in 65031 65032 65033; do
  python scripts/audit_grpo_e2e_cpu.py "/tmp/vare-grpo-retained-new/grpo-e2e-cpu-v5-main/seed-$seed"
done
```

For the supplemental reference replay, use final PR #71 code or the paired auditor/validator sources in `grpo-preclip-reference.tar.xz`; the original release tag remains the frozen primary-data snapshot. With the same pinned model and patched RVL directories prepared above:

```bash
for seed in 65031 65032 65033; do
  python scripts/audit_grpo_e2e_cpu.py "/tmp/vare-grpo-retained-new/grpo-e2e-cpu-v5-main/seed-$seed" --model-dir /tmp/vare-grpo-model-new --rvl-source /tmp/vare-rvl-grpo-new
done
```

Re-run the normalization microcost with the history archive's `measure_grpo_transition_cost.py` and `grpo-transition-scores-baseline.pt`, supplying a new JSON output path. Cross-host comparisons must check recorded host/dependencies; the runner's conformity flag checks source/model/patch and declared rounds/seeds, not host equivalence.

Unverified: separate rollout worker processes, GRPO SIGKILL/restart and durable scheduler/replay recovery, nonzero allowed lag, multi-epoch/off-policy clip boundaries, authenticated reward/attested publication, GPU/distributed execution, and independently graded semantic task improvement. PR #69 recovery was not copied or relabeled as GRPO evidence. **Next priority: upstream review and integration of the RVL numerical patch**, with broader numerical/backend coverage before any larger learning claim.
