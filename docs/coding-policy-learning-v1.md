# Coding-policy learning v1: Qwen3-4B 4bit LoRA on MLX (one Apple-silicon Mac)

Status: **protocol frozen, no real run yet.** Every number below that comes
from a run is from the Linux box with a mock policy or a tiny random model.
Those numbers are labeled *synthetic* and say nothing about any real policy.
The evidence level is E0/E1: the harness and the real-task graders are
checked. There is no E4 claim.

## Question

On held-out repositories, does LoRA training of
`mlx-community/Qwen3-4B-Instruct-2507-4bit@50d4277` on verified Train tasks
raise the trusted pass rate compared with the no-update policy? All arms use
identical inference budgets.

## Pipeline

```
build-curriculum  SWE-smith-py@77cab90 (MIT) -> 80 Train / 42 held-out candidate tasks
preflight         one Train task: disk, setup time, grade time, generation, LoRA memory smoke
calibrate         every task, on the run machine, before inference:
                  bug fails F2P, P2P green, reference fix passes, prompt <= 2560 tokens
rollout           base policy, 40 Train tasks x 4 samples (T=0.7); outputs persisted before grading
train             sft_reference | rft_self   (same LoRA config, 100 iterations each)
evaluate          no_update | sft_reference | rft_self, greedy, 512 new tokens, same held-out set
compare           exact McNemar, Holm over the 2 comparisons, paired bootstrap
```

Code: `src/vare/learning_pipeline/`. CLI: `scripts/learning_pipeline_v1.py`.
Protocol: `protocols/coding_policy_learning_mlx_4b_v1.json`. It freezes the
hashes of the manifests, configs, lock files and all pipeline code. `run-all`
stops on any mismatch.

### Curriculum and contamination control

- **Source.** `SWE-bench/SWE-smith-py`, pinned revision, MIT license.
  - Each row is a synthetic bug-introducing patch in a public repository, with FAIL_TO_PASS and PASS_TO_PASS tests.
  - The reverse of the patch is the reference fix.
  - SWE-rebench V2 (CC-BY-4.0) is recorded only as a license entry, because it needs containers.
- **Repositories.**
  - Train: Suor/funcy, andialbrecht/sqlparse, gweis/isodate, luozhouyang/python-string-similarity, pytest-dev/iniconfig.
  - Held-out: python-hyper/h11, msiemens/tinydb, borntyping/python-colorlog.
  - The split is by repository, so the two sets share no repository.
  - Each repository has a permissive license recorded (MIT or BSD-3-Clause). `check-licenses` prints the table.
- **Exclusions.**
  - Whole owners are excluded: `huggingface/*`, `vllm-project/*` and `ml-explore/*`.
  - So is every repository named by an `owner/repo#N` identifier in the VARE tree. The scan must find `vllm-project/vllm#64830` and `ml-explore/mlx-lm#1904`, or the build fails.
  - Rows are deduplicated by normalized patch hash and by (repository, file, issue-text hash).
- **Task filters.**
  - The fix touches one library file and at most 30 changed lines.
  - At most 30 FAIL_TO_PASS and 400 PASS_TO_PASS tests.
  - The issue text is at most 3000 characters and the file at most 8000 characters.
  - The reference blocks must restore the clean file byte for byte.
- **Contamination.** This is UNKNOWN. The upstream repositories are public. The mutations are synthetic, and the claims are relative to the no-update arm on the same tasks.

### Execution without containers

Each repository is checked out at a pinned mirror commit
(`configs/learning_pipeline_v1/repos.lock.json`). It gets its own venv with
`pip install -e . pytest==8.3.5` plus pinned extras. Grading works like this:

1. `git checkout -f` and `git clean -fd`.
2. Apply the bug patch, then overwrite only the target file.
3. Run pytest on the F2P and P2P ids, honoring the repository's own `addopts`.
4. Parse the `-rA` output.

Grading fails closed. A missing test, a timeout, a change to any other file
or an edit that cannot be applied all count as failure.

### Verifiers and logging (hypothesis-compatible)

Every candidate is graded twice:

- `cheap_f2p_v1` runs F2P only. Its wall time is measured on its own run.
- `trusted_full_v1` runs F2P plus all P2P. Its verdict is the reward and the metric.

Each record in `rollouts/graded.jsonl` and `eval/<arm>/graded.jsonl`
(schema `vare.learning_pipeline.rollout/v1`) holds:

- `checkpoint_id` and `checkpoint_step`;
- `group_id` (checkpoint × task) and `task_group` (repository);
- `sample_index`, the seed and the decoding parameters;
- `output_text` and `final_diff`;
- the status of every requested test from each verifier;
- wall times and token counts.

Training rows have provenance sidecars (`data/<arm>/*.provenance.jsonl`). An
RFT row points back to the exact rollout it came from. This is enough to
replay budgeted verifier-error correction or the allocation of trusted
verification during training offline, without the model. v1 RFT is single
round, so all rollouts come from the base checkpoint (step 0, G=4).

### Arms and budgets

| arm | training data | notes |
|---|---|---|
| no_update | none | frozen base |
| sft_reference | reference fixes of calibrated Train tasks, in the edit format | uses privileged reference solutions |
| rft_self | the base policy's own trusted-passing samples, at most 2 per task | also uses the 160-sample rollout budget |
| grpo | not run | stub only (`grpo.py`): estimated 8–12 GiB peak |

Both trained arms use the same LoRA settings:

- rank 8, scale 20, 8 layers;
- batch 1, 100 iterations, learning rate 5e-5, Adam;
- maximum sequence 3072, prompt masked, gradient checkpointing, seed 0.

Inference is the same for every arm: greedy decoding, prompt at most 2560
tokens, 512 new tokens, one sample per held-out task.

### Stop rules (frozen)

- Free disk below 4 GiB stops the run. This is checked before every task, sample and arm, and against the preflight projection.
- Available memory at start below 8 GiB stops the run. So does a training-smoke peak above 11 GiB.
- A projected time above 3 h stops the run. The hard wall-clock cap is 3.5 h, with at most 45 min of training per arm.
- Fewer than 32 qualified held-out tasks stops the run before inference.
- If RFT has fewer than 8 examples or fewer than 4 tasks, `rft_self` is NO_GO and is not trained.
- A frozen hash mismatch or an adapter with all-zero LoRA B matrices stops the run.

### Success criterion (frozen)

An arm counts as `improved` vs no_update only if all three hold:

- the Holm-adjusted exact McNemar p is below 0.05;
- the pass-rate lift is at least 0.10;
- arm-only wins outnumber base-only wins.

Power with n=40 (normal approximation, one comparison) is about 0.66 when
the discordant rates are 0.20 and 0.025. A smaller real effect will probably
read as `no_detectable_change`. This is one seed: an `improved` result would
still need multi-seed replication before an E4 claim.

## Resource estimates for an M1 Pro with 32 GB

**Memory.**

- Inference was measured earlier at a 3.4–3.75 GiB peak.
- For LoRA training, the analytic estimate (`memory-estimate`) at sequence length 3072 is about 8.9 GiB with fp32 logits and about 6.2 GiB with bf16. At 4096 it is about 10.9 GiB, which is why the cap is 3072.
- The preflight smoke measures the real peak on a padded sequence of about 0.85 × 3072 tokens and stops above 11 GiB.

**Time (estimate).**

| step | minutes |
|---|---|
| setup and calibration | about 20 |
| preflight | about 8 |
| rollouts (160 × about 15 s) | about 45 |
| training, 2 × 100 iterations | about 35 (not measured on the Mac yet) |
| evaluation (3 × 42 tasks) | about 30 |
| total | about 2.3 h |

**Disk (estimate).** About 0.3 GiB for repositories and venvs, 0.22 GiB for
dataset shards and 2.3 GiB for the model if it is not already cached.

## Box checks (Linux x86, CPU)

These check plumbing only.

- **Unit tests.** `tests/test_learning_pipeline_v1.py` covers:
  - parsing and applying the edit format;
  - exclusions and the identifier scan;
  - manifest disjointness;
  - the prompt hiding tests and the reference;
  - the pytest parser;
  - the dataset builders;
  - Holm correction and the decision rule;
  - the GRPO advantages;
  - the resource guards;
  - the frozen-hash check.
- **Calibration of all 122 candidate tasks.** This is a real grader run, on Linux with Python 3.13. All 122 pass the grading checks (bug fails F2P, P2P green, reference passes). The prompt-length filter then gives:
  - with the real Qwen3 tokenizer: 78/80 Train and 42/42 held-out tasks qualify (2 Train prompts are too long);
  - with the character estimate used in mock mode: 80/80 Train and 39/42 held-out.
  - The count is recomputed on the run machine before inference.
- **Mock run** (`--mode mock`, SYNTHETIC: the mock copies reference fixes with a fixed probability). It runs every phase end to end in about 70 s and exercises grading and the per-test logs on real tasks. Its `improved` verdicts come from the mock's built-in success rate and mean nothing about any model.
- **Tiny-model run** (`--mode synthetic_model`, a random 2-layer Qwen3-architecture 4-bit model with the real tokenizer, SYNTHETIC, reduced budgets in a box-only protocol). It covers:
  - real MLX generation (all outputs malformed, as expected from random weights);
  - mlx-lm LoRA training through `lora_worker`, with metrics, checkpoints and a non-zero LoRA B check;
  - adapter reload and evaluation on all 42 held-out tasks;
  - the RFT NO_GO path (zero self successes).
  - CPU timings and memory from this model (for example a 5.35 GB training-smoke peak) say nothing about the 4B model on Metal.

mlx-lm 0.32.2 `lora.run()` drops a caller's `training_callback`, so
`lora_worker` calls `load`, `load_dataset` and `train_model` itself to keep
the metrics callback.

Box calibration fixed three environment issues before freezing:

- funcy needs the `whatever` package;
- tinydb's `addopts` needs pytest-cov and PyYAML;
- colorlog needs its `-p no:logging` addopts honored, and isodate needs its setuptools-scm version file kept (`git clean` without `-x`).

## Mac command

```bash
cd VARE && git checkout research/learning-pipeline-v1
python3.12 -m venv .venv-lp && source .venv-lp/bin/activate
pip install --require-hashes -r requirements/mlsys-agent-mlx.lock.txt
pip install -r requirements/learning-pipeline-v1.txt
python scripts/learning_pipeline_v1.py verify
python scripts/learning_pipeline_v1.py preflight          # stops here on a disk/memory/time rule
python scripts/learning_pipeline_v1.py run-all            # resumable; re-run the same command after a crash
```

Outputs go to `artifacts/learning_pipeline_v1/run-001/`, which is
git-ignored. The cache is `~/.cache/vare-learning-pipeline-v1/`. If a stop
rule fires, its reason is in `STOPPED.json`.
