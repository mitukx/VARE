# Held-out coding task readiness update — 2026-10-25

**Decision: NO-GO for model inference, SFT/GRPO, held-out capability claims, and real-coding asynchronous comparison.** VARE now has 10 independently controlled source/grader certificates toward the unchanged 15-family gate. External model pretraining or task-recognition exposure is unknown, so 10 is not a claim of full model eligibility. This work measures deterministic task/grader certification and execution-environment reproducibility only; it demonstrates no model capability gain.

## Exact-head CI regression fixed

The E2E source inventory regression came from the VARE runtime package growing from 77 to 78 Python files when `src/vare/mlsys_heldout_task_pilot_v1.py` was added, while the auditor still recognized only the historical inventory. The repair now defines explicit 77-file and 78-file source inventories and binds each to its pinned base commit and source digests. It continues to accept the complete historical 77-file artifacts; a new 78-file diagnostic run is admitted only when its committed runtime inventory and hashes match. Missing files, unexpected files, changed source hashes, and mismatched base commits are rejected for both versions. The regression suite also verifies a fresh 78-file campaign from the exact committed tree. The focused local inventory selection passed **9 tests** on Python 3.13, including positive acceptance of both pinned versions and adversarial missing/unexpected file, changed-hash, and mismatched-commit cases.

## Count and frozen allocation

The original 15-task allocation and its Train/Development/Held-out assignments remain unchanged. Six of its nine held-out families now have complete current certificates. Four additional independent families are certified in prospective held-out extensions. The source/grader-certified count is **10/15**, leaving five more families to certify. External model pretraining/task-recognition exposure is not established, so this is not a claim that ten tasks are fully model-eligible. Since six original families qualify, the outside-pool target remains nine successful families; four of nine have been certified so far.

| Original task family | Split | Current decision | Basis |
| --- | --- | --- | --- |
| Transformers #48319 stale best checkpoint | Held-out | Certified | Baseline fails, exact fix passes, mutant fails; v8 raw records and v9 adjudicated summary |
| Transformers #49208 resumed metrics | Held-out | Certified | Baseline fails, exact fix passes, mutant fails; v8 records carried into v9 |
| Transformers #49412 fusion idempotence | Held-out | Excluded | Nine retained Qwen3 attempts exposed the task family |
| MLX-LM #1902 continuation truncation | Held-out | Failed | The known fix does not pass the executed grader controls |
| MLX-LM #1921 padded log probabilities | Held-out | Certified | Baseline fails, fix passes, incorrect mutant fails under v9 |
| MLX-LM #1926 Falcon-H1 cache/gradient | Held-out | Certified | Isolated baseline/fix/mutant and patch-admission controls passed |
| MLX-LM #1935 zero max tokens | Held-out | Excluded | The candidate grader requires a weight download and is not offline eligible |
| MLX-LM #1938 Phi3Small mask axis | Held-out | Certified | Isolated baseline/fix/mutant and patch-admission controls passed |
| MLX-LM #1979 SSM padding mask | Held-out | Certified | Baseline fails, exact fix passes, incorrect mutant fails |
| Transformers #45852; MLX-LM #4662, #1795, #1904, #1912, #1971 | Train/Development | Not held-out | Frozen split; not promoted into confirmation tasks |

The four prospective unique families are Transformers #49147 (Gemma3n FP16 gradient clipping), MLX-LM #1898 (Qwen3 nested RoPE configuration), #1877 (compressed-tensors format routing), and #1880 (GPTQ grouped error propagation). Their baseline/fix/mutant evidence and protocol hashes are linked below. Candidate model/agent prompts were not run or supplied to a model.

## New task measurements

| Family | Baseline | Exact fix | Incorrect mutant | Peak cgroup memory; max | Outcome |
| --- | --- | --- | --- | --- | --- |
| MLX-LM #1877 | Unsupported FP8 was accepted; three supported mappings passed | All four independent load-path checks passed | Unsupported FP8 was accepted | 67,883,008 / 67,817,472 / 68,419,584 B; 1,610,612,736 B | Certified |
| MLX-LM #1880 | Multi-group reference mismatch; max absolute difference 0.3834570646 | Passed the selected multi-group numerical reference | Restored indexing mutant failed the reference | 72,089,600 / 68,063,232 / 68,022,272 B; 1,610,612,736 B | Certified |

The #1877 grader calls the actual `load_model` compressed-tensors dispatch with quantization mocked. It verifies that unsupported `float-quantized` fails before quantization and that NVFP4, MXFP4, and packed affine formats retain their mappings. For #1880, a column-by-column GPTQ reference is compared with the grouped implementation over deterministic correlated synthetic data; it is not a model-weight or model-quality test.

The #1880 baseline was executed exactly once. Its first runner misparsed `unittest` output because `tqdm` carriage returns separated the case header from the standalone `FAIL` marker. A 16-check raw-log adjudication confirmed the expected baseline failure, source and grader hashes, exit status, image, and cgroup readings. The baseline was not rerun; only the not-yet-consumed exact-fix and mutant controls ran under the next frozen protocol. The #1877 and #1880 source/exposure scan found zero task-specific matches in 147 retained VARE refs and 5,796 local model/agent evidence files. The original source catalog also records no VARE model/agent exposure for the newly certified original tasks. These finite local scans do not establish absence from external pretraining; model eligibility beyond the audited VARE scope remains unknown.

## Candidate attrition

The frozen 24-item manual queue remains unchanged. Its only newly certified outside-pool family is Transformers #49147. Other reviewed queue outcomes include:

- Transformers #49116 is exposed: the retained rollout input hash covers the historical task prompt, so it is retired even though one representative rollout made no edit.
- MLX-LM #1925 has baseline/fix controls, but the frozen source catalog excludes it as overlapping #1921; its incorrect mutant also passes the Phixtral case.
- MLX-LM #1902 fails its known-fix control; #1935 remains excluded for weight downloads.
- #1910 overlaps the existing #1904 parser family; #1787 and #1930 overlap train task #1795; #4668 overlaps #4662 and its original test passes without memory instrumentation.
- TRL #7768 is a test-only LoRA/MoE behavior check; vLLM #52626 is NVIDIA-specific, #56841 requires a Linux/vLLM connector stack, and Transformers #49500 requires distributed multi-GPU execution.
- The scalar math candidate #4679 was screened out because its upstream test diff duplicates the degrees assertion in the radians test, leaving the relevant scalar behavior without an independent oracle.

The additional #1877 and #1880 candidates came from the wider 57-PR survey, not the frozen manual 24. Their prospective allocation is explicitly separate. Across current evidence, four outside-pool families are certified; five more unique outside-pool successes are needed. Counts are scoped to this finite audited inventory and do not estimate benchmark-wide task prevalence.

## Rebuild and memory measurement

The pinned MLX CPU Dockerfile was rebuilt from scratch with `--pull --no-cache --platform=linux/arm64`, using the pinned Python base digest and all 34 locked package versions. `pip check` and imports for MLX, MLX-LM, GPTQ, and utilities passed in a network-disabled, read-only, unprivileged 1.5-GiB container. The rebuilt image ID is `sha256:c320782d44021ae0e00e96f46df4c53f04792a17fd6bfe60581b8c8b0dd215a4`; the previous local ID was `sha256:2f907d158fa9aa08a2073ca580f85facf99e52e5cc792ac1b096f19f1cc56939`. The IDs differ, so byte-identical Docker image reproduction is **not** claimed. The pinned recipe, lock-matching package inventory, and import smoke are reproducible evidence.

New qualification runs read cgroup v2 `memory.peak` and `memory.max`; all reported values are positive and within the enforced 1,610,612,736-byte limit. Historical pilot rows reporting `0 B` remain unusable and are preserved unchanged.

## Reproduction

The MLX qualification runner expects a full-history upstream clone at `source/mlx-lm.git/.git` so every pinned source and grader revision can be archived. Recreate that input and the tested arm64 image with:

```sh
git clone --filter=blob:none https://github.com/ml-explore/mlx-lm.git source/mlx-lm.git
git -C source/mlx-lm.git fetch --unshallow origin
docker --context colima build --pull --no-cache --platform=linux/arm64 \
  -f benchmarks/mlsys_heldout_qualification_v1/Dockerfile.mlx \
  -t vare-mlx-heldout-cert-rebuild:v1 benchmarks/mlsys_heldout_qualification_v1
```

The append-only run records, protocol locks, raw output, and hashes are retained under the result directories named in the matrix. The tested rebuild used all 34 locked package versions; image identity differed from the previous local image, so reproduction is claimed at the pinned recipe/dependency/import level, not as a byte-identical image digest.

## Evidence and limits

- [Task matrix and hashed evidence references](../results/mlsys-heldout-task-readiness-v2.json) — SHA-256 `cde4489049e9f6e685528651907d63a1ef88ad4121fb33c4914468a7add2d65a`.
- #1877/#1880 exposure audit — SHA-256 `9cc1ba8c23d11cd56d06b9385405b0a8fc9ab7409dd8aa83d6b4d383d6e6679f`.
- #1877 frozen protocol — SHA-256 `5d638dcd84d3f2a08df796b01f24aeac80a368ce704036212bd304b2a97cfb94`; run summary — `a33b4ca18adefc7756fa9ce0b51eaf88b60640de1838c27ff66da55be5327274`.
- #1880 frozen follow-up protocol — SHA-256 `ff0778b3554ff3af979762524cbc3434bec99fa0481969c8da53cb0965dcb06e`; run summary — `adab1d9151401ff9946ee8e40617bd468c2ff49de05c3b79f86fab58e6bc6c2c`; baseline adjudication — `43919c001fa2248bf274837190687e6227834e42c8b07ad7d2bc5e08efdecb07`; adjudicator source is pinned in the matrix.
- Pinned image rebuild attestation — SHA-256 `45974505e96810237c677a862364762fcecdcca4ac2cce71249153924dfc13a7`; raw smoke log — `4700c06d3b952815f1f9363a853d61e1fe916f63c4d446115da0222c858cee62`.
- #49147 and #1898 summaries — `0f510808a0499fcff4edfc5bba32f81f25c59a9809f4e50b7c907eecca7636db` and `fe18209d203514b48a11e9c902d416766064bd292f924fb7ec29ed68896e0dc8`.

This cycle ran deterministic task tests only. It started no coding-agent/model inference, optimizer update, training, held-out model evaluation, async coding comparison, or recovery fault injection. Task certification is improved; async operational reliability is not measured here; model capability remains unmeasured and the 15-family gate remains closed.
