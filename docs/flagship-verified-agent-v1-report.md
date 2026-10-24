# Installed-repair leakage and a bounded coding-teacher NO-GO

**Decision: NO-GO for learning. The intended coding-agent improvement claim is not established.** This cycle collected two new upstream module-repair contracts, completed four local Mistral-7B attempts, and reproduced and closed two installed-repair access paths. It produced no successful model-generated repair, student update, held-out comparison, asynchronous learning experiment, or RSI intervention result.

The measured systems finding is that excluding known fixes from a candidate workspace does not exclude them from its dependency image. On MLX-LM GPTQ, a candidate containing only an import of the installed implementation passed every independent check. Replacing the installed Python source still left a usable compiled repair in `__pycache__`. The final image replaces installed task modules with immutable baseline bytes, removes package bytecode, and corrects installed source hashes and sizes in `RECORD`.

| GPTQ candidate, identical four oracle calls | Original image | Source replacement only | Final portable image |
| --- | ---: | ---: | ---: |
| Import installed `gptq_quantize`, without implementing a repair | 4/4; task accepted | 1/4; task rejected | 1/4; task rejected |
| Load installed compiled GPTQ code with `marshal` | 4/4; task accepted | 4/4; task accepted | 0/4; task rejected |
| Exact upstream repair supplied as candidate source | 4/4 | 4/4 | 4/4 |

These are deterministic, authored adversarial controls on one real upstream module, not model-generated successes or an estimated exploit prevalence. The source-only and final images have the same 187 Python source hashes. The final image contains zero MLX-LM `.pyc` files, and both installed task `RECORD` entries match their actual baseline hashes and sizes. The final image digest is `d13df18e8ba98a4a71d5729073346ff3ae590f0d9ae85f850561d27a626b24c5`.

The [compiled-repair comparison](../results/flagship-dependency-leakage-v1/summary.json), [final calibration](../results/flagship-dependency-leakage-v1/summary.json), and [image census](../results/flagship-dependency-leakage-v1/summary.json) retain the results. Both real known fixes pass all their cases (GPTQ 4/4; tokenizer 9/9). Both baselines and all four predeclared incorrect mutations fail their task gates. One incomplete GPTQ repair passes three of four cases, demonstrating why single- and two-group checks alone are insufficient for that mutation.

The final source/bytecode comparison and complete calibration took 13.186 and 100.719 monitored seconds, respectively, with 32.344 and 33.344 MiB maximum sampled operator-process RSS. Candidate containers were separately limited to one CPU, 512 MiB, 64 PIDs and 45 seconds per call, with network disabled, a read-only root, dropped capabilities, and no grader mount. Host swap growth was zero. These times are descriptive; execution order and concurrent correctness tests do not support a throughput claim. Image preparation/export cost is separate from grading. Build logs and exact image exports are retained.

The original image's `quant/gptq.py` hash is `bd4c08453a24fe41fea8ce5885f433b718628c16c4e7a0596900fc72553daf16`, exactly the upstream known-fix file. This access invalidates that image for positive capability or training-data claims. The initial qualification omitted this dependency-asset check; the operator missed it before inference. All four model candidates stayed byte-identical to known failing baselines, so they remain directly auditable observed negatives. Their original image qualification is revoked, and their records are classified exploratory and training-ineligible. No result was promoted or used for learning.

Two intermediate cleanup images also remain ineligible. The local legacy Docker builder returned success while omitting the heredoc body: 187 bytecode files and stale `RECORD` entries survived. The failed [record-consistency check](../results/flagship-dependency-leakage-v1/record-consistency-failure.json) and all earlier protocols/builds are preserved. The authoritative build uses [exec-form invocation of an explicit sanitizer](../benchmarks/flagship_development_v1/Dockerfile.portable); its build output records actual bytecode removal. This is a measured failure of these build invocations, not a general Docker security claim.

| Development contract | Immutable baseline | Exact upstream fix | Scope |
| --- | --- | --- | --- |
| [MLX-LM #1880](https://github.com/ml-explore/mlx-lm/pull/1880), GPTQ group propagation | `e5962529e5614ce00f14bdd39fe5fc6e410ca2b0` | `d4d38d96485c6546a920b3c8431e455c7aa60980` | Actual `mlx_lm/quant/gptq.py`; independent immediate column-update oracle |
| [MLX-LM #1974](https://github.com/ml-explore/mlx-lm/pull/1974), thinking markers | `93e6d4a6447b3080861985947f0820e76fcb957d` | `d86ed1fec122c0dc6717d4424bb479761125e48c` | Actual tokenizer module; real local WordLevel tokenizers and rendering |

The task, test, license and source hashes are in the [original qualification protocol](../protocols/flagship_development_qualification_v1.json). Exposure search covered 116 locally retained origin revisions and found no prior task reference in their docs, protocols, benchmarks or results. This is bounded exposure evidence, not a guarantee about all external work or model pretraining. Neither task belongs to the historical reserved held-out allocation. Both are now consumed Development tasks and contribute **zero** eligible untouched held-out tasks toward the 15-task learning gate. The final calibrations did not rerun a model.

These are narrow target-module contracts against digest-pinned runtime dependencies. They do not certify complete upstream repository integration, arbitrary Python behavior, pretrained quantization quality, every possible dependency shortcut, or general sandbox soundness. The independent expected outputs and reference implementation remain on the operator side; only target source and an invocation adapter enter the candidate container. Matching fixes in installed dependencies, their compiled forms and metadata are part of the tested boundary.

The requested 14B teacher was **NOT_RUN locally**. PR #88's Qwen2.5-Coder-14B GGUF study ran on Linux x86_64 with 24 GiB swap. Its checkpoint was not present in the inspected Mac caches or temporary storage, and 740.19 MiB free swap did not meet the conservative 2 GiB large-model preflight. No 14B weights were downloaded or loaded, and none of its consumed tasks was rerun. This is an availability/preflight decision, not evidence that 14B inference or coding would fail on this hardware.

The tested alternative was the largest cached instruction checkpoint, `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, snapshot `a4b8f870474b0eb527f466a03fbc187830d271f5`; its weights occupied 4,077,480,215 bytes. Coding usefulness was unknown rather than assumed equivalent to the coding-specialized 14B model. The separate smaller-model resource protocol retained the existing 7B-family 128 MiB free-swap floor, a 6 GiB process RSS cap, 25% free-memory floor and a stricter 32 MiB swap-growth abort. Historical resource protocols were unchanged.

The [teacher protocol](../protocols/flagship_cached_mistral_teacher_v1.json) was committed before model load. Each task received one greedy attempt (seed 101) and one sampled attempt (seed 211, temperature 0.6, top-p 0.95, top-k 40). Each had 12 turns, 768 new tokens per turn, 4,096 total new tokens, an 8,192-token context, 240 seconds, three public-test calls and one fixed independent grading batch. The existing controller came from attested PR #120 commit `58ecd9c9727a1544a5a08c9c250f1f36dd9a5f8d` in a separate checkout; that unmerged dependency was not silently merged into main. The checkpoint's plain chat template received losslessly flattened system/tool content. Actual input/token IDs, model tokens, raw model log-probabilities, tool records, patches, grades and telemetry are retained.

| Teacher measurement | Observed |
| --- | ---: |
| Independently correct attempts | 0/4 |
| Development tasks solved at two attempts | 0/2 |
| Valid edits / executed model tools / public-test calls | 0 / 0 / 0 |
| Generated tokens, including terminal tokens | 3,072 |
| Monitored attempt wall time / total experiment wall time | 133.158 s / 170.477 s |
| Successful repairs per total wall-hour | 0 |
| Maximum sampled process RSS / MLX peak allocation | 4,086.109 MiB / 4,907,659,232 bytes |
| Minimum host free-memory reading / swap growth | 36% / 0 MiB |
| Within-task binary reward groups | `[0, 0]`, `[0, 0]` |

All attempts generated explanatory plans containing tool examples, hit the 768-token first-turn cap and stopped on malformed or corrupt patch recovery. Three have `malformed_diff`; one parsed patch failed Git application. No model source inspection occurred through tools. This measures failure of the frozen model/controller configuration, not inability to solve the bugs with another interface. No further prompt variants were run. MLX 0.32.4, MLX-LM 0.32.0, Transformers 5.20.2 and Tokenizers 0.23.3 were checked before loading; the old controller's raw `mlx: unknown` field is preserved rather than rewritten.

The [trajectory index](../results/flagship-cached-mistral-teacher-v1/trajectories.jsonl) contains **zero positives and four negatives**, each bound to checkpoint, task, verifier and raw-artifact hashes. It explicitly marks every row training-ineligible. Calibration known fixes are not model trajectories. Raw model probabilities are not asserted to equal truncated-sampler behavior probabilities, so these are not GRPO-ready records. No student SFT smoke was forced from Development data, and constant-zero groups did not authorize GRPO.

The final [eligibility decision](../results/flagship-dependency-leakage-v1/eligibility-decision.json) keeps SFT, GRPO, model promotion, independent held-out comparison, model-coupled async execution and RSI intervention evaluation **NOT_RUN / NO_GO**. No 504-round synthetic benchmark was repeated. The fallback result is the bounded installed-repair isolation measurement above; no statistical general coding gain, repeated downstream improvement, or recursive self-improvement is claimed.

PRs #99, #102, #114, #117, #118, #120 and #121 were inspected at their recorded exact heads. The new experiment excludes the MLX-LM #1971/v9 and consumed-tokenizer grading paths with unresolved historical trust findings, reuses the corrected private null-comparison approach, and leaves frozen shared runtime bytes unchanged. The [PR audit](../results/flagship-development-qualification-v1/pr-audit.json) records heads and open findings. This operator review is not independent approval. No existing PR was merged or historical result rewritten.

Raw teacher and isolation records are stored as lossless tar archives with per-member SHA-256 inventories to keep the review small. Their original paths and bytes are preserved; the auditor reads both archives directly. To audit without a model, Docker, training or task execution, run from this checkout with Python 3.12:

```sh
python scripts/audit_flagship_results_v1.py \
  --manifest results/flagship-study-v1-manifest.json \
  --manifest-sha256 5147e6f2539e5582238afabe83af2b662a28d6967ce1f8739ecb0b4d1d59e48f
```

The auditor checks the complete raw archive inventory, exact policy identity, both complete reward groups, source/public-test bytes, token counts, resource ceilings, baseline grade agreement, source/bytecode delegation outcomes and final image metadata. It uses explicit exceptions, including under `python -O`. Provenance regressions deliberately alter policy and source identity and must be rejected.

For a live **historical integrity reproduction**, load the retained ARM64 image export, provide an upstream MLX-LM Git object store containing the four full source commits, and use a fresh output path:

```sh
DOCKER_CONTEXT=colima python scripts/replay_flagship_dependency_leakage_v1.py \
  --protocol protocols/flagship_installed_asset_isolation_v1.json \
  --protocol-sha256 1c059dccfad9c873e3de8b709641c119cb2d17519e93d95a7067698c49f92389 \
  --source-repository /ABSOLUTE/PATH/TO/MLX-LM-GIT \
  --output /ABSOLUTE/NEW/OUTPUT

DOCKER_CONTEXT=colima python scripts/probe_flagship_compiled_repair_v1.py \
  --protocol protocols/flagship_installed_asset_isolation_v1.json \
  --protocol-sha256 1c059dccfad9c873e3de8b709641c119cb2d17519e93d95a7067698c49f92389 \
  --output /ABSOLUTE/OTHER/NEW/OUTPUT
```

The image export is an external local deliverable, not a tracked Git blob or public registry publication; its SHA-256 is `4723319c19f96703f3671fe1de97818d61bbc028044849889964b1b67cfd655f`. Exact live environment reproduction on another host requires that archive (or an independently attested matching image). The source/module scope and platform restrictions remain explicit. The two teacher tasks and four attempts must not be presented as fresh evidence if historically reproduced.

The local validation record contains the focused candidate-isolation and new experiment/audit tests, Ruff results, independent byte audits, full manifests and the inference environment lock. Exact-head CI and independent review are separate integration gates; this report does not claim they have passed, and no merge is authorized by the experimental outcomes alone.
