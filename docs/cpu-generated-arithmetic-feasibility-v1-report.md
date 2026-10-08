# Generated arithmetic rollout feasibility v1

## Decision

**The frozen feasibility gate failed. Do not train on this task setup or open a formal confirmation cohort for it.** The pilot asked whether a cached Qwen2.5-0.5B-Instruct model could produce enough correct and incorrect, strictly parseable rollouts on a small generated arithmetic task to justify a separate preference-update study. It did not test a training method.

The protocol and all 64 prompts were committed before inference in `bbff71e`. The protocol SHA-256 is `4128dc05c79f19af9421aacaadec471d256a48f65775bfdb21ea6c8e2bd0f752`. Each of four declared expression compositions contributed 16 generated examples. These rows are permanently excluded from any later training or evaluation cohort.

## Results

| Measure | Result | Frozen rule |
| --- | ---: | ---: |
| Exact answers | 1/64 (1.56%) | 8–52 required |
| Strictly parsed outputs | 64/64 (100%) | At least 48/64 |
| Parsed but incorrect | 63/64 | At least 8 |
| Peak RSS | 3,336,716,288 bytes | At most 6 GiB |
| Wall time | 39.55 s | At most 3,600 s |
| Decision | **FAIL** | All conditions required |

The one correct answer was in `difference_times_sum`; the other three composition groups each scored 0/16. All outputs matched the required `FINAL: <integer>` syntax, so the failure came from arithmetic task performance rather than parser coverage. The independent audit reconstructed the oracle answers, strict parsing, metrics, gate, resource record and all bundle hashes. The audit passed; the experiment's frozen decision did not.

## Execution and audit

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, cached assets checked against the protocol hashes.
- Runtime: Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.3; greedy generation on CPU, network disabled, no paid compute.
- Runner: [`run_cpu_generated_arithmetic_feasibility_v1.py`](../scripts/run_cpu_generated_arithmetic_feasibility_v1.py).
- Independent, model-dependency-free audit: [`audit_cpu_generated_arithmetic_feasibility_v1.py`](../scripts/audit_cpu_generated_arithmetic_feasibility_v1.py).
- Retained raw bundle: [`run-1`](../results/cpu-generated-arithmetic-feasibility-v1/run-1/).
- Bundle manifest SHA-256: `db08b982bfd372d6f4a9c73649e64f7d93369575f03be90a7e4a443db0ac6072`.

The audit verifies records and retained resource metadata; it does not rerun inference. This was one small generated cohort on one cached model and one host. It does not establish model capability, preference learning, training impact, or held-out generalization.

## Next step

Retire this task setup. Do not relax the exact-match gate, reuse its 64 prompts, or treat oracle-correct answers as a basis for a formal update here. A future study needs a different task with a base-policy success rate suitable for a falsifiable intervention, a fresh frozen cohort, and a separate feasibility decision before any confirmation data is opened. The existing GSM8K forced-choice result and BoolQ non-passes remain the only relevant cached-language-model update evidence; neither establishes free-form task improvement.

Reproduce the lightweight checks and run from the repository root:

```bash
python3 -m pytest -q tests/test_generated_arithmetic_feasibility_v1.py
python3 scripts/run_cpu_generated_arithmetic_feasibility_v1.py \
  --output /tmp/vare-arithmetic-reproduction
python3 scripts/audit_cpu_generated_arithmetic_feasibility_v1.py \
  /tmp/vare-arithmetic-reproduction
```

The model run additionally requires the exact cached model revision and pinned runtime in the protocol. The runner refuses to overwrite existing output directories; use a new output path for any allowed reproduction.
