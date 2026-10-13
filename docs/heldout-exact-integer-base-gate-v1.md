# Held-out exact-integer base gate v1

Protocol id: `heldout-exact-integer-base-gate-v1`.

**Stopped.** The operator-reported Apple Silicon mlx run at code revision `5d36336ad0272d59ca815646fcb2f812849be1c5` finished `stop_zero` (0/64; Wilson 95% interval [0, 0.05662405979280534]). Do not rerun this protocol and do not change its prompt, parser, sample count, or threshold. The raw files are [the result record](heldout-exact-integer-base-gate-v1-result.md). A later model is a different protocol: [v2](heldout-exact-integer-base-gate-v2.md).

This note is the reproduction path for the frozen gate. The task pack and grader were committed before that run. Model weights are not in the repository. The retained outputs are the result record, not a reason to edit the pack.

## Frozen pack

| Object | Value |
| --- | --- |
| Tasks | `benchmarks/heldout/exact_integer_v1/tasks.jsonl` |
| Task count | 64 (16 each of `add_sub`, `mul_add`, `div_exact`, `paren`) |
| Generation seed | `20261013` |
| Tasks SHA-256 | `189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc` |
| Protocol file | `benchmarks/heldout/exact_integer_v1/protocol.json` |
| Protocol SHA-256 | `21c959a86423505a68fa53969837b883989857072a599110e08ee21ee5bf5276` |

Answers are stored only as SHA-256 of the decimal integer. The grader recomputes the value from the expression. The expression accepts digits and `+ - * / ( )` only, and `/` must divide evenly. The answer, as a whole integer token, must not appear in the prompt. Prompts are generated from the seed above. They are not rows from GSM8K, SVAMP, ARC, MATH, OpenBookQA, StrategyQA, BoolQ, SNLI, DeepMath, or the retired inventory word-problem pilot.

Confirm the pack before a run:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
root = Path("benchmarks/heldout/exact_integer_v1")
for name, expected in (
    ("tasks.jsonl", "189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc"),
    ("protocol.json", "21c959a86423505a68fa53969837b883989857072a599110e08ee21ee5bf5276"),
):
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    print(name, digest)
    assert digest == expected
PY
```

## Decoder

- k = 4 samples per task (256 samples total).
- Temperature 0.7, at most 48 new tokens.
- Sample seed = `20261013 + task_index * 4 + sample_index`.
- The tokenizer chat template is required. A missing template stops the run.
- A sample is correct only when its last non-empty line matches `^####\s+(-?\d+)\s*$` and that integer equals the expression.
- Primary metric: fraction of tasks with at least one correct sample. The summary adds a Wilson 95% interval with z = `1.959963984540054`, clamped to [0, 1].
- There is no subset flag.

Decisions, applied to the finished summary only:

- `stop_zero`: pass rate is 0. Stop. Do not change the prompt, parser, sample count, or threshold.
- `stop_saturated`: pass rate is 1. Stop. Do not retune the pack into a training set.
- `nondegenerate_base_signal`: pass rate is strictly between 0 and 1. This records base feasibility for this pack only. It authorizes a separate update-cost smoke, not a policy-update comparison.

## Models

| Backend | Repository | Revision | Where |
| --- | --- | --- | --- |
| `mlx` | `mlx-community/SmolLM2-360M-Instruct` | `5d5e6fee5beb5d4d4d85e2203ef23e1230a9aac7` | Apple Silicon, `mlx-lm==0.32.0` |
| `transformers` | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` | CUDA notebook GPU |

The MLX weights are the bf16 conversion (about 724 MB). Do not substitute another quantization. On CUDA the loader uses float16; otherwise it uses float32. Generation is one sample at a time, so the resident model is one 360M network. `mlx-lm` is not a dependency of this repository and is not installed by CI.

The MLX sampler is `mlx_lm.sample_utils.make_sampler(temp=0.7)`, with `mlx.core.random.seed(seed)` before each sample. The Transformers sampler uses `do_sample=True` and `temperature=0.7`, with `torch.manual_seed(seed)` before each sample.

## Resume

Each finished sample is appended to `samples.jsonl` and `fsync`ed. Re-running the same command with the same `--output` skips completed `(task_id, sample_index)` pairs. `summary.json` is written only when all 256 samples are present.

The runner refuses to append when `samples.jsonl` belongs to another pack, backend, or model id, or when `environment.json` differs from the current interpreter. A partial final line fails closed. Inspect it, delete only that incomplete line, and rerun. A disconnect during weight download is resumed by the Hugging Face cache (`HF_HOME`), separately from `samples.jsonl`.

## Update comparison

`scripts/heldout_exact_integer_update_comparison_v1.py` prints a refusal and exits 2. It does not sample, train, or compare best-of-n with GRPO or RFT. A non-degenerate base summary would still not be that comparison: the next separate check would be an update-cost smoke (one update, save/reload, and a resource record) under its own protocol. This gate does not include that smoke.

## Mac, Apple Silicon

The mlx run below is the retained result. Do not start it again.

`mlx` runs on Apple Silicon. The 360M bf16 model fits in 32 GB unified memory. The recorded checkout is `5d36336ad0272d59ca815646fcb2f812849be1c5`. The block below is the command that was run, not a request to repeat it:

```bash
git rev-parse HEAD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]" "mlx-lm==0.32.0" "huggingface_hub"
python -m pytest tests/test_heldout_exact_integer_v1.py -q
mkdir -p "$HOME/vare-heldout-exact-integer-v1"
python scripts/run_heldout_exact_integer_base_gate_v1.py \
  --backend mlx \
  --output "$HOME/vare-heldout-exact-integer-v1"
```

Resume with the same `run_heldout_exact_integer_base_gate_v1.py` command. Then:

```bash
python scripts/heldout_exact_integer_update_comparison_v1.py
echo "comparison_exit=$?"
```

`comparison_exit` must be 2. Read `summary.json` only after the gate command finishes with `"complete": true`. Keep the output directory. Do not edit the task pack or the parser after seeing outputs.

## Colab

Do not start a new v1 Colab run. The stopped result is the mlx record above. The notebook remains as the unused Colab path; do not run its cells.

Use a GPU runtime. The frozen decoder does not change if the assigned GPU is a T4, L4, or A100: one sample, float16, at most 48 new tokens. The notebook is `notebooks/heldout_exact_integer_base_gate_v1.ipynb`.

The notebook clones `https://github.com/mitukx/VARE.git` at `cursor/heldout-exact-integer-gate-af75` unless `VARE_REPO` and `VARE_REV` are set. Mount Drive when asked. Samples and the Hugging Face cache go to `MyDrive/vare/heldout-exact-integer-base-gate-v1` and `MyDrive/vare/huggingface-cache`. Those paths survive a runtime reset; `/content` does not.

After a disconnect, start a new GPU runtime, reopen the notebook, and run every cell again with the same revision and Drive path. Completed lines are skipped. The last cell runs the comparison script and stops when the exit code is 2.

## What a finished run does not establish

The retained pass rate is a base-model measurement on this pack. It is not evidence that an update helps, and it is not a multi-seed training result.
