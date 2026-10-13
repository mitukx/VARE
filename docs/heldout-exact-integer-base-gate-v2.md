# Held-out exact-integer base gate v2

Protocol id: `heldout-exact-integer-base-gate-v2`.

The protocol file was frozen before the mlx run. That run is now retained and is not to be repeated. v1 remains the stopped `stop_zero` record and is not edited to match v2.

## What changes, and what does not

v1 scored 0/64 with SmolLM2-360M-Instruct. The frozen final-line parser matched 3 of 256 samples, and those 3 were graded incorrect. That is a format failure and an arithmetic failure at 360M. It is not a reason to accept the prose those samples used.

v2 therefore keeps every outcome-bearing piece of v1:

- task pack bytes `189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc`
- prompt text inside that pack
- k = 4, temperature 0.7, at most 48 new tokens
- seed `20261013 + task_index * 4 + sample_index`
- final-line regex `^####\s+(-?\d+)\s*$`
- the same three acceptance decisions

The pack is reused because redrawing expressions after a zero would let the new draw depend on the observed run. The prompts stay contaminated for SmolLM2-360M only. They are not rewritten, and no item is dropped.

The model is the change. v2 pins an instruction-tuned 3B model, inside the 1.5B–3B range, that is not a retired Qwen2.5-0.5B pairing:

| Backend | Repository | Revision | Weights |
| --- | --- | --- | --- |
| `mlx` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | `4f83f8f146fdf28b512a06562b671d7af4fab457` | Published 4-bit mlx-community conversion. An unquantized mlx-community repository for this id was not available when the pin was written. |
| `transformers` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Upstream bf16 checkpoint, loaded float16 on CUDA and float32 otherwise. |

The two backends are separate pins. A 4-bit MLX result and a float16 CUDA result are not replicates of each other. `mlx-lm==0.32.0` is the Apple Silicon package pin. The 4-bit weights are about 1.7 GB and the upstream checkpoint is about 6.2 GB, which is the size basis for a 32 GB Apple Silicon machine and a 16 GB T4. The retained mlx run is recorded below. The CUDA backend has no retained run.

Protocol file: `benchmarks/heldout/exact_integer_v2/protocol.json`. SHA-256 `8c52c9cdff4403c15f43e4c617fc0e082e3a2a2a71ae78719fd5495d2534b85c`.

The runner refuses to start if that file's answer regex differs from the grader. There is no subset flag. `scripts/heldout_exact_integer_update_comparison_v2.py` exits 2.

## Retained mlx result

Operator-reported Apple Silicon run at code revision `89f671c00e7ed84fda1fc873e8814e0ff17bef2c`. Python 3.12.12 and `mlx-lm` 0.32.0. Do not rerun v2 and do not change its prompt, parser, sample count, threshold, or model revision. Do not train on this pack.

| Field | Value |
| --- | --- |
| Decision | `nondegenerate_base_signal` |
| Tasks passed | 29/64 (0.453125) |
| Wilson 95% | [0.3372944778061167, 0.5742640277994588] |
| Final-line parser matches | 250/256 |
| Graded-correct samples | 109/256 |
| Record | `results/heldout-exact-integer-base-gate-v2/m1-mlx-89f671c/` |
| Manifest SHA-256 | `ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278` |

The parser count describes the retained file. The gate metric remains pass@4. This result authorizes only a separate one-update save/reload resource smoke. It is not a capability claim. The commands below are the recorded run, not a request to repeat it.

## Mac

The block below is the command that produced the retained mlx result. Do not start it again. Do not start the unused CUDA command after seeing this result.

```bash
git rev-parse HEAD
python3 - <<'PY'
import hashlib
from pathlib import Path
root = Path('.')
expected = {
    'benchmarks/heldout/exact_integer_v1/tasks.jsonl': '189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc',
    'benchmarks/heldout/exact_integer_v2/protocol.json': '8c52c9cdff4403c15f43e4c617fc0e082e3a2a2a71ae78719fd5495d2534b85c',
}
for name, digest in expected.items():
    got = hashlib.sha256((root / name).read_bytes()).hexdigest()
    print(name, got)
    assert got == digest
PY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]" "mlx-lm==0.32.0" "huggingface_hub"
python -m pytest tests/test_heldout_exact_integer_v1.py tests/test_heldout_exact_integer_v2.py -q
mkdir -p "$HOME/vare-heldout-exact-integer-v2"
python scripts/run_heldout_exact_integer_base_gate_v2.py \
  --backend mlx \
  --output "$HOME/vare-heldout-exact-integer-v2"
```

Use a new output directory. The v1 directory must stay untouched. If the process stops, run the same command again. Completed `(task_id, sample_index)` rows in that directory are skipped. `summary.json` is written only after 256 samples. A changed `environment.json` refuses to append.

Then:

```bash
python scripts/heldout_exact_integer_update_comparison_v2.py
echo "comparison_exit=$?"
```

`comparison_exit` was required to be 2. The CUDA command below was not run. Do not start it after the mlx result.

```bash
python scripts/run_heldout_exact_integer_base_gate_v2.py \
  --backend transformers \
  --output "$HOME/vare-heldout-exact-integer-v2-transformers"
```

## What the retained rate does not establish

The 29/64 rate is a base-model measurement for this 4-bit checkpoint and this pack. It does not show that an update helps. It authorizes only a separate one-update save/reload resource smoke.
