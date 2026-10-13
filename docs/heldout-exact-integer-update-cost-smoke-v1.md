# Exact-integer update-cost smoke v1

Protocol id: `heldout-exact-integer-update-cost-smoke-v1`.

The exact-integer v2 mlx gate returned `nondegenerate_base_signal` (29/64). That result authorizes this smoke only: one optimizer update, a save, a reload, and a resource record. No smoke run is retained in this tree. This file does not authorize a multi-arm comparison.

Protocol file: `protocols/heldout_exact_integer_update_cost_smoke_v1.json`. SHA-256 `0d6959238b63b3b9a08b8db49c52a0905bbc51f617cd4b1ba8d5241da3c940d0`.

The runner refuses to start if those bytes change. It also checks that the v2 manifest is still `ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278`. There is no step-count flag.

## What one update uses

The training row is the first train-split seed from [integer seed split v1](integer-seed-split-v1.md):

| Field | Value |
| --- | --- |
| Seed | `1029953` |
| Expression | `23544 + 19091 - 10702` |
| Completion | `#### 31933` |

That expression is not in the 64-task pack `189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc`. The pack is not scored. The prompt tokens are masked. The loss is the completion tokens only.

The update is Adam, learning rate `1e-5`, batch size 1, one step, LoRA rank 8, scale 20, dropout 0, on the last block's `self_attn.q_proj` and `self_attn.v_proj`. Seed `20261015`. Maximum sequence length 256.

| Backend | Repository | Revision | What differs |
| --- | --- | --- | --- |
| `mlx` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | `4f83f8f146fdf28b512a06562b671d7af4fab457` | Published 4-bit weights. `mlx-lm==0.32.0` `train` with `val_dataset=None`. |
| `transformers` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Upstream checkpoint, float16 on CUDA and float32 otherwise. `torch.optim.Adam` and one PEFT LoRA step on `q_proj` and `v_proj`. |

These are separate smokes. A 4-bit MLX record and a float16 CUDA record are not replicates. `mlx-lm`, `peft`, and `torch` are not package dependencies of this repository.

A passing record has `optimizer_steps == 1`, a finite loss, trainable parameters that changed, a reload that matches the saved tensors, `heldout_pack_used_for_update == false`, a finite non-negative `wall_seconds`, and `peak_memory_bytes > 0`. MLX peak memory is the allocator peak reported by `mlx-lm` (`get_peak_memory() / 1e9`, stored again as bytes). The CUDA peak is reset before the weights load and read after the step. On Linux without CUDA, the figure is `ru_maxrss * 1024`.

Anything else is a failed smoke. Keep `smoke_result.json`. Do not change the learning rate, rank, keys, train row, or step count after reading the loss.

## Mac

Run this on the machine that will execute the mlx update. Do not point `--output` at a v1 or v2 result directory.

```bash
git rev-parse HEAD
python3 - <<'PY'
import hashlib
from pathlib import Path
root = Path('.')
expected = {
    'protocols/heldout_exact_integer_update_cost_smoke_v1.json': '0d6959238b63b3b9a08b8db49c52a0905bbc51f617cd4b1ba8d5241da3c940d0',
    'benchmarks/heldout/exact_integer_v1/tasks.jsonl': '189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc',
    'results/heldout-exact-integer-base-gate-v2/m1-mlx-89f671c/manifest.json': 'ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278',
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
python -m pytest tests/test_update_cost_smoke_v1.py tests/test_format_vs_correctness_v1.py -q
mkdir -p "$HOME/vare-exact-integer-update-cost-smoke-v1"
python scripts/run_exact_integer_update_cost_smoke_v1.py \
  --backend mlx \
  --output "$HOME/vare-exact-integer-update-cost-smoke-v1"
echo "smoke_exit=$?"
python scripts/format_vs_correctness_comparison_v1.py
echo "comparison_exit=$?"
```

`comparison_exit` must be 2. The comparison script does not read `smoke_result.json`.

## Colab

Use a GPU runtime. The transformers command below is the T4-sized path: one layer, batch size 1, sequence length 256, float16 when CUDA is present. It downloads the upstream 3B checkpoint (about 6.2 GB). If the runtime resets, files under `/content` are gone, so the cells mount Drive before the update.

`notebooks/heldout_exact_integer_update_cost_smoke_v1.ipynb` is the same sequence. Set `VARE_REV` to the commit you intend to run. The default is `cursor/exact-integer-update-cost-smoke-af75`.

```python
import os
from pathlib import Path

os.environ.setdefault("VARE_REPO", "https://github.com/mitukx/VARE.git")
os.environ.setdefault("VARE_REV", "cursor/exact-integer-update-cost-smoke-af75")
print(os.environ["VARE_REPO"], os.environ["VARE_REV"])
```

```python
import os
import subprocess
from pathlib import Path

repo = Path("/content/vare-update-cost-smoke-v1")
if not (repo / ".git").exists():
    subprocess.check_call(["git", "clone", os.environ["VARE_REPO"], str(repo)])
subprocess.check_call(["git", "fetch", "origin", os.environ["VARE_REV"]], cwd=repo)
subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
```

```python
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        "/content/vare-update-cost-smoke-v1",
        "transformers",
        "accelerate",
        "peft",
        "huggingface_hub",
    ]
)
```

```python
import os
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
root = Path("/content/drive/MyDrive/vare")
root.mkdir(parents=True, exist_ok=True)
output = root / "heldout-exact-integer-update-cost-smoke-v1-transformers"
output.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(root / "huggingface-cache")
os.environ["HF_HUB_CACHE"] = str(root / "huggingface-cache" / "hub")
print("output", output)
```

```python
import hashlib
import subprocess
import sys
from pathlib import Path

repo = Path("/content/vare-update-cost-smoke-v1")
expected = {
    "protocols/heldout_exact_integer_update_cost_smoke_v1.json": "0d6959238b63b3b9a08b8db49c52a0905bbc51f617cd4b1ba8d5241da3c940d0",
    "benchmarks/heldout/exact_integer_v1/tasks.jsonl": "189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc",
    "results/heldout-exact-integer-base-gate-v2/m1-mlx-89f671c/manifest.json": "ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278",
}
for name, digest in expected.items():
    got = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    print(name, got)
    assert got == digest
completed = subprocess.run(
    [
        sys.executable,
        str(repo / "scripts/run_exact_integer_update_cost_smoke_v1.py"),
        "--backend",
        "transformers",
        "--output",
        str(output),
    ],
    cwd=repo,
)
print("smoke_exit", completed.returncode)
refused = subprocess.run(
    [sys.executable, str(repo / "scripts/format_vs_correctness_comparison_v1.py")],
    cwd=repo,
)
print("comparison_exit", refused.returncode)
```

`comparison_exit` must be 2. Do not install `mlx` on Colab for this protocol.

## After a run

Hash `smoke_result.json` and `environment.json`. Adapter files at or above 1,048,576 bytes stay outside git; record their SHA-256 instead of tracking the bytes. A later commit can retain the small JSON the same way the v2 manifest retains the gate. This protocol has no retained smoke yet.

## What a passing smoke does not establish

One finite update that reloads is a resource and serialization check. It is not a held-out task-success gain. It does not authorize [the five-arm comparison](format-vs-correctness-comparison-v1.md). That entry point exits 2.
