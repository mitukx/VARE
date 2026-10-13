# Format versus correctness comparison v2

Protocol id: `format-vs-correctness-comparison-v2`.

The exact-integer v2 base gate was non-degenerate, and the [update-cost smoke](heldout-exact-integer-update-cost-smoke-v1.md) was accepted. Those two records are what allow this comparison. No arm has been run. `scripts/format_vs_correctness_comparison_v1.py` still exits 2.

Protocol file: `protocols/format_vs_correctness_comparison_v2.json`. SHA-256 `8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4`.

The runner refuses to start if those bytes, the smoke manifest `8926c6c6383201bb3f1a067707210db7abd94c90a10f40ae0686784bc4ce511e`, the 64-task pack, or the confirm split change. There is no step-count flag.

## What is frozen

| Piece | Value |
| --- | --- |
| Arms | `base`, `format_sft`, `format_reward_rl`, `correctness_reward_rl`, `random_reward_rl` |
| Run seeds | `20261016` through `20261020` |
| Optimizer | Adam, learning rate `1e-5`, batch size 1, 16 steps |
| LoRA | Rank 8, scale 20, dropout 0, last block `self_attn.q_proj` and `self_attn.v_proj` |
| Train rows | The 16 train-split seeds after `1029953`. That seed was consumed by the smoke and is not reused. |
| `format_sft` target | The correct strict line `#### <value>` |
| Reward groups | 2 samples. None are dropped. Advantage is the population-standardized group difference, with epsilon `1e-8`. A tied group has advantage 0 and the step still runs. |
| `random_reward_rl` | Bernoulli 0.5 from a frozen seed. The draw does not read the completion. |
| Decoder | Temperature 0.7, at most 48 new tokens, strict final-line parser |
| Eval samples | 4 per task |
| Curve | Unbiased pass@1 and pass@4 at steps 8 and 16. `base` is step 0. |

The four updated arms each take 16 optimizer steps. That is the matched budget. `base` takes none. Reward arms also draw 480 training completions in total. `format_sft` does not. The retained smoke wall clock is 8.762955541998963 seconds for weight load, one step, and reload, with peak allocator memory 1959434978 bytes. This protocol uses 9 seconds per optimizer step as the planning bound that measurement supports. It is not a measured marginal step time. Under that bound, 4 × 5 × 16 × 9 = 2880 seconds of optimizer steps. Generation time was not measured.

The decision set is the committed confirm split, all 64 seeds. It is not used for training or early stopping. Dev is unused. The primary contrast is `correctness_reward_rl` minus `format_sft` on unbiased strict pass@1, pooling the five run seeds within each confirm task, with a paired bootstrap of 2000 resamples, seed `20261016`, alpha 0.05. The decision is `positive_contrast` when the interval lies above 0, `negative_contrast` when it lies below 0, and `interval_includes_zero` otherwise. McNemar's exact test and the lenient extractor are reported and do not decide.

The 64-task pack is scored at the final checkpoint only. Its curve cannot change the decision. The pack's earlier 29/64 base rate is not a threshold. The train split is `a + b - c` in one integer band. The pack mixes four expression families, so a pack curve is a transfer measurement.

`mlx` uses `mlx-community/Qwen2.5-3B-Instruct-4bit` at `4f83f8f146fdf28b512a06562b671d7af4fab457` with `mlx-lm==0.32.0`. `transformers` uses `Qwen/Qwen2.5-3B-Instruct` at `aa8e72537993ba99e69dfaafa59ed015b17504d1`, float16 on CUDA. The backends are not replicates. A resumed arm reloads the last contiguous checkpoint and its Adam state. A gap in the step numbers fails closed. A resumed process keeps completed rows in `progress.jsonl`.

A positive contrast does not promote a checkpoint. Seeing the interval does not authorize a larger step count. That would be a new protocol.

## Mac

```bash
git rev-parse HEAD
python3 - <<'PY'
import hashlib
from pathlib import Path
root = Path('.')
expected = {
    'protocols/format_vs_correctness_comparison_v2.json': '8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4',
    'results/heldout-exact-integer-update-cost-smoke-v1/m1-mlx-f22b3b9/manifest.json': '8926c6c6383201bb3f1a067707210db7abd94c90a10f40ae0686784bc4ce511e',
    'benchmarks/heldout/exact_integer_v1/tasks.jsonl': '189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc',
    'benchmarks/heldout/integer_seed_split_v1/splits.json': '4544a9a852f615f7cf9c52d7a4826ed70ea6c5f57829754ac2e10f3c1fea4287',
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
python -m pytest tests/test_format_vs_correctness_v2.py -q
mkdir -p "$HOME/vare-format-vs-correctness-v2"
python scripts/run_format_vs_correctness_comparison_v2.py \
  --backend mlx \
  --output "$HOME/vare-format-vs-correctness-v2" \
  --check
python scripts/run_format_vs_correctness_comparison_v2.py \
  --backend mlx \
  --output "$HOME/vare-format-vs-correctness-v2"
echo "comparison_exit=$?"
```

`--check` prints the unit counts and does not train. The second command is the run. If it stops, run that same command again in the same output directory. Do not point it at a gate or smoke directory. `summary.json` is written only after 320 train rows and 17920 eval rows. `progress.jsonl` can exceed 1,048,576 bytes. Do not track it. Record its SHA-256. The same limit applies to checkpoint files.

## Colab

Use a GPU runtime. This is the transformers backend, not a replicate of the 4-bit mlx run. An out-of-memory failure is a failed run of this protocol. Do not switch the checkpoint to 4-bit inside it. `notebooks/format_vs_correctness_comparison_v2.ipynb` is the same sequence. The default revision is `cursor/format-vs-correctness-v2-af75`.

```python
import os
os.environ.setdefault("VARE_REPO", "https://github.com/mitukx/VARE.git")
os.environ.setdefault("VARE_REV", "cursor/format-vs-correctness-v2-af75")
print(os.environ["VARE_REPO"], os.environ["VARE_REV"])
```

```python
import os, subprocess
from pathlib import Path
repo = Path("/content/vare-format-vs-correctness-v2")
if not (repo / ".git").exists():
    subprocess.check_call(["git", "clone", os.environ["VARE_REPO"], str(repo)])
subprocess.check_call(["git", "fetch", "origin", os.environ["VARE_REV"]], cwd=repo)
subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
```

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-e",
    "/content/vare-format-vs-correctness-v2",
    "transformers", "accelerate", "peft", "huggingface_hub",
])
```

```python
import os
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
root = Path("/content/drive/MyDrive/vare")
root.mkdir(parents=True, exist_ok=True)
output = root / "format-vs-correctness-comparison-v2-transformers"
output.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(root / "huggingface-cache")
os.environ["HF_HUB_CACHE"] = str(root / "huggingface-cache" / "hub")
print("output", output)
```

```python
import hashlib, subprocess, sys
from pathlib import Path
repo = Path("/content/vare-format-vs-correctness-v2")
expected = {
    "protocols/format_vs_correctness_comparison_v2.json": "8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4",
    "results/heldout-exact-integer-update-cost-smoke-v1/m1-mlx-f22b3b9/manifest.json": "8926c6c6383201bb3f1a067707210db7abd94c90a10f40ae0686784bc4ce511e",
    "benchmarks/heldout/exact_integer_v1/tasks.jsonl": "189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc",
    "benchmarks/heldout/integer_seed_split_v1/splits.json": "4544a9a852f615f7cf9c52d7a4826ed70ea6c5f57829754ac2e10f3c1fea4287",
}
for name, digest in expected.items():
    got = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    print(name, got)
    assert got == digest
checked = subprocess.run([
    sys.executable, str(repo / "scripts/run_format_vs_correctness_comparison_v2.py"),
    "--backend", "transformers", "--output", str(output), "--check",
], cwd=repo)
print("check_exit", checked.returncode)
completed = subprocess.run([
    sys.executable, str(repo / "scripts/run_format_vs_correctness_comparison_v2.py"),
    "--backend", "transformers", "--output", str(output),
], cwd=repo)
print("comparison_exit", completed.returncode)
```

Run the last cell again after a disconnect. Completed rows in that Drive directory are skipped.
