# Colab runbook: DeepMath base inference gate v5

This is the single inference-only TRAIN gate specified by [protocol v5](../protocols/qwen25_deepmath_grpo_math500_v5.lock.json). It runs 32 frozen TRAIN prompts with four sampled completions per prompt. It performs zero optimizer updates and does not load MATH-500. v1-v4 protocols, results, and runbooks remain immutable.

**Execution status:** v5 was stopped before model/data download because the exact pinned package installation on Colab 2026.07 failed `pip check`: preinstalled `omegaconf 2.3.1` conflicts with the frozen ANTLR 4.13.2 pin, and preinstalled IPython reports missing `jedi`. Do not rerun v5. See the [preflight report](qwen25-deepmath-colab-gpu-gate-v5-preflight-report.md).

The only scientific environment changes from v4 are GPU eligibility and selected Colab runtime. v5 sets minimum total CUDA VRAM to 15,000,000,000 bytes, below the previously observed free T4 value 15,637,086,208 bytes. The peak reserved-memory cap remains `min(14 GiB, 0.90 × CUDA total VRAM)`; on that observed T4 it is 14,073,377,587 bytes. All model/data hashes, prompt IDs, seed, reward, grader, chat input, generation settings, budgets, and PASS criteria are unchanged.

## 1. Use the free T4 and the pinned past runtime

In the v5 Colab notebook, open **Runtime → Change runtime type** and select **T4 GPU** plus runtime version **2026.07**. This past-runtime option was visible in Colab on 2026-10-11 and is selected on the v5 notebook. The intended runtime is Python 3.12.13 with the exact package pins below. Connect only if Colab offers the T4 with no purchase, upgrade, or paid-compute prompt. Do not select G4, L4, A100, or a premium runtime. If free T4 is unavailable, stop: do not switch hardware or request a paid upgrade.

Mount the user's Google Drive and store the clone, downloaded assets, and run files separately. The v5 notebook is a Drive copy of the v4 notebook; v4 remains intact. Store all experiment artifacts on Drive so Colab disconnects do not discard completed groups.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export VARE_REV='19dc779888798d5110a457983732af498318a815'
export VARE_DIR=/content/vare-v5
export VARE_DRIVE=/content/drive/MyDrive/VARE/deepmath-gate-v5
export HF_HOME=/content/drive/MyDrive/VARE/huggingface-cache
mkdir -p "$VARE_DRIVE" "$HF_HOME"
git clone https://github.com/mitukx/VARE.git "$VARE_DIR"
git -C "$VARE_DIR" checkout "$VARE_REV"
cd "$VARE_DIR"
git rev-parse HEAD | tee "$VARE_DRIVE/code-revision.txt"
test "$(git rev-parse HEAD)" = "$VARE_REV"
test -z "$(git status --porcelain --untracked-files=all)"
```

The final two checks must pass. Use this exact `VARE_REV` on every resume. Runner identity also binds the protocol lock, all source hashes, budgets, model/data/cohort, Python and pinned package versions, CUDA runtime/driver, and GPU identity/memory.

## 2. Verify the free GPU and exact runtime before downloads

Do not run an inference cell yet. Capture the environment and confirm the gate pins. Do not install packages before confirming the runtime's existing CUDA-enabled PyTorch.

```bash
python --version | tee "$VARE_DRIVE/python-version.txt"
python - <<'PY' | tee "$VARE_DRIVE/runtime-preflight.txt"
import platform, torch, subprocess
print('python', platform.python_version())
print('torch', torch.__version__, 'CUDA build', torch.version.cuda)
assert (3, 12) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13)
assert torch.__version__.split('+')[0] == '2.11.0'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
print('device', p.name, 'total_vram_bytes', p.total_memory, 'capability', torch.cuda.get_device_capability(0))
assert 'T4' in p.name and p.total_memory >= 15_000_000_000
print(subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True).stdout)
PY
```

The runtime must be Python 3.12.x, PyTorch 2.11.0 with CUDA, one Tesla T4, and at least 15,000,000,000 CUDA memory bytes. For the previously seen T4 capacity, the runner applies a reserved-memory ceiling of 14,073,377,587 bytes. It refuses to run above that derived ceiling. Do not change either memory limit if model loading or generation OOMs.

Install only the other packages at the exact versions in the locked requirements, then verify resolver consistency. This must not replace the CUDA Torch wheel. If an exact pin is unavailable or `pip check` fails, stop without inference; do not try alternate versions.

```bash
cd "$VARE_DIR"
grep -v '^torch==' requirements/math500-study-gpu-gate.txt > "$VARE_DRIVE/gate-requirements-no-torch.txt"
python -m pip install -r "$VARE_DRIVE/gate-requirements-no-torch.txt"
python -m pip check
python -m pip freeze | tee "$VARE_DRIVE/pip-freeze.txt"
python - <<'PY'
import importlib.metadata as m
import platform, torch
expected = {
    'torch':'2.11.0', 'transformers':'5.5.4', 'trl':'1.1.0',
    'accelerate':'1.13.0', 'datasets':'4.8.4', 'math-verify':'0.9.0',
    'latex2sympy2-extended':'1.11.0', 'antlr4-python3-runtime':'4.13.2',
    'pyarrow':'22.0.0',
}
assert (3, 12) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13)
for name, want in expected.items():
    got = m.version(name).split('+')[0]
    assert got == want, (name, got, want)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('exact pinned packages and one CUDA device verified')
PY
```

Save a full device/runtime record:

```bash
nvidia-smi | tee "$VARE_DRIVE/nvidia-smi-before.txt"
```

## 3. Download only pinned model and TRAIN data, then verify

Use only the file list and hashes in v5. Download the fixed model revision and TRAIN parquet into Drive. Do not download or read MATH-500 in this gate.

```bash
mkdir -p "$VARE_DRIVE/model" "$VARE_DRIVE/data-ready"
MODEL_REV=7ae557604adf67be50417f59c2c2f167def9a775
for file in config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json; do
  curl --fail --location --retry 3 \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/${MODEL_REV}/${file}" \
    --output "$VARE_DRIVE/model/$file"
done
curl --fail --location --retry 3 \
  'https://huggingface.co/datasets/trl-lib/DeepMath-103K/resolve/066c50a88d4e14cefc056e31111db2dba17f6c68/data/train-00000-of-00001.parquet' \
  --output "$VARE_DRIVE/data-ready/deepmath-train.parquet"
sha256sum "$VARE_DRIVE/model"/* | tee "$VARE_DRIVE/model-sha256.txt"
sha256sum "$VARE_DRIVE/data-ready/deepmath-train.parquet" | tee "$VARE_DRIVE/train-data-sha256.txt"
```

Compare all seven model digests and the parquet digest with [the v5 lock](../protocols/qwen25_deepmath_grpo_math500_v5.lock.json). The runner independently verifies every model file and the TRAIN parquet before model loading. The pre-registered 32-prompt ID fingerprint is `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`; both the CPU validator and runner use the same serializer. Any mismatch is a hard stop.

## 4. Confirm the prompt and runtime contract

Run this TRAIN-only preflight against the downloaded parquet. Do not use `validate_math500_study_v1.py` here: that broader validator also opens MATH-500. This check imports its frozen TRAIN deduplication/split functions but reads no MATH-500 file, and verifies the exact frozen fingerprint before GPU inference.

```bash
cd "$VARE_DIR"
python - "$VARE_DRIVE/data-ready/deepmath-train.parquet" <<'PY' | tee "$VARE_DRIVE/cpu-train-preflight.json"
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from scripts.prompt_id_hash_v3 import verify_prompt_id_sha256
from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256, deduplicate_train_rows, select_splits, sha256_file,
)
protocol = json.load(open('protocols/qwen25_deepmath_grpo_math500_v5.lock.json'))
data = Path(sys.argv[1])
assert sha256_file(data) == EXPECTED_DEEPMATH_SHA256
rows = pq.read_table(data, columns=['prompt', 'solution']).to_pylist()
eligible, stats = deduplicate_train_rows(rows)
selected = select_splits([], eligible)['base_gate_ids']
fingerprint = verify_prompt_id_sha256(
    selected, protocol['training_data']['sample']['base_gate_id_list_sha256'], expected_count=32,
)
print(json.dumps({'status':'pass', 'train_sha256':EXPECTED_DEEPMATH_SHA256,
                  'gate_prompt_ids_sha256':fingerprint, 'deduplication':stats}, indent=2))
PY
```

The locked prompt is the original single user message, with no system prompt. The runner's tokenizer call and stochastic generation settings match the pinned TRL 1.1.0 GRPO input contract: model chat template, `add_generation_prompt=True`, temperature 0.7, top-p 0.95, top-k 0, and 1,024 maximum new tokens. Four completions are sampled per prompt. Do not edit the prompt or generation parameters.

## 5. Run exactly one inference-only gate

Before starting, ensure the Drive path has sufficient space. All 32 prompt groups, raw generated token IDs, decoded completions, training rewards, independent checker results, format flags, per-group reward variances, and resumable state are written to Drive. No optimizer update occurs; no MATH-500 file is loaded.

```bash
set -o pipefail
cd "$VARE_DIR"
python -m scripts.run_qwen25_deepmath_base_gate_v5 \
  --model-dir "$VARE_DRIVE/model" \
  --data-dir "$VARE_DRIVE/data-ready" \
  --output "$VARE_DRIVE/gpu-base-gate-v5.json" \
  --max-wall-seconds 7200 \
  --max-generated-tokens 131072 \
  2>&1 | tee -a "$VARE_DRIVE/console.log"
```

The fixed ceilings are cumulative: 7,200 UTC seconds from the first journal creation through model load, generation, scoring, pauses, and resumed attempts; 131,072 generated token IDs across complete prompt groups. After an interruption/OOM only, resume the same command by adding `--resume`, and only if the same commit, protocol, files, Python/packages, CUDA driver/runtime, T4 memory identity, budget, model/data, and Drive journal are available. The runner rejects identity changes; it does not reset elapsed time, token count, or peak VRAM. Do not resume after a completed statistical/task-success FAIL.

## Frozen PASS criteria and stop conditions

PASS requires all of the following as fixed in v5: independent success 5–90%; valid final-box rate at least 90%; at least 6/32 mixed-reward groups; positive mean within-group reward variance; and every time, generated-token, and reserved-VRAM limit satisfied. The reserved-VRAM cap is `min(14 GiB, 0.90 × total CUDA memory)`. Report the 32-group reward distribution, reward/checker disagreements, completion lengths, EOS/truncation, throughput, and peak allocated/reserved memory.

Stop and preserve logs/state if any asset or cohort hash differs, exact runtime/dependencies are unavailable, the CUDA device is not one free T4 meeting the fixed threshold, the runtime identity changes, the checker/reward errors, any resource budget expires, OOM occurs, or any statistical PASS criterion fails. Do not alter thresholds, seed, prompts, settings, or try again with a second gate. An OOM may be resumed only under the exact same identity and cumulative budgets; if it recurs or cannot resume safely, report FAIL.

A PASS only establishes that this fixed base-model inference produced usable reward variation within the measured T4 inference budget. It does not establish GRPO training feasibility, learning, generalization, or capability improvement. After the result is preserved, stop and request separate authorization before any optimizer update, SFT, GRPO training, extra seed, or MATH-500 evaluation.
