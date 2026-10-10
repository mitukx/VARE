# Colab runbook: DeepMath base inference gate v4

This runbook is for the single, inference-only TRAIN gate in [protocol v4](../protocols/qwen25_deepmath_grpo_math500_v4.lock.json). This gate performs no optimizer update and does not load MATH-500. Use only a Colab runtime that is explicitly available at zero additional cost. If Colab offers a paid compute option, upgrade, or purchase, stop without selecting it.

CPU preflight on 2026-10-11 reran `scripts.validate_math500_study_v1` against the already retained frozen data; it passed and produced base-gate fingerprint `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`, exactly matching the protocol's frozen 32-ID expectation. The validator and GPU runner use the shared `scripts/prompt_id_hash_v3.py` serializer. No IDs or split rules changed.

Protocol v4 corrects the runner's tokenizer-config hash from the v3 typo. The expected digest was checked against the bytes downloaded from the exact pinned model revision; v1/v2/v3 protocol and result files remain unchanged.

## 1. Select and verify a runtime

Choose a Colab runtime with Python 3.11 or 3.12, PyTorch 2.11.0, and exactly one CUDA GPU with at least 16 GiB. The gate requires the exact package versions in `requirements/math500-study-gpu-gate.txt`; its CUDA build must work with the Colab host driver. The runner fails closed on a version, device-count, or VRAM mismatch.

Mount Google Drive and keep the clone, downloaded assets, and run artifacts in separate locations. The code checkout must remain clean for every start and resume; do not save artifacts inside it.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export VARE_REV='f9f5485172b13b5e6cfd1b05a245b6bac47cc899'
export VARE_DIR=/content/vare
export VARE_DRIVE=/content/drive/MyDrive/VARE/deepmath-gate-v4
export HF_HOME=/content/drive/MyDrive/VARE/huggingface-cache
mkdir -p "$VARE_DRIVE" "$HF_HOME"
git clone https://github.com/mitukx/VARE.git "$VARE_DIR"
git -C "$VARE_DIR" checkout "$VARE_REV"
cd "$VARE_DIR"
git rev-parse HEAD | tee "$VARE_DRIVE/code-revision.txt"
git status --porcelain --untracked-files=all
```

The final `git status` must print nothing. Keep `VARE_REV` identical for every resume. The runner independently records Git HEAD and source-file hashes in its resume identity.

## 2. Install the pinned environment without replacing Colab's CUDA PyTorch

First check the runtime's installed torch. The gate needs `torch==2.11.0`; do not let a generic pip install replace Colab's CUDA-enabled build with a different wheel. Install the remaining exact requirements separately, then verify all versions.

```bash
cd "$VARE_DIR"
python --version
python - <<'PY'
import torch
print('torch', torch.__version__)
print('CUDA build', torch.version.cuda)
assert torch.__version__.split('+')[0] == '2.11.0'
assert torch.cuda.is_available()
PY
grep -v '^torch==' requirements/math500-study-gpu-gate.txt > /tmp/vare-gate-requirements.txt
python -m pip install -r /tmp/vare-gate-requirements.txt
python -m pip freeze | tee "$VARE_DRIVE/pip-freeze.txt"
```

If the preinstalled CUDA torch is not 2.11.0, stop and choose a compatible Colab runtime. Do not improvise a different torch/CUDA combination. Verify the gate's exact package versions before downloading or running:

```bash
python - <<'PY'
import importlib.metadata as m
import platform
import torch
assert (3, 11) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13)
assert torch.__version__.split('+')[0] == '2.11.0'
for package, expected in {
    'transformers':'5.5.4', 'trl':'1.1.0', 'accelerate':'1.13.0',
    'datasets':'4.8.4', 'math-verify':'0.9.0', 'pyarrow':'22.0.0',
}.items():
    observed = m.version(package).split('+')[0]
    assert observed == expected, (package, observed, expected)
print('Python', platform.python_version())
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
PY
```

## 3. Download and verify only gate assets

Use the revisions and file hashes in v4. The dataset command fetches only the pinned TRAIN parquet. Do not download or open MATH-500 in this inference gate.

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
```

Confirm every model hash against `model.model_files_sha256` in v4 and the parquet hash against `training_data.sha256`. The runner repeats these checks before CUDA validation or model loading and verifies the fixed 32-ID cohort hash. Any missing or changed asset is a hard stop.

```bash
sha256sum "$VARE_DRIVE/model"/{config.json,generation_config.json,merges.txt,model.safetensors,tokenizer.json,tokenizer_config.json,vocab.json}
sha256sum "$VARE_DRIVE/data-ready/deepmath-train.parquet"
```

## 4. Confirm one-GPU capacity

This inspection does not load the model or perform inference. Stop if there is not exactly one visible GPU or if total VRAM is below 16 GiB. Save the device and driver details with the run.

```bash
nvidia-smi | tee "$VARE_DRIVE/nvidia-smi-before.txt"
python - <<'PY' | tee "$VARE_DRIVE/cuda-device.txt"
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
device = torch.cuda.get_device_properties(0)
print(device.name, device.total_memory, torch.cuda.get_device_capability(0))
assert device.total_memory >= 16 * 1024**3
PY
```

## 5. Run the frozen inference-only gate

Run from the clean checkout. The 7,200-second cap includes model loading, all attempts, scoring, and time between restarts. The cap is charged as cumulative wall time. The 131,072-token ceiling is the sum of generated token IDs, including each first EOS and excluding trailing pad IDs. Checkpoints and state are written directly to Google Drive after each complete prompt group.

```bash
set -o pipefail
cd "$VARE_DIR"
python -m scripts.run_qwen25_deepmath_base_gate_v4 \
  --model-dir "$VARE_DRIVE/model" \
  --data-dir "$VARE_DRIVE/data-ready" \
  --output "$VARE_DRIVE/gpu-base-gate-v4.json" \
  --max-wall-seconds 7200 \
  --max-generated-tokens 131072 \
  2>&1 | tee -a "$VARE_DRIVE/console.log"
```

The durable files are `gpu-base-gate-v4.json` on completion, `gpu-base-gate-v4.json.partial.jsonl` after each completed group, and `gpu-base-gate-v4.json.state.json` for decision, identity, cumulative elapsed/token totals, and maximum allocated/reserved VRAM. The partial JSONL retains raw generated token IDs and is fsynced per group.

Each completion record includes the fixed prompt ID, raw generated token IDs, decoded text, ID-counted generated length, training reward, independent exact-checker result, final-box validity, EOS status, truncation status, and reward/checker disagreement. The final report summarizes mean/min/max length, EOS and truncation rates, mismatch count, task success, reward variance, cumulative tokens/time/throughput, and peak memory.

If Colab interrupts or OOMs, retain all three files. Correct only an execution resource issue, reconnect the same Drive, checkout the exact `VARE_REV`, restore the exact pinned packages/runtime, and run the same command with `--resume` appended. Resume rejects changed protocol, budget, code revision/source hashes, model/data hashes, cohort, generation settings, or runtime/device identity. It skips completed groups. A partially generated group is rerun; its incomplete token IDs are not counted. Time between attempts is charged to the original 7,200-second wall budget.

## Stop conditions

Stop and preserve logs/state if any integrity hash differs; the gate finds a different prompt-ID cohort; Python/package/CUDA/GPU requirements fail; peak reserved memory exceeds the frozen cap; the wall or generated-token ceiling is reached; the checker/reward contract errors; or the frozen success/format/reward-variance criteria fail. Do not edit prompts, increase a ceiling, switch runtime/model/data, or retry the same cohort after a statistical/task-success stop.

This gate uses exactly 32 fixed TRAIN prompts, four sampled completions each, and zero optimizer updates. It does not demonstrate learning, MATH-500 performance, or capability improvement. A passing gate permits only a separate request for the one-step update/save-reload resource smoke specified by the protocol.
