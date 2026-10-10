# Colab runbook: DeepMath base inference gate v6

v6 is the final isolated-environment attempt for the fixed inference-only gate. v5 remains frozen as an environment NO-GO: global Colab `pip check` found preinstalled-package conflicts, and no model inference ran. v6 changes Python dependency isolation and environment provenance only. It does not change the model, TRAIN parquet, prompt IDs, reward, independent checker, seed, sampling, limits, or pass thresholds.

**Do not rerun v5.** If any v6 preflight below fails, preserve the logs and stop; do not create another protocol revision or switch services.

## Fixed experiment

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Data: `trl-lib/DeepMath-103K`, fixed TRAIN parquet only
- Cohort: 32 frozen TRAIN prompts, 4 completions each, maximum 128 completions
- Generation: seed 20261012 + prompt index, temperature 0.7, top-p 0.95, top-k 0, max 1,024 new tokens, user-only input, no system prompt, model chat template
- Reward/checker: pinned TRL 1.1.0 `accuracy_reward`; independent v2 checker
- Limits: 131,072 actual generated token IDs, cumulative 7,200 wall seconds, reserved VRAM at most `min(14 GiB, 90% of device total)`
- Optimizer updates: 0. MATH-500 must not be downloaded or loaded.
- PASS: independent success 5–90%, valid final-box rate at least 90%, at least 6/32 mixed-reward groups, positive mean group reward variance, and all resource limits met.

## 1. Start the free Colab T4 runtime

Use the existing v5 notebook as a reference or make a Drive copy titled **VARE DeepMath Gate v6**. Select the free Tesla T4 runtime and Python 3.12 runtime. Do not accept purchase, upgrade, or paid-compute options. Mount Drive and retain all durable assets under `MyDrive/VARE/deepmath-gate-v6`.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export VARE_REV='4ad878fc90f71b925ea76fa075f17d034e758836'
export VARE_DIR=/content/vare-v6
export VARE_DRIVE=/content/drive/MyDrive/VARE/deepmath-gate-v6
export VENV_DIR=/content/vare-v6-venv
export HF_HOME=/content/drive/MyDrive/VARE/huggingface-cache
mkdir -p "$VARE_DRIVE" "$HF_HOME"
git clone https://github.com/mitukx/VARE.git "$VARE_DIR"
git -C "$VARE_DIR" checkout "$VARE_REV"
cd "$VARE_DIR"
git rev-parse HEAD | tee "$VARE_DRIVE/code-revision.txt"
test "$(git rev-parse HEAD)" = "$VARE_REV"
test -z "$(git status --porcelain --untracked-files=all)"
```

The run uses this exact clean source revision. Keep the virtual environment, model, data, and outputs outside the clone. A Colab restart may require rebuilding the venv at the same path; resume only after the runner confirms identical environment identity.

## 2. Create a genuinely isolated Python environment

Record the host runtime and free GPU before installing anything. The host must report Python 3.12.x, one Tesla T4, CUDA available, and at least 15,000,000,000 total VRAM bytes. The host CUDA driver is used by the isolated CUDA-enabled PyTorch wheel.

```bash
python --version | tee "$VARE_DRIVE/host-python.txt"
nvidia-smi | tee "$VARE_DRIVE/nvidia-smi-before.txt"
python - <<'PY' | tee "$VARE_DRIVE/host-runtime.txt"
import platform, torch
print('python', platform.python_version())
print('host torch', torch.__version__, 'CUDA build', torch.version.cuda)
print('CUDA available', torch.cuda.is_available(), 'device count', torch.cuda.device_count())
assert (3, 12) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
print('device', p.name, 'total_vram_bytes', p.total_memory, 'capability', torch.cuda.get_device_capability(0))
assert 'T4' in p.name and p.total_memory >= 15_000_000_000
PY
```

Create the venv **without** `--system-site-packages`, disable user packages, and install the fixed CUDA wheel and remaining packages inside it. The host's preinstalled OmegaConf/IPython packages are not modified or inherited.

```bash
python -m venv "$VENV_DIR"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128' \
  2>&1 | tee "$VARE_DRIVE/venv-torch-install.log"
awk '!/^torch==/' requirements/math500-study-gpu-gate-v6.txt > "$VARE_DRIVE/gate-requirements-no-torch.txt"
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple -r "$VARE_DRIVE/gate-requirements-no-torch.txt" \
  2>&1 | tee "$VARE_DRIVE/venv-dependencies-install.log"
"$VENV_DIR/bin/python" -m pip check 2>&1 | tee "$VARE_DRIVE/venv-pip-check.log"
"$VENV_DIR/bin/python" -m pip freeze | tee "$VARE_DRIVE/venv-pip-freeze.txt"
```

Stop if either install fails or venv `pip check` is not clean. Do not change a pin, uninstall a host package, use host `site-packages`, or fall back to CPU/MPS. Verify exact imports, path isolation, pins, and CUDA from the venv itself:

```bash
cd "$VARE_DIR"
"$VENV_DIR/bin/python" - <<'PY' | tee "$VARE_DRIVE/venv-preflight.json"
import importlib, importlib.metadata as md, json, platform, site, sys, torch
from scripts.gpu_gate_venv_v6 import current_environment_provenance
from scripts.gpu_gate_vram_v5 import maximum_reserved_vram_bytes
from scripts.qwen25_deepmath_inputs_v2 import base_gate_generation_config
from scripts.math500_grader_v2 import exact_match, has_valid_final_box
from trl.rewards import accuracy_reward

expected = {
 'torch':'2.11.0', 'transformers':'5.5.4', 'trl':'1.1.0', 'accelerate':'1.13.0',
 'datasets':'4.8.4', 'math-verify':'0.9.0', 'latex2sympy2-extended':'1.11.0',
 'antlr4-python3-runtime':'4.13.2', 'pyarrow':'22.0.0'
}
assert (3,12) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3,13)
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
for name, version in expected.items():
    assert md.version(name).split('+')[0] == version, (name, md.version(name), version)
provenance = current_environment_provenance()
assert provenance['system_site_packages'] is False
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert 'T4' in props.name and props.total_memory >= 15_000_000_000
reserved_cap = maximum_reserved_vram_bytes(props.total_memory, .90)
assert reserved_cap == min(14 * 1024**3, int(.90 * props.total_memory))
# CPU-only reward/checker fixture: no model is loaded and no GPU tensor is used.
rewards = accuracy_reward(
    completions=[[{'role':'assistant','content':r'\boxed{2}'}],
                 [{'role':'assistant','content':r'\boxed{3}'}]],
    solution=['2','2'])
assert len(rewards) == 2 and rewards[0] is not None and rewards[1] is not None
assert bool(rewards[0]) is True and bool(rewards[1]) is False
assert exact_match(r'\boxed{2}', '2') and not exact_match(r'\boxed{3}', '2')
assert has_valid_final_box(r'\boxed{2}') and not has_valid_final_box('no final answer')
print(json.dumps({'status':'pass','python':platform.python_version(),'prefix':sys.prefix,
 'base_prefix':sys.base_prefix,'sys_path':sys.path,'environment':provenance,
 'packages':{name:md.version(name) for name in expected},'torch_cuda':torch.version.cuda,
 'gpu':props.name,'total_vram_bytes':props.total_memory,'reserved_vram_cap_bytes':reserved_cap,
 'cpu_reward_fixture':rewards},indent=2,default=str))
PY
```

The environment report includes `sys.prefix`, `sys.base_prefix`, all `sys.path` entries, and actual files for every required third-party module. The v6 runner repeats the isolation and `pip check` checks and includes this provenance in the resume identity.

## 3. Verify code, protocol, model, data, and prompt IDs

The fixed inference runner refuses to start unless the Git checkout is clean, all execution source hashes equal the v6 lock, model and parquet hashes match, and the exact 32-prompt fingerprint is reproduced. Download only the pinned model files and TRAIN parquet to Drive. Never fetch MATH-500 for this gate.

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

Compare all seven model hashes and the data hash with `protocols/qwen25_deepmath_grpo_math500_v6.lock.json`. The expected weight SHA-256 is `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; the TRAIN parquet SHA-256 is `e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7`. The fixed prompt-ID fingerprint is `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`.

Run the TRAIN-only CPU cohort validator, then perform one more environment and import preflight immediately before generation. Do not load MATH-500 or inspect its examples.

```bash
"$VENV_DIR/bin/python" - "$VARE_DRIVE/data-ready/deepmath-train.parquet" <<'PY' | tee "$VARE_DRIVE/cpu-train-preflight.json"
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from scripts.prompt_id_hash_v3 import verify_prompt_id_sha256
from scripts.validate_math500_study_v1 import EXPECTED_DEEPMATH_SHA256, deduplicate_train_rows, select_splits, sha256_file
protocol=json.load(open('protocols/qwen25_deepmath_grpo_math500_v6.lock.json'))
data=Path(sys.argv[1])
assert sha256_file(data)==EXPECTED_DEEPMATH_SHA256
rows=pq.read_table(data,columns=['prompt','solution']).to_pylist()
eligible,stats=deduplicate_train_rows(rows)
ids=select_splits([],eligible)['base_gate_ids']
fingerprint=verify_prompt_id_sha256(ids,protocol['training_data']['sample']['base_gate_id_list_sha256'],expected_count=32)
print(json.dumps({'status':'pass','train_sha256':EXPECTED_DEEPMATH_SHA256,'prompt_ids_sha256':fingerprint,'deduplication':stats},indent=2))
PY
```

If a venv, package, CUDA, GPU, Git, code hash, asset hash, or cohort check fails, save the output to Drive and stop without launching the runner.

## 4. Execute the one permitted inference gate

Only after every preflight above passes, launch exactly once. Store logs, JSONL journal, state, and final result on Drive. This command performs inference and scoring only: no optimizer update and no MATH-500 access.

```bash
set -o pipefail
cd "$VARE_DIR"
"$VENV_DIR/bin/python" -m scripts.run_qwen25_deepmath_base_gate_v6 \
  --model-dir "$VARE_DRIVE/model" \
  --data-dir "$VARE_DRIVE/data-ready" \
  --output "$VARE_DRIVE/gpu-base-gate-v6.json" \
  --max-wall-seconds 7200 \
  --max-generated-tokens 131072 \
  2>&1 | tee -a "$VARE_DRIVE/console.log"
```

The runner checkpoints each complete prompt group. On interruption/OOM, resume only with `--resume`, the same clean code revision, exact same venv/runtime/GPU/model/data/protocol/budgets, and the same Drive journal. Cumulative wall time, generated token IDs, and peak VRAM remain charged. A completed statistical FAIL ends the pairing; do not rerun it.

## 5. Stop and report

Apply v6's fixed PASS thresholds exactly. Retain the runner JSON, state, partial JSONL, full console log, preflight records, package freeze, and `nvidia-smi` output. Report unmeasured values as unmeasured. Regardless of PASS/FAIL, stop after this gate. No SFT, GRPO update, one-step smoke, MATH-500 evaluation, extra seed, or tuning is authorized by this runbook.
