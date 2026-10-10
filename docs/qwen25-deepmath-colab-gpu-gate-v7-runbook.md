# Colab runbook: DeepMath base inference gate v7

v7 is the single authorized update to v6's final environment attempt. v1–v6 locks and stop records stay frozen. v6 stopped before asset download or inference because Colab Python 3.12.13 lacks `ensurepip`. v7 changes only venv/pip bootstrap and its provenance. The fixed model, data, prompt cohort, reward, independent checker, seed, sampling, thresholds, and cumulative resource budgets are unchanged.

If any v7 preflight fails, retain the logs and stop. Do not create v8 or alter a pin. Do not accept paid compute. This gate performs inference and scoring only: zero optimizer updates, and no MATH-500 access.

## Frozen gate

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Data: `trl-lib/DeepMath-103K`, pinned TRAIN parquet only
- Cohort: 32 frozen TRAIN prompts × 4 sampled completions (128 maximum)
- Sampling: seed 20261012 + prompt index; temperature 0.7; top-p 0.95; top-k 0; max 1,024 new tokens; original user-only prompt; no system message; tokenizer's chat template
- Reward and checker: pinned TRL 1.1.0 `accuracy_reward`; independent v2 checker
- Limits: maximum 131,072 generated token IDs; 7,200 cumulative wall seconds including restarts; peak reserved VRAM no greater than `min(14 GiB, 90% of total VRAM)`
- GPU: one free Tesla T4; minimum total VRAM 15,000,000,000 bytes
- PASS: independent success 5–90%; valid final-box rate ≥90%; at least 6/32 mixed-reward groups; positive mean within-group reward variance; all compute limits met

## 1. Create a v7 notebook copy and durable directories

Use a copy of the existing Colab notebook, titled **VARE DeepMath Gate v7**. Select the free Tesla T4 runtime with Python 3.12. Do not upgrade or purchase compute. Mount Drive and preserve all output under `MyDrive/VARE/deepmath-gate-v7`; keep v6 records untouched.

```python
from google.colab import drive
drive.mount('/content/drive')
```

The protocol is committed before execution. The run uses the exact v7 code/protocol commit `95316981b8d654a60bb5d22a18bf9e5e8c4cc26e`; the later runbook-only commit does not change execution code or the protocol hash.

```bash
export VARE_REV='95316981b8d654a60bb5d22a18bf9e5e8c4cc26e'
export VARE_DIR=/content/vare-v7
export VARE_DRIVE=/content/drive/MyDrive/VARE/deepmath-gate-v7
export VENV_DIR=/content/vare-v7-venv
export HF_HOME=/content/drive/MyDrive/VARE/huggingface-cache
mkdir -p "$VARE_DRIVE" "$HF_HOME"
git clone https://github.com/mitukx/VARE.git "$VARE_DIR"
git -C "$VARE_DIR" checkout "$VARE_REV"
cd "$VARE_DIR"
git rev-parse HEAD | tee "$VARE_DRIVE/code-revision.txt"
test "$(git rev-parse HEAD)" = "$VARE_REV"
test -z "$(git status --porcelain --untracked-files=all)"
```

## 2. Record free T4 and host runtime

The host must be Python 3.12.x, expose exactly one CUDA device named Tesla T4, and report at least 15,000,000,000 total VRAM bytes. Stop and save logs if any assertion fails.

```bash
/usr/bin/python3 --version | tee "$VARE_DRIVE/host-python.txt"
/usr/bin/python3 -m pip --version | tee "$VARE_DRIVE/host-pip-version.txt"
nvidia-smi | tee "$VARE_DRIVE/nvidia-smi-before.txt"
/usr/bin/python3 - <<'PY' | tee "$VARE_DRIVE/host-runtime.txt"
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

## 3. Bootstrap pip into an isolated venv

Use the host's official pip `--python` support (pip ≥22.3) to install pip inside a venv created without `ensurepip`. This is the only bootstrap method authorized for v7. It must not use `--system-site-packages`, modify the host Python, use `get-pip.py`, or run `apt`.

```bash
/usr/bin/python3 -m venv --without-pip "$VENV_DIR"
/usr/bin/python3 -m pip --version | tee "$VARE_DRIVE/host-pip-bootstrap.txt"
/usr/bin/python3 - <<'PY' | tee "$VARE_DRIVE/host-pip-version.txt"
import pip
from packaging.version import Version
print(pip.__version__)
assert Version(pip.__version__) >= Version('22.3')
PY
/usr/bin/python3 -m pip --python "$VENV_DIR" install pip \
  2>&1 | tee "$VARE_DRIVE/target-pip-bootstrap.log"
"$VENV_DIR/bin/python" -m pip --version | tee "$VARE_DRIVE/target-pip-version.txt"
```

The measured free-T4 smoke used host pip 24.1.2 and installed target pip 26.2.1. v7 requires target pip exactly 26.2.1; otherwise stop and save the failure. Official behavior: Python `venv --without-pip` skips ensurepip; pip's `--python` option manages another interpreter/venv and is supported from pip 22.3.

## 4. Install the exact v6 runtime pins inside v7 venv

Disable user-site imports. Install the exact CUDA 12.8 PyTorch wheel and then the v7 requirements file, which is byte-identical to v6. Do not inherit or modify Colab's global packages.

```bash
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export VARE_HOST_PIP_VERSION=$(/usr/bin/python3 -c 'import pip; print(pip.__version__)')
"$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128' \
  2>&1 | tee "$VARE_DRIVE/venv-torch-install.log"
awk '!/^torch==/' requirements/math500-study-gpu-gate-v7.txt > "$VARE_DRIVE/gate-requirements-no-torch.txt"
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple -r "$VARE_DRIVE/gate-requirements-no-torch.txt" \
  2>&1 | tee "$VARE_DRIVE/venv-dependencies-install.log"
"$VENV_DIR/bin/python" -m pip check 2>&1 | tee "$VARE_DRIVE/venv-pip-check.log"
"$VENV_DIR/bin/python" -m pip freeze | tee "$VARE_DRIVE/venv-pip-freeze.txt"
```

Stop if an installation fails, `pip check` fails, or any pin differs. In particular, do not resolve package conflicts by changing versions. Run the venv preflight from the clean clone:

```bash
cd "$VARE_DIR"
"$VENV_DIR/bin/python" - <<'PY' | tee "$VARE_DRIVE/venv-preflight.json"
import importlib.metadata as md, json, platform, site, sys, torch
from scripts.gpu_gate_venv_v7 import current_environment_provenance
from scripts.gpu_gate_vram_v5 import maximum_reserved_vram_bytes
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
assert provenance['pip_bootstrap']['target_pip_version'] == '26.2.1'
assert torch.version.cuda == '12.8' and torch.cuda.is_available() and torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert 'T4' in props.name and props.total_memory >= 15_000_000_000
cap = maximum_reserved_vram_bytes(props.total_memory, .90)
assert cap == min(14 * 1024**3, int(.90 * props.total_memory))
rewards = accuracy_reward(
 completions=[[{'role':'assistant','content':r'\boxed{2}'}], [{'role':'assistant','content':r'\boxed{3}'}]],
 solution=['2','2'])
assert len(rewards)==2 and bool(rewards[0]) is True and bool(rewards[1]) is False
assert exact_match(r'\boxed{2}', '2') and not exact_match(r'\boxed{3}', '2')
assert has_valid_final_box(r'\boxed{2}') and not has_valid_final_box('no final answer')
print(json.dumps({'status':'pass','python':platform.python_version(),'environment':provenance,
 'packages':{n:md.version(n) for n in expected},'torch_cuda':torch.version.cuda,
 'gpu':props.name,'total_vram_bytes':props.total_memory,'reserved_vram_cap_bytes':cap,
 'cpu_reward_fixture':rewards},indent=2,default=str))
PY
```

This verifies `sys.prefix != sys.base_prefix`, no global or user `site-packages`, all imported third-party modules (including pip) under the venv, exact versions, clean venv `pip check`, CUDA, T4, the pinned reward fixture, and independent checker fixtures. OmegaConf/IPython from the host must not appear in imported module paths.

## 5. Verify hashes and fixed TRAIN cohort

Download only the pinned model files and TRAIN parquet. Do not download or load MATH-500. The runner independently verifies every model hash, data hash, prompt-ID fingerprint, clean Git state, execution manifest, runtime, and cumulative budget before generation.

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

Expected fixed cohort fingerprint: `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`. The seven model-file hashes and TRAIN parquet hash are in `protocols/qwen25_deepmath_grpo_math500_v7.lock.json`. The v7 runner checks these itself; any mismatch is a stop condition.

Run the TRAIN-only CPU fingerprint validator and save its output:

```bash
"$VENV_DIR/bin/python" - "$VARE_DRIVE/data-ready/deepmath-train.parquet" <<'PY' | tee "$VARE_DRIVE/cpu-train-preflight.json"
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from scripts.prompt_id_hash_v3 import verify_prompt_id_sha256
from scripts.validate_math500_study_v1 import EXPECTED_DEEPMATH_SHA256, deduplicate_train_rows, select_splits, sha256_file
protocol=json.load(open('protocols/qwen25_deepmath_grpo_math500_v7.lock.json'))
data=Path(sys.argv[1])
assert sha256_file(data)==EXPECTED_DEEPMATH_SHA256
rows=pq.read_table(data,columns=['prompt','solution']).to_pylist()
eligible,stats=deduplicate_train_rows(rows)
ids=select_splits([],eligible)['base_gate_ids']
fingerprint=verify_prompt_id_sha256(ids,protocol['training_data']['sample']['base_gate_id_list_sha256'],expected_count=32)
print(json.dumps({'status':'pass','train_sha256':EXPECTED_DEEPMATH_SHA256,'prompt_ids_sha256':fingerprint,'deduplication':stats},indent=2))
PY
```

## 6. Freeze preflight and execute once

Before generation, retain `git rev-parse HEAD`, clean status, manifest, protocol SHA-256, venv preflight, package freeze, `pip check`, model/data hashes, prompt fingerprint, CUDA/GPU/runtime capture, and Drive destination. If any item fails, save logs and stop. The source revision and protocol must not be edited during the run.

Only after all checks pass, invoke this one inference-only gate. It writes durable per-group journals/checkpoints and a final result under Drive.

```bash
set -o pipefail
cd "$VARE_DIR"
"$VENV_DIR/bin/python" -m scripts.run_qwen25_deepmath_base_gate_v7 \
  --model-dir "$VARE_DRIVE/model" \
  --data-dir "$VARE_DRIVE/data-ready" \
  --output "$VARE_DRIVE/gpu-base-gate-v7.json" \
  --max-wall-seconds 7200 \
  --max-generated-tokens 131072 \
  2>&1 | tee -a "$VARE_DRIVE/console.log"
```

The runner persists each complete prompt group and cumulative accounting. If interrupted or OOM occurs, preserve Drive files and stop to report the failure; do not reset budgets or launch an altered run. Resume is permitted only when the runner's exact identity checks match, including code, protocol, model/data/cohort, runtime, GPU, venv provenance, and original cumulative limits. Do not rerun after a completed PASS or FAIL.

## 7. Stop and report

Report PASS/FAIL/environment NO-GO against the frozen thresholds. Include all 32 reward groups, independent success, final-box validity, training-reward disagreement, completion lengths, EOS/truncation, actual token IDs, cumulative wall time/throughput, and peak allocated/reserved VRAM. Retain the JSON result, durable state/journal, console, environment logs, package freeze, GPU capture, and hashes in `MyDrive/VARE/deepmath-gate-v7`.

Regardless of outcome, stop after this single gate. Do not run an optimizer step, SFT, GRPO, extra seed/prompts, MATH-500, or another GPU experiment. A passing inference gate establishes only a base-policy reward-signal observation, not training feasibility or capability improvement.
