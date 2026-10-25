# MLX #4748 Qwen3 Development pilot evidence

This directory preserves the v4 task qualification and the single v6 Qwen3 model attempt. Task and model protocols are Development-only. The frozen task definition and checkpoint are in `task.json`, `TASK.md`, and the versioned protocol files. External model pretraining exposure is unknown.

## Contents

- `qualification/`: qualification protocols v1–v4, all pre-run supersession records, the executed v4 runner, original baseline/mutant/fix raw logs, resource samples, manifests, grader sources, and the final independent reconstruction audit.
- `model/`: model protocols v1–v6, locks, setup-failure dispositions, runners v1–v6, the task package, environment freeze, and complete model-run-v6 output/event/resource trace.
- `audits/`: original qualification/model audit reports and the preserved failed audit attempt made directly from the relocated package path. That failed attempt had one path-layout check fail; all result, source, resource, and manifest checks passed. The frozen v2 auditor passed when the bundle was staged under its recorded work-tree path.
- `evidence-manifest.json`: byte sizes and SHA-256 for every file in this package except the manifest itself.

The 12,308,480-byte `qualification/runs/run-v4/baseline-source.tar` is intentionally not committed because VARE's tracked-blob policy rejects new blobs at or above 1 MiB. The original run manifest records its digest `02963739a6ad52fd0d79da69f4df580e73915635ef95a2e99520d3e62986826b`. It is reproducible byte-for-byte from baseline commit `c6abf0dbb4fc141fad739b32ae486c06728e49e5` using `git archive`.

## Verify the committed bytes

From this directory's parent:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("results/mlx-pr4748-qwen3-development-pilot-v1")
manifest = json.loads((root / "evidence-manifest.json").read_text())
bad = []
for row in manifest["files"]:
    path = root / row["path"]
    data = path.read_bytes() if path.is_file() else b""
    if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        bad.append(row["path"])
if bad:
    raise SystemExit("evidence hash mismatch: " + ", ".join(bad))
print(f"verified {len(manifest['files'])} evidence files")
PY
```

## Reconstruct the qualification audit

This reproduces the frozen v2 independent audit of the retained raw run; it does not repeat builds or change any frozen record. Run from a clean VARE checkout on macOS arm64. The upstream repository must contain the pinned baseline and fix commits.

```bash
tmp="$(mktemp -d)"
mkdir -p "$tmp/work" "$tmp/source"
cp -R results/mlx-pr4748-qwen3-development-pilot-v1/qualification \
  "$tmp/work/mlx-core-pr4748-development-v1"
git clone https://github.com/ml-explore/mlx.git "$tmp/source/mlx.git"
git -C "$tmp/source/mlx.git" fetch origin c6abf0dbb4fc141fad739b32ae486c06728e49e5
git -C "$tmp/source/mlx.git" fetch origin ae765377fbb9f0c238e2f4f6981def1fc01a75bb
git -C "$tmp/source/mlx.git" archive --format=tar c6abf0dbb4fc141fad739b32ae486c06728e49e5 \
  > "$tmp/work/mlx-core-pr4748-development-v1/runs/run-v4/baseline-source.tar"
python3 - "$tmp" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / "work/mlx-core-pr4748-development-v1/runs/run-v4-independent-audit-v2.json"
if p.exists(): p.unlink()
PY
(cd "$tmp" && python3 work/mlx-core-pr4748-development-v1/independent_audit_v4.py)
```

Expected status is `PASS_TASK_CERTIFICATION_PILOT`, with all checks true and source tar SHA-256 `02963739a6ad52fd0d79da69f4df580e73915635ef95a2e99520d3e62986826b`. The path-sensitive auditor intentionally verifies that the locked protocol is staged at its recorded `work/mlx-core-pr4748-development-v1/protocol-v4.json` location.

The original model-run auditor has already been rerun against the packaged files and returns `AUDIT_PASS_NO_SOURCE_EDIT`. To independently repeat it, use a temporary copy of `model/` and remove only the auditor's non-overwriting output path `runs/model-run-v6-independent-audit.json` in that temporary copy before invoking `audit_model_run_v1.py`. Do not run model inference; the recorded attempt is consumed.

## Rerun qualification (optional, consumes CPU only)

The exact invocation fields and all dependency commits/tree hashes are in `qualification/protocol-v4.json`. The runner's CLI requires `--protocol`, `--protocol-sha256`, `--source-repo`, `--dependencies`, `--build-python`, `--output`, and `--source-worktree`. Dependencies are pre-cached for the original run and the build/oracle/test subprocesses deny network. Re-running creates a new timestamped run and must use new, nonexistent output and work-tree paths; never write over `runs/run-v4`.

The completed baseline/mutant/fix measurements were 2/6, 2/6, and 6/6 total oracle cases respectively: both controls passed the two direct cases and failed all four vmapped cases; the exact fix passed all six. The exact fix alone passed the upstream regression test. The model attempt was one Development sample, produced no edit, and was not graded under the frozen no-edit rule.
