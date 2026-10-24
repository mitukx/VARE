#!/usr/bin/env python3
"""Run the preregistered #1912 installed-source/bytecode counterfactual."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "benchmarks/verifier_integrity_pilot_v1"
PROTOCOL = ROOT / "protocols/verifier_integrity_pilot_v1/protocol-v2.json"
sys.path.insert(0, str(ROOT / "src"))

from vare import mlxlm_1912_repo_task as task  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha(path.read_bytes())


def run(command: list[str], *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DOCKER_CONTEXT"] = "colima"
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def locked_protocol(expected: str) -> dict[str, Any]:
    raw = PROTOCOL.read_bytes()
    if sha(raw) != expected:
        raise ValueError("operator-supplied frozen protocol digest mismatch")
    protocol = json.loads(raw)
    for relative, digest in protocol["implementation_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise ValueError(f"frozen implementation changed: {relative}")
    for name, digest in protocol["candidate_sha256"].items():
        if sha_file(PILOT / "candidates" / name) != digest:
            raise ValueError(f"frozen candidate changed: {name}")
    return protocol


def build_variant(
    name: str, context: Path, dockerfile: Path, tag: str,
    registry_tag: str, output: Path
) -> dict[str, Any]:
    started = time.monotonic()
    command = [
        "docker", "--context", "colima", "build", "--pull=false", "--network=none",
        "--platform=linux/arm64", "--tag", tag, "--file", str(dockerfile),
        str(context),
    ]
    built = run(command)
    (output / f"build-{name}.stdout.log").write_text(built.stdout, encoding="utf-8")
    (output / f"build-{name}.stderr.log").write_text(built.stderr, encoding="utf-8")
    build_seconds = time.monotonic() - started
    run(["docker", "--context", "colima", "tag", tag, registry_tag])
    push_started = time.monotonic()
    pushed = run(["docker", "--context", "colima", "push", registry_tag])
    (output / f"push-{name}.stdout.log").write_text(pushed.stdout, encoding="utf-8")
    (output / f"push-{name}.stderr.log").write_text(pushed.stderr, encoding="utf-8")
    inspect = json.loads(
        run(["docker", "--context", "colima", "image", "inspect", registry_tag]).stdout
    )[0]
    digest = next(
        (item.split("@", 1)[1] for item in inspect.get("RepoDigests", []) if "@" in item),
        None,
    )
    if not digest:
        raise RuntimeError(f"registry did not return an immutable digest for {name}")
    return {
        "tag": registry_tag,
        "image": registry_tag.rsplit(":", 1)[0] + "@" + digest,
        "digest": digest,
        "image_id": inspect["Id"],
        "size_bytes": inspect["Size"],
        "build_seconds": build_seconds,
        "push_seconds": time.monotonic() - push_started,
        "platform": inspect["Os"] + "/" + inspect["Architecture"],
    }


def image_probe(image: str) -> dict[str, Any]:
    code = r'''
import base64,csv,hashlib,importlib.metadata as md,json,marshal,pathlib,site,zipfile
from types import CodeType
site_root=pathlib.Path(site.getsitepackages()[0]); package=site_root/'mlx_lm'
target=package/'sample_utils.py'; cache=package/'__pycache__/sample_utils.cpython-312.pyc'
fix='3aac92d804b6b91c82a4e2844697a8723de26f6af57e10013a482b967e739cfa'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def walk(c):
    yield c
    for x in c.co_consts:
        if isinstance(x,CodeType): yield from walk(x)
def function(c): return next((x for x in walk(c) if x.co_name=='apply_top_p'),None)
installed=function(compile(target.read_text(),str(target),'exec'))
compiled=None
if cache.is_file(): compiled=function(marshal.loads(cache.read_bytes()[16:]))
metadata=md.distribution('mlx-lm'); record=pathlib.Path(metadata._path)/'RECORD'
rows=list(csv.reader(record.open(newline='',encoding='utf-8')))
source_row=[r for r in rows if r and r[0]=='mlx_lm/sample_utils.py']
pyc_rows=[r for r in rows if r and r[0].endswith('.pyc') and r[0].startswith('mlx_lm/')]
source_files=[p for p in site_root.rglob('*.py') if p.is_file()]
source_exact=[str(p) for p in source_files if h(p)==fix]
pycs=list(site_root.rglob('*.pyc'))
named_code=[]; exact_code=[]
for p in pycs:
    try: candidate=function(marshal.loads(p.read_bytes()[16:]))
    except Exception: continue
    if candidate is not None:
        named_code.append(str(p))
        if installed is not None and candidate.co_code==installed.co_code and candidate.co_names==installed.co_names and candidate.co_consts==installed.co_consts:
            exact_code.append(str(p))
wheels=[]; wheel_reference=[]
for root in [pathlib.Path('/root/.cache'),pathlib.Path('/tmp'),site_root,pathlib.Path('/opt')]:
    if not root.exists(): continue
    for p in root.rglob('*.whl'):
        wheels.append(str(p))
        try:
            with zipfile.ZipFile(p) as z:
                n='mlx_lm/sample_utils.py'
                if n in z.namelist() and hashlib.sha256(z.read(n)).hexdigest()==fix: wheel_reference.append(str(p))
        except Exception: pass
req=[]
for item in metadata.metadata.get_all('Requires-Dist') or []: req.append(item)
direct=pathlib.Path(metadata._path)/'direct_url.json'
result={'target_sha256':h(target),'target_path':str(target),'pyc_present':cache.is_file(),'pyc_sha256':h(cache) if cache.is_file() else None,'pyc_function_matches_installed_source':bool(compiled and installed and compiled.co_code==installed.co_code and compiled.co_names==installed.co_names and compiled.co_consts==installed.co_consts),'target_source_rows_in_RECORD':source_row,'target_pyc_rows_in_RECORD':pyc_rows,'source_exact_fix_paths':source_exact,'pyc_apply_top_p_functions':named_code,'pyc_exact_function_paths':exact_code,'wheel_count':len(wheels),'wheel_exact_fix_paths':wheel_reference,'direct_url_json':direct.read_text() if direct.exists() else None,'requires_dist':req,'mlx_lm_python_files':len(list(package.rglob('*.py'))),'mlx_lm_pyc_files':len(list(package.rglob('*.pyc'))),'all_site_packages_pyc_files':len(pycs)}
print(json.dumps(result,sort_keys=True))
'''
    completed = run(
        [
            "docker", "--context", "colima", "run", "--rm", "--pull=never",
            "--network=none", "--read-only", "--memory=512m", "--cpus=0.5",
            "--pids-limit=64", "--entrypoint", "python", image, "-c", code,
        ],
        timeout=120,
    )
    return json.loads(completed.stdout)


def evaluate(image: str, candidate: str, content: bytes, out: Path) -> dict[str, Any]:
    workspace = out / "workspaces" / image.split("@")[-1].split(":")[-1] / candidate
    target = workspace / "mlx_lm/sample_utils.py"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(content)
    original = task.verify_protocol_v4

    def verify_with_image() -> dict[str, Any]:
        frozen = json.loads(json.dumps(original()))
        frozen["runtime"]["image"] = image
        return frozen

    task.verify_protocol_v4 = verify_with_image
    started = time.monotonic()
    try:
        result = task.grade_workspace(workspace, image=image)
    finally:
        task.verify_protocol_v4 = original
    result["candidate_id"] = candidate
    result["image"] = image
    result["candidate_elapsed_seconds"] = time.monotonic() - started
    result["candidate_source_sha256"] = sha(content)
    result["evaluator_case_count"] = len(result["cases"])
    (out / "outcomes").mkdir(exist_ok=True)
    file = out / "outcomes" / f"{workspace.parent.name}-{candidate}.json"
    file.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["DOCKER_CONTEXT"] = "colima"
    protocol = locked_protocol(args.protocol_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    source_repo = args.source_repo.resolve(strict=True)
    for rev, path, expected in protocol["upstream_sources"]:
        data = subprocess.run(
            ["git", "-C", str(source_repo), "show", f"{rev}:{path}"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        if sha(data) != expected:
            raise ValueError(f"upstream source differs from frozen bytes: {rev}:{path}")

    images = {"original": protocol["images"]["original_image"]}
    for name, build in protocol["images"]["derived_images"].items():
        images[name] = build_variant(
            name,
            PILOT / build["context"],
            PILOT / build["dockerfile"],
            build["local_tag"],
            build["registry_tag"],
            output,
        )
    audits = {name: image_probe(item if isinstance(item, str) else item["image"])
              for name, item in images.items()}
    (output / "image-audits.json").write_text(
        json.dumps(audits, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    baseline = (PILOT / "candidates/baseline.py").read_bytes()
    candidate_sources = {
        "baseline": baseline,
        "known-fix": (PILOT / "candidates/known_fix.py").read_bytes(),
        "mutant-keep-all": (PILOT / "candidates/mutant_keep_all.py").read_bytes(),
        "mutant-drop-all": (PILOT / "candidates/mutant_drop_all.py").read_bytes(),
        "delegate-installed-source": (PILOT / "candidates/delegate_installed.py").read_bytes(),
        "delegate-compiled-bytecode": (PILOT / "candidates/delegate_bytecode.py").read_bytes(),
    }
    outcomes = []
    for condition, image_record in images.items():
        image = image_record if isinstance(image_record, str) else image_record["image"]
        for candidate, content in candidate_sources.items():
            outcomes.append(evaluate(image, candidate, content, output))
    rows = [
        {
            "condition": condition,
            "candidate": row["candidate_id"],
            "image": row["image"],
            "passed": row["passed"],
            "case_passed": sum(case["passed"] for case in row["cases"]),
            "case_count": row["evaluator_case_count"],
            "candidate_elapsed_seconds": row["candidate_elapsed_seconds"],
        }
        for condition, record in images.items()
        for row in outcomes
        if row["image"] == (record if isinstance(record, str) else record["image"])
    ]
    summary = {
        "protocol_sha256": args.protocol_sha256,
        "protocol_id": protocol["protocol_id"],
        "preregistration_head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference": "NOT_RUN",
        "optimizer_steps": 0,
        "images": images,
        "image_audits": audits,
        "outcomes": rows,
        "primary_decision": protocol["decision_rule"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
