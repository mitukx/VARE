"""Exercise the v10 candidate boundary with hostile import-time behavior."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRADE = ROOT / "scripts/grade_transformers_fresh_dev_v5.py"
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)
TASK = "transformers-49481-bloom-static-cache-v1"
TARGET = "src/transformers/models/bloom/modeling_bloom.py"
ATTACK = r'''
def _vare_boundary_probe():
    import os
    from pathlib import Path
    import socket
    assert not Path("/grader").exists()
    assert not Path("/subject/test_public.py").exists()
    assert not Path("/subject/evaluator").exists()
    assert not Path("/subject/vare_fresh_dev_adapter.py").exists()
    assert not Path("/subject/.git").exists()
    assert not any(token in key.lower() for key in os.environ for token in ("secret", "token", "credential", "api_key"))
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=1):
            raise AssertionError("outbound network was available")
    except OSError:
        pass
    try:
        Path("/subject/forged-grade.json").write_text('{"passed": true}')
    except OSError:
        pass
    else:
        raise AssertionError("candidate subject was writable")
    print('{"passed": true, "value": "forged"}')
_vare_boundary_probe()
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(workspace: Path, output: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(GRADE), "--task-id", TASK, "--workspace", str(workspace),
         "--image", IMAGE, "--output", str(output)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "DOCKER_CONTEXT": "colima"},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/transformers-fresh-dev-cohort-v1/run-13/redteam.json",
    )
    args = parser.parse_args()
    source_workspace = Path.home() / ".cache/vare-new-task-qual/bloom/workspace-fix-v6"
    report_path = Path(os.path.abspath(os.fspath(args.output.expanduser())))
    if report_path.exists():
        raise SystemExit("refusing to overwrite red-team evidence")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vare-v10-redteam-") as temp:
        root = Path(temp)
        hostile = root / "hostile-source"
        shutil.copytree(source_workspace, hostile)
        target = hostile / TARGET
        target.write_text(target.read_text(encoding="utf-8") + ATTACK, encoding="utf-8")
        output = root / "hostile-grade.json"
        returncode, transcript = invoke(hostile, output)
        result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        results.append({
            "attack": "candidate_import_probe_and_forged_stdout",
            "returncode": returncode,
            "grader_passed": result.get("passed") if result else None,
            "failure": result.get("failure") if result else None,
            "boundary_assertions": [
                "no /grader mount", "no public test or evaluator in /subject",
                "no legacy adapter or git history", "no secret-like environment variable names",
                "outbound network blocked", "candidate workspace read-only",
                "candidate stdout cannot forge host result",
            ],
            "candidate_target_sha256": sha256(target),
            "grader_observation": result.get("verified") if result else None,
            "transcript": transcript[-1500:],
        })

        cases = (
            ("sitecustomize.py", lambda w: (w / "sitecustomize.py").write_text("raise RuntimeError('loaded')\n")),
            ("modified_imported_cache_utils", lambda w: (w / "src/transformers/cache_utils.py").write_text("# tampered\n")),
            ("malicious_symlink", lambda w: (w / "evil.py").symlink_to(w / TARGET)),
            ("forged_grade_file", lambda w: (w / "grade.json").write_text('{"passed":true}\n')),
        )
        for index, (name, mutate) in enumerate(cases):
            workspace = root / f"rejected-{index}"
            shutil.copytree(source_workspace, workspace)
            mutate(workspace)
            out = root / f"rejected-{index}.json"
            returncode, transcript = invoke(workspace, out)
            results.append({
                "attack": name,
                "rejected_before_candidate_execution": returncode != 0 and not out.exists(),
                "returncode": returncode,
                "transcript": transcript[-1200:],
            })

    report = {
        "protocol_id": "vare-transformers-fresh-development-cohort-v10",
        "protocol_sha256": "12eea4272bb6c1e40a8dab1ebf8efee176ad83fffc5e151f5e77cdc14b6344e0",
        "grader_sha256": sha256(GRADE),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "results": results,
        "all_passed": bool(results)
        and results[0]["grader_passed"] is True
        and all(row.get("rejected_before_candidate_execution") is True for row in results[1:]),
    }
    if report_path.exists() or report_path.is_symlink():
        raise SystemExit("refusing to overwrite red-team evidence")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "result_count": len(results)}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
