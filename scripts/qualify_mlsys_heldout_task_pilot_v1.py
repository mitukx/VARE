#!/usr/bin/env python3
"""Run the frozen source-only calibration for reserved MLX-LM tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402
from vare.mlsys_heldout_task_pilot_v1 import (  # noqa: E402
    LOCK_PATH,
    PROTOCOL_PATH,
    QualificationError,
    load_frozen_protocol,
    materialize_candidate,
    mutate_target,
    package_delta,
    python_package_blobs,
    sha256,
    sha256_file,
    unified_target_patch,
    validate_patch,
    verify_task_source,
    write_exclusive_atomic,
)


TASK_NAMES = {
    "mlxlm-1902-continuation-score-truncation": "mlxlm_1902_continuation_score",
    "mlxlm-1926-falcon-cache-gradient": "mlxlm_1926_falcon_cache_gradient",
    "mlxlm-1938-phi3small-logits-mask": "mlxlm_1938_phi3small_logits_mask",
    "mlxlm-1979-ssm-padding-mask": "mlxlm_1979_ssm_padding_mask",
}
VARIANTS = ("pinned_baseline", "pinned_known_fix_calibration_only", "prespecified_incorrect_mutant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exclusive_json(path: Path, value: Any) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write_exclusive_atomic(path, raw)


def _append_event(stream, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, timeout=30,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    ).strip()


def _image_provenance(image: str, context: str) -> dict[str, Any]:
    inspect = subprocess.check_output(
        ["docker", "--context", context, "image", "inspect", image, "--format", "{{json .}}"],
        text=True,
        timeout=20,
    )
    image_row = json.loads(inspect)
    info = subprocess.check_output(
        ["docker", "--context", context, "info", "--format", "{{.OSType}}/{{.Architecture}} {{.ServerVersion}}"],
        text=True,
        timeout=20,
    ).strip()
    environment = subprocess.check_output(
        [
            "docker", "--context", context, "run", "--rm", "--pull=never",
            "--network=none", "--memory=256m", "--cpus=1", image, "python", "-c",
            "import importlib.metadata as m, json, mlx.core as mx, sys; "
            "names=('mlx','numpy','tqdm','pytest','lm-eval'); "
            "print(json.dumps({'python':sys.version.split()[0], 'versions':{n:(m.version(n) if any(d.metadata['Name'].lower()==n.lower() for d in m.distributions()) else None) for n in names}, 'default_device':str(mx.default_device())}))",
        ],
        text=True,
        timeout=30,
    ).strip()
    return {
        "image_reference": image,
        "image_id": image_row["Id"],
        "repo_digests": sorted(image_row.get("RepoDigests") or []),
        "repo_tags": sorted(image_row.get("RepoTags") or []),
        "image_size_bytes": image_row.get("Size"),
        "os": image_row.get("Os"),
        "architecture": image_row.get("Architecture"),
        "docker_server": info,
        "candidate_dependency_preflight": json.loads(environment),
        "package_metadata_note": "mlx-lm is supplied only as source under /subject; lm-eval absence is recorded and the #1902 adapter executes only the pinned target method AST.",
    }


_MEM = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*(B|kB|KB|KiB|MB|MiB|GB|GiB)\b")
_UNITS = {"B": 1, "kB": 1000, "KB": 1000, "KiB": 1024, "MB": 1_000_000, "MiB": 1024**2, "GB": 1_000_000_000, "GiB": 1024**3}


class DockerMemorySampler:
    """Poll Docker's per-container memory meter; samples are approximate peaks."""

    def __init__(self, context: str) -> None:
        self.context = context
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                ids = subprocess.check_output(
                    ["docker", "--context", self.context, "ps", "-q", "--filter", "name=vare-call-"],
                    text=True,
                    timeout=2,
                ).split()
                for container_id in ids:
                    output = subprocess.check_output(
                        ["docker", "--context", self.context, "stats", "--no-stream", "--format", "{{.MemUsage}} {{.CPUPerc}}", container_id],
                        text=True,
                        timeout=2,
                    ).strip()
                    usage = output.split("/")[0].strip()
                    match = _MEM.match(usage)
                    if match:
                        memory = int(float(match.group(1)) * _UNITS[match.group(2)])
                        self.samples.append({"at_utc": _now(), "container_id": container_id, "memory_bytes": memory, "raw": output})
            except (OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.15)

    def summary(self) -> dict[str, Any]:
        maxima: dict[str, int] = {}
        for sample in self.samples:
            key = sample["container_id"]
            maxima[key] = max(maxima.get(key, 0), sample["memory_bytes"])
        return {
            "method": "docker stats --no-stream sampled every 150ms while vare-call containers are running",
            "sample_count": len(self.samples),
            "sampled_container_count": len(maxima),
            "sampled_peak_bytes": max(maxima.values(), default=None),
            "per_container_peak_bytes": maxima,
            "samples": self.samples,
            "limitation": "Short-lived container peaks can occur between polling samples; the 1536 MiB container limit is authoritative containment, sampled peaks are not exact high-water marks.",
        }


def _swap_used_bytes() -> int | None:
    try:
        text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"used = ([0-9.]+)([KMG])", text)
    if not match:
        return None
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def _calls(task_id: str) -> list[FunctionCall]:
    if task_id == "mlxlm-1902-continuation-score-truncation":
        calls = []
        prefixes = {4: [1, 2, 3], 3: [2, 3], 2: [3]}
        for max_tokens, expected_prefix in prefixes.items():
            calls.append(
                FunctionCall(
                    case_id=f"max_tokens_{max_tokens}",
                    module="vare_task_adapter",
                    function="run",
                    args=[{
                        "max_tokens": max_tokens,
                        "context_tokens": [[1, 2, 3]],
                        "complete_tokens": [[1, 2, 3, 4, 5]],
                    }],
                    expected={
                        "result": [[-9.0, False]],
                        "processed_prefix": expected_prefix,
                        "scored_inputs": [[4, 5]],
                        "score_invocations": 1,
                    },
                )
            )
        return calls
    if task_id == "mlxlm-1926-falcon-cache-gradient":
        return [
            FunctionCall(
                case_id="tiny_three_layer_no_cache_and_gradient",
                module="vare_task_adapter",
                function="run",
                args=[{"seed": 0}],
                expected={
                    "cache_equivalent": True,
                    "loss_finite": True,
                    "layers": [{"finite": True, "nonzero": True}] * 3,
                },
            )
        ]
    if task_id == "mlxlm-1938-phi3small-logits-mask":
        return [
            FunctionCall(
                case_id="dummy_token_logits_masked_along_vocab_axis",
                module="vare_task_adapter",
                function="run",
                args=[{"seed": 0}],
                expected={
                    "logits_shape": [1, 3, 100352],
                    "dummy_logits_shape": [1, 3, 91],
                    "all_dummy_logits_negative_infinity": True,
                    "layer_count": 4,
                },
            )
        ]
    if task_id == "mlxlm-1979-ssm-padding-mask":
        return [
            FunctionCall(
                case_id="right_padding_matches_unpadded_outputs_and_final_state",
                module="vare_task_adapter",
                function="run",
                args=[{"seed": 0}],
                expected={
                    "valid_outputs_match_unpadded": True,
                    "final_state_matches_unpadded": True,
                    "outputs_finite": True,
                    "state_finite": True,
                },
            )
        ]
    raise QualificationError(f"no grader calls declared for {task_id}")


def _patch_adversarial_checks(contract: dict[str, Any], base: bytes, fixed: bytes) -> dict[str, Any]:
    target = contract["target_path"]
    valid = unified_target_patch(target, base, fixed)
    valid_paths = validate_patch(valid, target)
    wrong_path = (
        "diff --git a/../grader/answers.json b/../grader/answers.json\n"
        "--- a/../grader/answers.json\n+++ b/../grader/answers.json\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    denied = {}
    for name, patch in (
        ("path_traversal", wrong_path),
        ("non_allowlisted_target", unified_target_patch("mlx_lm/other.py", b"a\n", b"b\n")),
    ):
        try:
            validate_patch(patch, target)
        except QualificationError:
            denied[name] = True
        else:
            denied[name] = False
    return {
        "pinned_fix_patch_sha256": sha256(valid.encode()),
        "pinned_fix_patch_paths": valid_paths,
        "path_traversal_rejected": denied["path_traversal"],
        "non_allowlisted_target_rejected": denied["non_allowlisted_target"],
    }


def _shared_boundary_probe(image: str, root: Path, sampler: DockerMemorySampler) -> dict[str, Any]:
    subject = root / "boundary-probe"
    subject.mkdir()
    probe = subject / "vare_task_adapter.py"
    probe.write_text(
        "from pathlib import Path\n"
        "import socket, subprocess, sys\n"
        "def run():\n"
        "    try:\n"
        "        s=socket.create_connection(('1.1.1.1',53),timeout=0.25); s.close(); net=True\n"
        "    except OSError:\n"
        "        net=False\n"
        "    try:\n"
        "        Path('/subject/probe-write').write_text('x'); write=True\n"
        "    except OSError:\n"
        "        write=False\n"
        "    print('VARE_RESULT forged-frame')\n"
        "    return {'grader_absent':not Path('/grader').exists(), 'git_absent':not Path('/subject/.git').exists(), 'tests_absent':not Path('/subject/tests').exists(), 'network_blocked':not net, 'subject_readonly':not write}\n",
        encoding="utf-8",
    )
    probe.chmod(0o444)
    expected = {
        "grader_absent": True,
        "git_absent": True,
        "tests_absent": True,
        "network_blocked": True,
        "subject_readonly": True,
    }
    call = FunctionCall("separated_no_network_boundary", "vare_task_adapter", "run", [], expected)
    report = grade_calls(
        subject,
        [call],
        task_kind="correctness",
        image=image,
        timeout_s=3,
        memory_mb=1536,
        discard_candidate_stdout=True,
    )
    return {"passed": report.passed, "cases": [case.to_dict() for case in report.cases]}


def run(
    source_repo: Path,
    exposure_audit_path: Path,
    output: Path,
    protocol_path: Path = PROTOCOL_PATH,
    lock_path: Path = LOCK_PATH,
    resume_existing: bool = False,
) -> dict[str, Any]:
    protocol, lock = load_frozen_protocol(protocol_path, lock_path)
    exposure_audit = json.loads(exposure_audit_path.read_text(encoding="utf-8"))
    selected_task_ids = {task["task_id"] for task in protocol["task_contracts"]}
    if not selected_task_ids.issubset(set(exposure_audit.get("task_ids", []))):
        raise QualificationError("exposure audit does not cover every frozen task")
    all_matches = exposure_audit.get("candidate_rollout_or_training_matches", [])
    selected_matches = [
        row for row in all_matches
        if not row.get("task_ids") or selected_task_ids.intersection(row["task_ids"])
    ]
    if selected_matches:
        raise QualificationError("retained VARE artifact indicates prior candidate exposure; retire task before grading")
    prior_failure = None
    if resume_existing:
        if output.is_symlink() or not output.is_dir():
            raise QualificationError("resume requires an existing real output directory")
        if any((output / name).exists() for name in ("qualification.json", "manifest.json", "manifest.sha256")):
            raise QualificationError("a finalized output cannot be resumed")
        failure_path = output / "failure.json"
        if not failure_path.is_file():
            raise QualificationError("resume requires a retained prior failure record")
        prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
        expected_failure = (
            "variant changed paths violate allowlist: "
            "mlxlm-1938-phi3small-logits-mask/prespecified_incorrect_mutant"
        )
        if (
            prior_failure.get("protocol_sha256") != lock["protocol_sha256"]
            or prior_failure.get("error_type") != "QualificationError"
            or prior_failure.get("error") != expected_failure
        ):
            raise QualificationError("retained failure is not the declared resumable v2 staging interruption")
    else:
        if output.exists() or output.is_symlink():
            raise QualificationError("output directory already exists; results are append-only")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True)
    events_path = output / "events.jsonl"
    events = events_path.open("a" if resume_existing else "x", encoding="utf-8")
    started = time.monotonic()
    if resume_existing:
        host_started = json.loads((output / "host-preflight.json").read_text(encoding="utf-8"))
        if host_started.get("protocol_host_swap_growth_abort_bytes") != protocol["runtime"]["host_swap_growth_abort_bytes"]:
            raise QualificationError("retained host preflight does not match the frozen protocol")
        if sha256_file(output / "exposure-audit.json") != sha256_file(exposure_audit_path):
            raise QualificationError("retained exposure audit differs from the frozen input")
    else:
        host_started_swap = _swap_used_bytes()
        host_started = {
            "at_utc": _now(),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "machine": subprocess.check_output(["uname", "-m"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) if sys.platform == "darwin" else None,
            "swap_used_bytes": host_started_swap,
            "protocol_host_swap_growth_abort_bytes": protocol["runtime"]["host_swap_growth_abort_bytes"],
        }
        _exclusive_json(output / "host-preflight.json", host_started)
    host_started_swap = host_started.get("swap_used_bytes")
    os.environ["DOCKER_CONTEXT"] = protocol["runtime"]["docker_context"]
    try:
        if not resume_existing:
            write_exclusive_atomic(output / "exposure-audit.json", exposure_audit_path.read_bytes())
        image_info = _image_provenance(protocol["runtime"]["image"], protocol["runtime"]["docker_context"])
        expected_image_id = "sha256:" + protocol["runtime"]["image"].split("@sha256:", 1)[1]
        if image_info["image_id"] != expected_image_id:
            raise QualificationError("container image digest does not match frozen protocol")
        if f"{image_info['os']}/{image_info['architecture']}" != protocol["runtime"]["architecture"]:
            raise QualificationError("container architecture does not match frozen protocol")
        runtime_path = output / "runtime-image.json"
        if resume_existing:
            if json.loads(runtime_path.read_text(encoding="utf-8")) != image_info:
                raise QualificationError("runtime image provenance changed during resume")
        else:
            _exclusive_json(runtime_path, image_info)
        origin = _git(source_repo, "remote", "get-url", "origin")
        if origin.rstrip("/").removesuffix(".git") != "https://github.com/ml-explore/mlx-lm":
            raise QualificationError("source repository origin does not match pinned upstream")
        source_repo_state = {
            "origin": origin,
            "is_bare": _git(source_repo, "rev-parse", "--is-bare-repository") == "true",
            "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
        }
        previous_records: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
        existing_event_hashes: dict[str, str] = {}
        if resume_existing:
            event_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            for row in event_rows:
                if row.get("event") == "variant_complete":
                    existing_event_hashes[row["record"]] = row["record_sha256"]
            allowed_names = {
                f"run-{(task_index - 1) * len(VARIANTS) + variant_index:02d}-"
                f"{TASK_NAMES[task['task_id']]}-{variant}.json"
                for task_index, task in enumerate(protocol["task_contracts"], start=1)
                for variant_index, variant in enumerate(VARIANTS, start=1)
            }
            for path in sorted(output.glob("run-*.json")):
                if path.name not in allowed_names:
                    raise QualificationError("resume found an unexpected candidate record")
                record = json.loads(path.read_text(encoding="utf-8"))
                record_hash = sha256_file(path)
                if existing_event_hashes.get(path.name) != record_hash:
                    raise QualificationError("resume candidate record has no matching durable event hash")
                key = (record.get("task_id"), record.get("variant"))
                if key in previous_records or record.get("decision") not in {"PASS", "FAIL"}:
                    raise QualificationError("resume candidate record identity or decision is invalid")
                if record.get("candidate_visible_grader") or record.get("candidate_visible_git_history"):
                    raise QualificationError("resume candidate record violates source-only visibility")
                previous_records[key] = (record, record_hash)
            if not previous_records:
                raise QualificationError("resume found no completed variants to preserve")
            _append_event(events, {
                "event": "resumption_started",
                "at_utc": _now(),
                "preserved_variant_records": sorted(path.name for path in output.glob("run-*.json")),
                "prior_failure_sha256": sha256_file(output / "failure.json"),
            })
        sampler = DockerMemorySampler(protocol["runtime"]["docker_context"])
        sampler.start()
        records = []
        candidate_invocations = sum(
            len(record.get("case_results", [])) for record, _ in previous_records.values()
        )

        def reserve_invocations(count: int) -> None:
            nonlocal candidate_invocations
            proposed = candidate_invocations + count
            if proposed > protocol["runtime"]["maximum_new_candidate_invocations"]:
                raise QualificationError("fixed candidate invocation budget would be exceeded")
            candidate_invocations = proposed

        event_number = max((record.get("event_id", 0) for record, _ in previous_records.values()), default=0)
        with tempfile.TemporaryDirectory(prefix="vare-qualification-subjects-") as scratch_name:
            scratch = Path(scratch_name)
            if not resume_existing:
                _append_event(events, {"event": "source_and_image_preflight_passed", "at_utc": _now(), "source_repo": source_repo_state})
            for task in protocol["task_contracts"]:
                task_id = task["task_id"]
                source = verify_task_source(source_repo, task)
                base_tree = python_package_blobs(source_repo, task["baseline_commit"])
                adapter_path = ROOT / "benchmarks/mlsys_repo_eval_v3" / TASK_NAMES[task_id] / "grader_adapter.py"
                adapter_hash = sha256_file(adapter_path)
                baseline_target = source["baseline"]["target_bytes"]
                fixed_target = source["fixed"]["target_bytes"]
                mutant_target = mutate_target(task_id, fixed_target)
                adversarial = _patch_adversarial_checks(task, baseline_target, fixed_target)
                task_record: dict[str, Any] = {
                    "task_id": task_id,
                    "split": task["split"],
                    "source": {
                        "repository": task["repository"],
                        "baseline_commit": task["baseline_commit"],
                        "fixed_commit": task["fixed_commit"],
                        "baseline_target_sha256": sha256(baseline_target),
                        "fixed_target_sha256": sha256(fixed_target),
                        "grader_path": task["grader_path"],
                        "grader_revision": task["grader_revision"],
                        "grader_sha256": source["grader"]["sha256"],
                        "license": source.get("license"),
                        "candidate_prompt_path": task["candidate_prompt_path"],
                        "candidate_prompt_sha256": task["candidate_prompt_sha256"],
                    },
                    "candidate_adapter_sha256": adapter_hash,
                    "candidate_source_python_file_count": len(base_tree),
                    "candidate_source_tree_sha256": sha256(
                        json.dumps({path: sha256(content) for path, content in sorted(base_tree.items())}, sort_keys=True, separators=(",", ":")).encode()
                    ),
                    "patch_admission_controls": adversarial,
                    "variants": [],
                }
                targets = {
                    "pinned_baseline": baseline_target,
                    "pinned_known_fix_calibration_only": fixed_target,
                    "prespecified_incorrect_mutant": mutant_target,
                }
                for variant in VARIANTS:
                    prior = previous_records.get((task_id, variant))
                    if prior is not None:
                        record, record_hash = prior
                        if record.get("candidate_target_sha256") != sha256(targets[variant]):
                            raise QualificationError("preserved candidate target differs from the frozen variant")
                        task_record["variants"].append({
                            "variant": variant,
                            "decision": record["decision"],
                            "record": next(name for name, digest in existing_event_hashes.items() if digest == record_hash),
                            "record_sha256": record_hash,
                        })
                        continue
                    event_number += 1
                    name = f"{event_number:02d}-{TASK_NAMES[task_id]}-{variant}"
                    candidate_root = scratch / name
                    target_bytes = targets[variant]
                    patch_text = "" if variant == "pinned_baseline" else unified_target_patch(task["target_path"], baseline_target, target_bytes)
                    patch_paths = [] if not patch_text else validate_patch(patch_text, task["target_path"])
                    if patch_text:
                        patch_file = scratch / f"{name}.patch"
                        patch_file.write_text(patch_text, encoding="utf-8")
                    expected_source_files = materialize_candidate(base_tree, task["target_path"], target_bytes, adapter_path, candidate_root)
                    changes = package_delta(base_tree, expected_source_files)
                    expected_changes = (
                        []
                        if target_bytes == base_tree[task["target_path"]]
                        else [task["target_path"]]
                    )
                    if changes != expected_changes:
                        raise QualificationError(f"variant changed paths violate allowlist: {task_id}/{variant}")
                    if expected_source_files[task["target_path"]] != sha256(target_bytes):
                        raise QualificationError(f"candidate target hash drift: {task_id}/{variant}")
                    record = {
                        "event_id": event_number,
                        "started_at_utc": _now(),
                        "task_id": task_id,
                        "variant": variant,
                        "candidate_files": len(expected_source_files),
                        "candidate_file_manifest_sha256": sha256(json.dumps(expected_source_files, sort_keys=True, separators=(",", ":")).encode()),
                        "candidate_changed_paths": changes,
                        "candidate_target_sha256": expected_source_files[task["target_path"]],
                        "patch_sha256": sha256(patch_text.encode()) if patch_text else None,
                        "patch_paths": patch_paths,
                        "candidate_visible_grader": False,
                        "candidate_visible_git_history": False,
                        "candidate_visible_task_prompt": False,
                        "network": "none",
                    }
                    # The exact test-derived outputs remain in this trusted parent process.
                    calls = _calls(task_id)
                    reserve_invocations(len(calls))
                    grader = grade_calls(
                        candidate_root,
                        calls,
                        task_kind="correctness",
                        image=protocol["runtime"]["image"],
                        timeout_s=protocol["runtime"]["candidate_timeout_seconds"],
                        memory_mb=protocol["runtime"]["candidate_memory_mb"],
                        discard_candidate_stdout=True,
                    )
                    record["grader_passed"] = grader.passed
                    record["case_results"] = [case.to_dict() for case in grader.cases]
                    record["completed_at_utc"] = _now()
                    record["decision"] = "PASS" if grader.passed else "FAIL"
                    record_path = output / f"run-{event_number:02d}-{TASK_NAMES[task_id]}-{variant}.json"
                    record_hash = _exclusive_json(record_path, record)
                    task_record["variants"].append({"variant": variant, "decision": record["decision"], "record": record_path.name, "record_sha256": record_hash})
                    _append_event(events, {"event": "variant_complete", "at_utc": _now(), "task_id": task_id, "variant": variant, "record": record_path.name, "record_sha256": record_hash})
                task_decisions = {row["variant"]: row["decision"] for row in task_record["variants"]}
                task_record["certificate_gate_passed"] = (
                    task_decisions.get("pinned_baseline") == "FAIL"
                    and task_decisions.get("pinned_known_fix_calibration_only") == "PASS"
                    and task_decisions.get("prespecified_incorrect_mutant") == "FAIL"
                    and adversarial["path_traversal_rejected"]
                    and adversarial["non_allowlisted_target_rejected"]
                )
                task_record["status"] = "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" if task_record["certificate_gate_passed"] else "UNQUALIFIED"
                records.append(task_record)
                _append_event(events, {"event": "task_gate_complete", "at_utc": _now(), "task_id": task_id, "status": task_record["status"]})
            reserve_invocations(1)
            boundary = _shared_boundary_probe(protocol["runtime"]["image"], scratch, sampler)
        sampler.close()
        if not boundary["passed"]:
            raise QualificationError("shared no-network/grader-separation boundary probe failed")
        end_swap = _swap_used_bytes()
        if host_started_swap is not None and end_swap is not None:
            growth = max(0, end_swap - host_started_swap)
            if growth > protocol["runtime"]["host_swap_growth_abort_bytes"]:
                raise QualificationError("host swap growth exceeded frozen abort threshold")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        exposure = {
            "audit_status": "CLEAN_WITHIN_AUDITED_VARE_SCOPE",
            "candidate_model_prompts": 0,
            "candidate_model_rollouts": 0,
            "training_or_update": False,
            "retained_candidate_artifact_matches": len(selected_matches),
            "audit_file_sha256": sha256_file(output / "exposure-audit.json"),
            "external_pretraining_corpus": exposure_audit.get("external_pretraining_corpus", "UNKNOWN_NOT_AUDITED"),
            "scope_limit": exposure_audit.get("scope_limit"),
            "note": "No candidate coding model or learning agent received either prompt, fix, test source, or grader result. This is not a model pretraining-corpus audit.",
        }
        overall = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": lock["protocol_sha256"],
            "started_at_utc": host_started["at_utc"],
            "completed_at_utc": _now(),
            "source_repo": source_repo_state,
            "task_certificates": records,
            "candidate_invocations": candidate_invocations,
            "candidate_invocation_budget": protocol["runtime"]["maximum_new_candidate_invocations"],
            "shared_boundary_probe": boundary,
            "candidate_memory_sampling": sampler.summary(),
            "host_resource_usage": {
                "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": usage.ru_utime,
                "system_cpu_seconds": usage.ru_stime,
                "python_launcher_peak_rss_bytes": maxrss,
                "swap_start_bytes": host_started_swap,
                "swap_end_bytes": end_swap,
                "swap_growth_bytes": None if host_started_swap is None or end_swap is None else max(0, end_swap - host_started_swap),
            },
            "exposure_audit": exposure,
            "decision": (
                "TASK_GRADER_CERTIFICATE_PASS_WITHIN_AUDITED_VARE_SCOPE"
                if all(row["certificate_gate_passed"] for row in records)
                else "NO_GO"
            ),
            "recovered_prior_attempt_failure": (
                {
                    "error_type": prior_failure["error_type"],
                    "error": prior_failure["error"],
                    "failure_record_sha256": sha256_file(output / "failure.json"),
                    "completed_variants_reused_without_reexecution": sorted(
                        f"{task_id}/{variant}" for task_id, variant in previous_records
                    ),
                }
                if prior_failure is not None
                else None
            ),
            "model_training": "NOT_RUN",
            "model_rollouts": "NOT_RUN",
            "claim_boundary": protocol["claim_boundary"],
            "dependency_friction": image_info["package_metadata_note"],
            "source_provenance": {
                "repository_head": _git(ROOT, "rev-parse", "HEAD"),
                "qualification_runner_sha256": sha256_file(Path(__file__).resolve()),
                "qualification_module_sha256": sha256_file(ROOT / "src/vare/mlsys_heldout_task_pilot_v1.py"),
                "isolation_harness_sha256": sha256_file(ROOT / "src/vare/isolated_candidate.py"),
                "container_isolation_sha256": sha256_file(ROOT / "src/vare/container_isolation.py"),
                "exposure_audit_runner_sha256": sha256_file(ROOT / "scripts/audit_heldout_task_exposure_pilot_v1.py"),
            },
        }
        overall_hash = _exclusive_json(output / "qualification.json", overall)
        _append_event(events, {"event": "qualification_summary_written", "at_utc": _now(), "path": "qualification.json", "sha256": overall_hash})
        events.close()
        file_names = sorted(path.name for path in output.iterdir() if path.is_file())
        manifest = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": lock["protocol_sha256"],
            "file_count": len(file_names),
            "files": {name: sha256_file(output / name) for name in file_names},
            "manifest_semantics": "SHA-256 inventory; files are created exclusively, no replacement or overwrite. Not a digital signature.",
        }
        manifest_hash = _exclusive_json(output / "manifest.json", manifest)
        write_exclusive_atomic(output / "manifest.sha256", f"{manifest_hash}  manifest.json\n".encode())
        return overall
    except Exception as exc:
        try:
            if "sampler" in locals():
                sampler.close()
            _append_event(events, {"event": "qualification_failed_or_interrupted", "at_utc": _now(), "error_type": type(exc).__name__, "error": str(exc)})
            failure_path = output / "failure.json"
            if not failure_path.exists():
                _exclusive_json(failure_path, {"at_utc": _now(), "error_type": type(exc).__name__, "error": str(exc), "protocol_sha256": lock.get("protocol_sha256")})
        finally:
            events.close()
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True, help="bare or working Git clone of ml-explore/mlx-lm with pinned commits fetched")
    parser.add_argument("--exposure-audit", type=Path, required=True, help="completed retained-record audit output for this frozen task pair")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent result directory")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH, help="frozen pilot protocol")
    parser.add_argument("--protocol-lock", type=Path, default=LOCK_PATH, help="hash lock for the frozen pilot protocol")
    parser.add_argument("--resume-existing", action="store_true", help="resume only the declared interrupted #1938 staging run, preserving completed variants")
    args = parser.parse_args()
    result = run(
        args.source_repo.resolve(),
        args.exposure_audit.resolve(),
        args.output.resolve(),
        args.protocol.resolve(),
        args.protocol_lock.resolve(),
        args.resume_existing,
    )
    print(json.dumps({"decision": result["decision"], "tasks": [{"task_id": row["task_id"], "status": row["status"]} for row in result["task_certificates"]], "boundary_passed": result["shared_boundary_probe"]["passed"]}, sort_keys=True))
    return 0 if result["decision"] != "NO_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
