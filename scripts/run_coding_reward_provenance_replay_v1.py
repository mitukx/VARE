#!/usr/bin/env python3
"""Audit stale coding rewards and replay the fixed labels through a tiny SGD policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/coding_reward_provenance_replay_v1.json"
LOCK_PATH = ROOT / "protocols/coding_reward_provenance_replay_v1.lock.json"
TASK_PROTOCOL_PATH = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v4.json"
TASK_PROTOCOL_SHA256 = "6c309a8f32ecdaaea9a07d0527700f9188d9955cce2f3d662e6dc95a6de56e14"
OLD_PROTOCOL_PATH = ROOT / "protocols/verifier_integrity_pilot_v1/protocol-v3.json"
OLD_PROTOCOL_SHA256 = "c209985ec5ae31be26053aacd479f43c31f83349555777b7053e91bae04ee6fb"
EXPECTED_ARCHIVE_SHA256 = "4723319c19f96703f3671fe1de97818d61bbc028044849889964b1b67cfd655f"
EXPECTED_BASELINE_COMMIT = "47e43a4526d42a6dcee758f07eb04b78a8439ecb"
EXPECTED_ORIGINAL = "a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7"
EXPECTED_SOURCE_ONLY = "2dcf69441110ee643a6f5705d63e96980adbb4e73769afccc505aa6df8b4762e"
EXPECTED_SANITIZED = "7b00ef0fe1d1912317a88d04ed9bdf00721e65285eca65a892fca48f70fcff25"
CANDIDATES = (
    "baseline",
    "known-fix",
    "mutant-keep-all",
    "mutant-drop-all",
    "delegate-installed-source",
    "delegate-compiled-bytecode",
)
CANDIDATE_FILES = {
    "baseline": "baseline.py",
    "known-fix": "known_fix.py",
    "mutant-keep-all": "mutant_keep_all.py",
    "mutant-drop-all": "mutant_drop_all.py",
    "delegate-installed-source": "delegate_installed.py",
    "delegate-compiled-bytecode": "delegate_bytecode.py",
}
RUN_ID = "published-base-image-independent-reproduction-1"
TASK_CASE_IDS = (
    "tiny-threshold-1e-08",
    "tiny-threshold-1e-06",
    "tiny-threshold-0.0001",
    "float32-threshold-1e-08",
    "float32-threshold-0.0001",
    "float32-threshold-0.001",
    "float16-threshold-1e-08",
    "float16-threshold-0.0001",
    "float16-threshold-0.001",
    "bfloat16-threshold-1e-08",
    "bfloat16-threshold-0.0001",
    "bfloat16-threshold-0.001",
    "shifted_maximum_nonzero_values",
    "leading_maximum_nonzero_value",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha(value: Any) -> str:
    return sha256(canonical_bytes(value))


def write_new(path: Path, data: bytes) -> None:
    """Create one output atomically without replacing an existing record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_from_image(image: str) -> str:
    marker = "@sha256:"
    if marker not in image:
        raise ValueError(f"image reference is not digest-pinned: {image}")
    value = image.split(marker, 1)[1]
    if len(value) != 64:
        raise ValueError(f"invalid image digest: {image}")
    return value


def image_for(summary: dict[str, Any], condition: str) -> str:
    item = summary["images"][condition]
    return item if isinstance(item, str) else item["image"]


def row_index(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {(row["condition"], row["candidate"]): row for row in summary["outcomes"]}
    if len(rows) != len(summary["outcomes"]):
        raise ValueError("duplicate candidate-image result row")
    return rows


def locked_task_identity() -> dict[str, Any]:
    """Load verifier identities from the frozen task protocol, never a receipt."""
    if sha256_file(TASK_PROTOCOL_PATH) != TASK_PROTOCOL_SHA256:
        raise ValueError("task protocol differs from the frozen replay identity")
    task_protocol = load_json(TASK_PROTOCOL_PATH)
    implementation = task_protocol["implementation_sha256"]
    source_name = "src/vare/mlxlm_1912_repo_task.py"
    adapter_name = "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/grader_adapter.py"
    source_sha = implementation[source_name]
    adapter_sha = implementation[adapter_name]
    if sha256_file(ROOT / source_name) != source_sha:
        raise ValueError("grader source differs from frozen task protocol")
    if sha256_file(ROOT / adapter_name) != adapter_sha:
        raise ValueError("grader adapter differs from frozen task protocol")
    prompt_name = "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md"
    if sha256_file(ROOT / prompt_name) != implementation[prompt_name]:
        raise ValueError("task prompt differs from frozen task protocol")
    for name in ("src/vare/container_isolation.py", "src/vare/isolated_candidate.py"):
        if sha256_file(ROOT / name) != implementation[name]:
            raise ValueError(f"isolation runner differs from frozen task protocol: {name}")
    if len(TASK_CASE_IDS) != 14 or len(set(TASK_CASE_IDS)) != 14:
        raise ValueError("frozen case identity list is malformed")
    return {
        "grader_source_sha256": source_sha,
        "grader_adapter_sha256": adapter_sha,
        "case_ids": TASK_CASE_IDS,
    }


def validate_outcome_identity(
    value: dict[str, Any],
    path: Path,
    row: dict[str, Any],
    *,
    grader_source_sha256: str,
    grader_adapter_sha256: str,
    case_ids: tuple[str, ...],
) -> None:
    cases = value.get("cases")
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError(f"outcome has malformed case records: {path}")
    actual_case_ids = [case.get("case_id") for case in cases]
    if actual_case_ids != list(case_ids):
        raise ValueError(f"outcome case identity differs from frozen task protocol: {path}")
    if any(not isinstance(case.get("passed"), bool) for case in cases):
        raise ValueError(f"outcome has a non-boolean case decision: {path}")
    case_passed = sum(case["passed"] for case in cases)
    if (
        value.get("grader_source_sha256") != grader_source_sha256
        or value.get("grader_adapter_sha256") != grader_adapter_sha256
    ):
        raise ValueError(f"outcome verifier identity differs from frozen task protocol: {path}")
    if value.get("evaluator_case_count") != len(case_ids):
        raise ValueError(f"outcome evaluator case count differs from frozen task protocol: {path}")
    if (
        value.get("candidate_id") != row["candidate"]
        or value.get("passed") is not row["passed"]
        or not isinstance(value.get("passed"), bool)
        or value.get("image") != row["image"]
        or case_passed != row["case_passed"]
        or len(cases) != row["case_count"]
        or value["passed"] is not all(case["passed"] for case in cases)
    ):
        raise ValueError(f"outcome file differs from summary row: {path}")


def read_outcome(
    run_dir: Path,
    row: dict[str, Any],
    *,
    grader_source_sha256: str,
    grader_adapter_sha256: str,
    case_ids: tuple[str, ...],
) -> tuple[dict[str, Any], Path]:
    image_digest = digest_from_image(row["image"])
    path = run_dir / "outcomes" / f"{image_digest}-{row['candidate']}.json"
    value = load_json(path)
    validate_outcome_identity(
        value,
        path,
        row,
        grader_source_sha256=grader_source_sha256,
        grader_adapter_sha256=grader_adapter_sha256,
        case_ids=case_ids,
    )
    return value, path


def verify_inputs(
    run_dir: Path, historical_dir: Path, archive: Path, protocol: dict[str, Any], lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    if sha256_file(PROTOCOL_PATH) != lock["protocol_sha256"]:
        raise ValueError("provenance replay protocol hash mismatch")
    actual_script = sha256_file(Path(__file__).resolve())
    if actual_script != lock["implementation_sha256"]:
        raise ValueError("provenance replay implementation hash mismatch")
    if protocol.get("protocol_id") != "coding-reward-provenance-replay-v1":
        raise ValueError("unexpected provenance replay protocol")
    if sha256_file(OLD_PROTOCOL_PATH) != OLD_PROTOCOL_SHA256:
        raise ValueError("historical task protocol changed")
    task_identity = locked_task_identity()
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256 or archive_sha != protocol["external_base_image_artifact"]["sha256"]:
        raise ValueError("published base-image archive digest mismatch")

    current = load_json(run_dir / "summary.json")
    historical = load_json(historical_dir / "summary.json")
    current_rows = row_index(current)
    old_rows = row_index(historical)
    if current["protocol_sha256"] != OLD_PROTOCOL_SHA256:
        raise ValueError("independent reproduction used an unexpected frozen protocol")
    if current["model_inference"] != "NOT_RUN" or current["optimizer_steps"] != 0:
        raise ValueError("source reproduction must contain no model inference or optimizer step")
    if set(current_rows) != {(cond, candidate) for cond in ("original", "source-only", "fully-sanitized") for candidate in CANDIDATES}:
        raise ValueError("candidate-image outcome set is incomplete or changed")
    if set(old_rows) != set(current_rows):
        raise ValueError("historical and reproduced candidate-image sets differ")
    if canonical_sha(current["image_audits"]) != canonical_sha(historical["image_audits"]):
        raise ValueError("reproduced installed-source/bytecode audits differ from the frozen run")

    for key, row in current_rows.items():
        old = old_rows[key]
        if any(row[name] != old[name] for name in ("passed", "case_passed", "case_count")):
            raise ValueError(f"reproduction differs from retained historical outcome: {key}")
        value, path = read_outcome(
            run_dir,
            row,
            grader_source_sha256=task_identity["grader_source_sha256"],
            grader_adapter_sha256=task_identity["grader_adapter_sha256"],
            case_ids=task_identity["case_ids"],
        )
        candidate_file = ROOT / "benchmarks/verifier_integrity_pilot_v1/candidates" / CANDIDATE_FILES[key[1]]
        if sha256_file(candidate_file) != value["candidate_source_sha256"]:
            raise ValueError(f"candidate source differs from scored candidate: {key[1]}")
        if row["image"] != image_for(current, key[0]):
            raise ValueError(f"row image and summary image differ: {key}")
    if len(current_rows) != 18:
        raise ValueError("expected exactly 18 fixed candidate-image evaluations")
    images = {
        condition: digest_from_image(image_for(current, condition))
        for condition in ("original", "source-only", "fully-sanitized")
    }
    expected = {"original": EXPECTED_ORIGINAL, "source-only": EXPECTED_SOURCE_ONLY, "fully-sanitized": EXPECTED_SANITIZED}
    if images != expected:
        raise ValueError(f"independent image digests differ from historical v3 image identity: {images}")
    if set(current["image_audits"]) != set(images):
        raise ValueError("missing image provenance audits")

    for condition in ("original", "source-only", "fully-sanitized"):
        audit = current["image_audits"][condition]
        if audit["target_path"] != "/usr/local/lib/python3.12/site-packages/mlx_lm/sample_utils.py":
            raise ValueError(f"unexpected audited target path for {condition}")
    original_audit = current["image_audits"]["original"]
    source_audit = current["image_audits"]["source-only"]
    clean_audit = current["image_audits"]["fully-sanitized"]
    if not original_audit["source_exact_fix_paths"] or not original_audit["pyc_exact_function_paths"]:
        raise ValueError("original image did not expose both fixed source and bytecode")
    if source_audit["target_sha256"] != "c93c1eef794725f9f7ce77b6212f61eb6d0fe17b9cd87c06cec6470ee12b07f2":
        raise ValueError("source-only image target is not the pinned baseline")
    if (
        source_audit["pyc_sha256"] != original_audit["pyc_sha256"]
        or not source_audit["pyc_apply_top_p_functions"]
    ):
        raise ValueError("source-only reproduction did not retain the behavior-bearing target bytecode")
    if clean_audit["target_sha256"] != source_audit["target_sha256"] or clean_audit["mlx_lm_pyc_files"] != 0:
        raise ValueError("fully sanitized target source/bytecode state differs from protocol")

    return protocol, current, historical, {"archive": archive, "run": run_dir / "summary.json", "historical": historical_dir / "summary.json"}


def build_provenance_records(run_dir: Path, current: dict[str, Any]) -> list[dict[str, Any]]:
    task_identity = locked_task_identity()
    task_prompt = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md"
    isolation_files = (
        "src/vare/container_isolation.py",
        "src/vare/isolated_candidate.py",
    )
    isolation_identity = {name: sha256_file(ROOT / name) for name in isolation_files}
    case_ids = list(task_identity["case_ids"])
    scorer_source_sha = task_identity["grader_source_sha256"]
    adapter_sha = task_identity["grader_adapter_sha256"]
    test_suite_sha = canonical_sha(
        {
            "grader_source_sha256": scorer_source_sha,
            "grader_adapter_sha256": adapter_sha,
            "task_protocol_sha256": sha256_file(ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v4.json"),
            "task_prompt_sha256": sha256_file(task_prompt),
            "executed_case_ids": case_ids,
            "case_definition_source": "src/vare/mlxlm_1912_repo_task.py",
        }
    )
    scorer_identity = canonical_sha(
        {
            "grader_source_sha256": scorer_source_sha,
            "grader_adapter_sha256": adapter_sha,
            "isolation_runner_sources": isolation_identity,
        }
    )
    old_protocol = load_json(OLD_PROTOCOL_PATH)
    baseline_commit = old_protocol["task"]["baseline_commit"]
    if baseline_commit != EXPECTED_BASELINE_COMMIT:
        raise ValueError("unexpected task baseline")
    rows = row_index(current)
    records: list[dict[str, Any]] = []
    for condition in ("original", "source-only", "fully-sanitized"):
        image = image_for(current, condition)
        image_digest = digest_from_image(image)
        audit = current["image_audits"][condition]
        audit_sha = canonical_sha(audit)
        environment_sha = canonical_sha({"image_manifest_digest": image_digest, "installed_provenance_sha256": audit_sha})
        epoch_sha = canonical_sha({"scorer_identity_sha256": scorer_identity, "test_suite_sha256": test_suite_sha, "environment_identity_sha256": environment_sha})
        for candidate in CANDIDATES:
            row = rows[(condition, candidate)]
            outcome, path = read_outcome(
                run_dir,
                row,
                grader_source_sha256=scorer_source_sha,
                grader_adapter_sha256=adapter_sha,
                case_ids=tuple(case_ids),
            )
            patch_id = canonical_sha({"target_path": "mlx_lm/sample_utils.py", "candidate_source_sha256": outcome["candidate_source_sha256"]})
            cases = [{"case_id": item["case_id"], "passed": item["passed"], "failure": item["failure"]} for item in outcome["cases"]]
            records.append(
                {
                    "task_id": "mlxlm-1912-top-p-min-token",
                    "task_baseline_commit": baseline_commit,
                    "candidate_id": candidate,
                    "candidate_patch_identity_sha256": patch_id,
                    "candidate_target_path": "mlx_lm/sample_utils.py",
                    "candidate_source_sha256": outcome["candidate_source_sha256"],
                    "scorer_identity_sha256": scorer_identity,
                    "grader_source_sha256": scorer_source_sha,
                    "grader_adapter_sha256": adapter_sha,
                    "executed_test_bundle_sha256": test_suite_sha,
                    "image_reference": image,
                    "executable_image_manifest_digest": image_digest,
                    "installed_source_bytecode_provenance_sha256": audit_sha,
                    "environment_identity_sha256": environment_sha,
                    "verifier_epoch_sha256": epoch_sha,
                    "result": {"task_passed": outcome["passed"], "reward": float(outcome["passed"]), "case_passed": row["case_passed"], "case_count": row["case_count"], "case_decisions": cases},
                    "evaluation_invocation_id": f"{RUN_ID}:{image_digest}:{candidate}",
                    "evaluated_at_utc": current["completed_at_utc"],
                    "timestamp_scope": "run-completion timestamp; not a per-candidate timestamp",
                    "execution_receipt_sha256": sha256_file(path),
                    "execution_receipt_path": path.relative_to(ROOT).as_posix(),
                    "model_identity": None,
                    "checkpoint_identity": None,
                    "trajectory_origin": "fixed scripted candidate; not model-generated",
                }
            )
    if len({record["scorer_identity_sha256"] for record in records}) != 1:
        raise ValueError("scoring code identity changed across conditions")
    if len({record["executed_test_bundle_sha256"] for record in records}) != 1:
        raise ValueError("executed test identity changed across conditions")
    if len({record["verifier_epoch_sha256"] for record in records}) != 3:
        raise ValueError("environment change did not create exactly three verifier epochs")
    return records


def optimizer_replay(rewards: list[int], invalid_indices: set[int]) -> dict[str, Any]:
    if len(rewards) != len(CANDIDATES):
        raise ValueError("optimizer replay requires one frozen reward per candidate ID")
    logits = torch.nn.Parameter(torch.zeros(len(CANDIDATES), dtype=torch.float64, device="cpu"))
    before = torch.softmax(logits.detach(), dim=0)
    invalid_before = float(before[list(sorted(invalid_indices))].sum())
    if not any(rewards):
        return {
            "optimizer_calls": 0,
            "optimizer_step_calls": 0,
            "initial_logits": logits.detach().tolist(),
            "final_logits": logits.detach().tolist(),
            "initial_probabilities": before.tolist(),
            "final_probabilities": before.tolist(),
            "invalid_delegate_mass_before": invalid_before,
            "invalid_delegate_mass_after": invalid_before,
            "invalid_delegate_mass_delta": 0.0,
            "loss": None,
        }
    optimizer = torch.optim.SGD([logits], lr=1.0)
    action_ids = torch.arange(len(CANDIDATES), dtype=torch.long)
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    advantages = reward_tensor - reward_tensor.mean()
    loss = -(advantages * torch.log_softmax(logits, dim=0)[action_ids]).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = torch.softmax(logits.detach(), dim=0)
    invalid_after = float(after[list(sorted(invalid_indices))].sum())
    return {
        "optimizer_calls": 1,
        "optimizer_step_calls": 1,
        "initial_logits": [0.0] * len(CANDIDATES),
        "final_logits": logits.detach().tolist(),
        "initial_probabilities": before.tolist(),
        "final_probabilities": after.tolist(),
        "invalid_delegate_mass_before": invalid_before,
        "invalid_delegate_mass_after": invalid_after,
        "invalid_delegate_mass_delta": invalid_after - invalid_before,
        "loss": float(loss.detach()),
        "torch_version": torch.__version__,
        "optimizer_identity": "torch.optim.SGD(lr=1.0), one CPU float64 step",
    }


def admission_results(
    records: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any]:
    by_key = {(item["image_reference"].split("@sha256:", 1)[1][:64], item["candidate_id"]): item for item in records}
    original_digest = EXPECTED_ORIGINAL
    trusted_digest = EXPECTED_SANITIZED
    stale = [by_key[(original_digest, candidate)] for candidate in CANDIDATES]
    trusted = [by_key[(trusted_digest, candidate)] for candidate in CANDIDATES]
    true_candidate = "known-fix"
    invalid_indices = {CANDIDATES.index("delegate-installed-source"), CANDIDATES.index("delegate-compiled-bytecode")}
    original_rewards = [int(item["result"]["task_passed"]) for item in stale]
    trusted_rewards = [int(item["result"]["task_passed"]) for item in trusted]
    if original_rewards != [0, 1, 0, 0, 1, 1]:
        raise ValueError(f"unexpected contaminated reward vector: {original_rewards}")
    if trusted_rewards != [0, 1, 0, 0, 0, 0]:
        raise ValueError(f"unexpected trusted reward vector: {trusted_rewards}")

    wall_s = sum(float(row["candidate_elapsed_seconds"]) for row in current["outcomes"] if row["condition"] == "fully-sanitized")
    mode_specs = (
        ("version-unaware-reuse", stale, original_rewards, 0, 0),
        ("blanket-stale-rejection", [], [0] * len(CANDIDATES), 0, len(CANDIDATES)),
        ("provenance-aware-trusted-regrade", trusted, trusted_rewards, len(CANDIDATES), 0),
    )
    arms = []
    for arm_id, admitted, rewards, regrades, rejected_old in mode_specs:
        invalid_positive = [
            CANDIDATES[index]
            for index in invalid_indices
            if index < len(rewards) and rewards[index] > 0
        ]
        accepted_positive = [CANDIDATES[index] for index, reward in enumerate(rewards) if reward > 0]
        false_rejection = int(true_candidate not in accepted_positive)
        replay = optimizer_replay(rewards, invalid_indices)
        case_execs = regrades * 14
        arm_wall = wall_s if arm_id == "provenance-aware-trusted-regrade" else 0.0
        arms.append(
            {
                "arm": arm_id,
                "admitted_reward_records": len(admitted),
                "admitted_positive_records": len(accepted_positive),
                "accepted_positive_candidate_ids": accepted_positive,
                "false_reward_admissions": len(invalid_positive),
                "false_reward_admission_denominator": 5,
                "false_reward_admission_rate_over_invalid_candidates": len(invalid_positive) / 5,
                "invalid_positive_candidate_ids": invalid_positive,
                "genuine_fix_false_rejections": false_rejection,
                "genuine_fix_false_rejection_denominator": 1,
                "grpo_eligible_groups_retained": 0,
                "reason_no_grpo_groups": "Fixed scripted candidate records contain no policy log-probabilities or on-policy rollout identity.",
                "additional_candidate_grade_invocations": regrades,
                "additional_test_case_executions": case_execs,
                "observed_additional_candidate_wall_seconds": arm_wall,
                "candidate_cpu_seconds_upper_bound_at_one_cpu_quota": arm_wall,
                "candidate_cpu_seconds_observed": None,
                "optimizer_replay": {
                    **replay,
                    "admitted_reward_vector": rewards,
                    "admitted_candidate_ids": [candidate for candidate, reward in zip(CANDIDATES, rewards, strict=True) if arm_id != "blanket-stale-rejection"],
                    "invalid_positive_records_in_optimizer_batch": len(invalid_positive),
                    "includes_patch_text_or_historical_fix_source": False,
                },
            }
        )
    return {
        "true_positive_candidate_id": true_candidate,
        "trusted_oracle_image_digest": trusted_digest,
        "scorer_code_identity_unchanged_across_image_change": True,
        "environment_identity_changed": True,
        "original_reward_vector": original_rewards,
        "trusted_reward_vector": trusted_rewards,
        "image_epoch_transition_stale_rewards": True,
        "candidate_case_count": 14,
        "sanitized_candidate_grade_wall_seconds_sum": wall_s,
        "sanitized_candidate_grade_wall_seconds_median": statistics.median(
            float(row["candidate_elapsed_seconds"])
            for row in current["outcomes"] if row["condition"] == "fully-sanitized"
        ),
        "arms": arms,
        "limitations": [
            "The original image is untrusted; the sanitized image is a per-target corrected reference, not a universal clean-environment certificate.",
            "CPU usage counters were not retained. The reported CPU figure is only an upper bound from one-core quota multiplied by observed candidate-grade wall time.",
            "The optimizer is a real PyTorch SGD API call on a six-logit scripted selector, not an LLM, VARE/RVL GRPO trainer, or on-policy update.",
            "No hidden tests, model-generated samples, checkpoint, or held-out task were used.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--published-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists; refusing to overwrite: {args.output}")
    protocol = load_json(PROTOCOL_PATH)
    lock = load_json(LOCK_PATH)
    torch.manual_seed(0)
    if torch.__version__.split("+", 1)[0] != protocol["offline_optimizer_replay"]["torch_version"]:
        raise RuntimeError(
            "optimizer replay torch version differs from frozen protocol: "
            f"expected {protocol['offline_optimizer_replay']['torch_version']}, found {torch.__version__}"
        )
    _, current, historical, input_paths = verify_inputs(
        args.run_dir.resolve(strict=True), args.historical_dir.resolve(strict=True), args.published_archive.resolve(strict=True), protocol, lock
    )
    records = build_provenance_records(args.run_dir.resolve(), current)
    results = admission_results(records, current)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "reward-provenance-records.jsonl"
    write_new(records_path, b"".join(canonical_bytes(item) + b"\n" for item in records))
    result_path = output / "admission-and-optimizer-replay.json"
    write_new(result_path, json.dumps(results, indent=2, sort_keys=True).encode() + b"\n")
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "implementation_sha256": lock["implementation_sha256"],
        "created_at_utc": current["completed_at_utc"],
        "reproduction_run_id": RUN_ID,
        "historical_run_id": "frozen-verifier-integrity-pilot-v3-run3",
        "independent_reproduction_match": {"candidate_image_conditions": 18, "matching_outcomes": 18},
        "base_image_archive_sha256": sha256_file(args.published_archive),
        "base_image_archive_bytes": args.published_archive.stat().st_size,
        "torch_version": torch.__version__,
        "model_inference_calls": 0,
        "language_model_optimizer_updates": 0,
        "grpo_eligible_groups": 0,
        "analysis": results,
    }
    summary_path = output / "summary.json"
    write_new(summary_path, json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n")
    manifest = {
        "manifest_version": 1,
        "protocol_sha256": lock["protocol_sha256"],
        "implementation_sha256": lock["implementation_sha256"],
        "inputs": {name: {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in input_paths.items()},
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (records_path, result_path, summary_path)
        },
        "note": "This manifest is not self-hashed or signed. Artifact integrity is content-addressed; publisher identity requires an external signature/attestation."
    }
    write_new(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"output": str(output), "manifest_sha256": sha256_file(output / "manifest.json"), "records": len(records), "arms": len(results["arms"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
