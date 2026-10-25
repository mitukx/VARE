from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vare.mlsys_heldout_task_pilot_v1 import (
    QualificationError,
    load_frozen_protocol,
    materialize_candidate,
    mutate_target,
    package_delta,
    sha256,
    unified_target_patch,
    validate_patch,
    verify_candidate_surface,
    write_exclusive_atomic,
)
from scripts.audit_heldout_task_exposure_pilot_v1 import PATTERNS
from scripts.qualify_mlsys_heldout_task_pilot_v1 import TASK_NAMES


def test_frozen_protocol_and_original_heldout_allocation_are_verified():
    protocol, lock = load_frozen_protocol()

    assert protocol["status"] == "FROZEN_BEFORE_NEW_GRADER_EXECUTION"
    assert lock["protocol_sha256"] == hashlib.sha256(
        (Path(__file__).resolve().parents[1] / lock["protocol_path"]).read_bytes()
    ).hexdigest()
    assert {task["split"] for task in protocol["task_contracts"]} == {"heldout"}
    for task_id, directory in TASK_NAMES.items():
        assert (Path(__file__).resolve().parents[1] / "benchmarks/mlsys_repo_eval_v3" / directory / "grader_adapter.py").is_file(), task_id


def test_second_frozen_protocol_keeps_phi3small_in_original_heldout_split():
    root = Path(__file__).resolve().parents[1]
    protocol, lock = load_frozen_protocol(
        root / "protocols/mlsys_heldout_task_qualification_pilot_v2.json",
        root / "protocols/mlsys_heldout_task_qualification_pilot_v2.lock.json",
    )

    assert protocol["status"] == "FROZEN_BEFORE_NEW_GRADER_EXECUTION"
    assert [task["task_id"] for task in protocol["task_contracts"]] == [
        "mlxlm-1938-phi3small-logits-mask"
    ]
    assert lock["exposure_audit_sha256"] == sha256(
        (root / protocol["exposure"]["audit_path"]).read_bytes()
    )


def test_third_frozen_protocol_keeps_ssm_task_in_original_heldout_split():
    root = Path(__file__).resolve().parents[1]
    protocol, lock = load_frozen_protocol(
        root / "protocols/mlsys_heldout_task_qualification_pilot_v3.json",
        root / "protocols/mlsys_heldout_task_qualification_pilot_v3.lock.json",
    )

    assert protocol["status"] == "FROZEN_BEFORE_NEW_GRADER_EXECUTION"
    assert [task["task_id"] for task in protocol["task_contracts"]] == [
        "mlxlm-1979-ssm-padding-mask"
    ]
    assert lock["exposure_audit_sha256"] == sha256(
        (root / protocol["exposure"]["audit_path"]).read_bytes()
    )


@pytest.mark.parametrize(
    ("task_id", "fixed", "needle", "replacement"),
    [
        (
            "mlxlm-1902-continuation-score-truncation",
            b"completion_start = len(prefix)\n",
            b"completion_start = len(prefix)\n",
            b"completion_start = len(prefix) - 1\n",
        ),
        (
            "mlxlm-1926-falcon-cache-gradient",
            b"cache = [(None, None)] * len(self.layers)\n",
            b"cache = [(None, None)] * len(self.layers)\n",
            b"cache = [(None, None)] * max(len(self.layers) - 1, 0)\n",
        ),
        (
            "mlxlm-1938-phi3small-logits-mask",
            b'out[..., self._dummy_tokenizer_ids] = -float("inf")\n',
            b'out[..., self._dummy_tokenizer_ids] = -float("inf")\n',
            b'out[self._dummy_tokenizer_ids] = -float("inf")\n',
        ),
        (
            "mlxlm-1979-ssm-padding-mask",
            b"dtA = dtA * mask[..., None]\n",
            b"dtA = dtA * mask[..., None]\n",
            b"dtA = dtA\n",
        ),
    ],
)
def test_prespecified_wrong_mutant_is_deterministic(task_id, fixed, needle, replacement):
    source = b"before\n" + fixed + b"after\n"

    mutated = mutate_target(task_id, source)

    assert mutated == b"before\n" + replacement + b"after\n"
    assert sha256(mutated) != sha256(source)


def test_ssm_exposure_pattern_matches_task_semantics_not_adjacent_cache_task():
    pattern = PATTERNS["mlxlm-1979-ssm-padding-mask"]

    assert pattern.search(b"Fix ssm_right_pad for right-padded sequences")
    assert pattern.search(b"MLX-LM #1979: state-space padding regression")
    assert not pattern.search(
        b"Task #1978 rotating cache attention after wrap; preserve behavior for caches with no retained prefix"
    )


def test_ssm_exposure_false_positive_scan_is_preserved_and_superseded():
    root = Path(__file__).resolve().parents[1]
    initial = json.loads(
        (root / "results/mlsys-heldout-task-qualification-pilot-v3-exposure-audit.json").read_text()
    )
    corrected = json.loads(
        (root / "results/mlsys-heldout-task-qualification-pilot-v4-exposure-audit.json").read_text()
    )

    assert len(initial["candidate_rollout_or_training_matches"]) == 4
    assert all(
        "1978-rotating-cache" in row["path"]
        for row in initial["candidate_rollout_or_training_matches"]
    )
    assert corrected["candidate_rollout_or_training_matches"] == []


def test_patch_allowlist_accepts_only_single_target_regular_file():
    target = "mlx_lm/evaluate.py"
    patch = unified_target_patch(target, b"old\n", b"new\n")

    assert validate_patch(patch, target) == [target, target]


@pytest.mark.parametrize(
    "patch",
    [
        "diff --git a/../grader/answer b/../grader/answer\n"
        "--- a/../grader/answer\n+++ b/../grader/answer\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a//grader/answer b//grader/answer\n"
        "--- a//grader/answer\n+++ b//grader/answer\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a/mlx_lm/other.py b/mlx_lm/other.py\n"
        "--- a/mlx_lm/other.py\n+++ b/mlx_lm/other.py\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a/mlx_lm/evaluate.py b/mlx_lm/evaluate.py\n"
        "--- a/mlx_lm/evaluate.py\n+++ b/mlx_lm/evaluate.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/mlx_lm/models/other.py b/mlx_lm/models/other.py\n"
        "--- a/mlx_lm/models/other.py\n+++ b/mlx_lm/models/other.py\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a/mlx_lm/evaluate.py b/mlx_lm/evaluate.py\n"
        "--- a/mlx_lm/evaluate.py\n+++ b/mlx_lm/evaluate.py\nnew file mode 100644\n",
    ],
)
def test_patch_allowlist_rejects_traversal_absolute_non_target_multi_file_and_create(patch):
    with pytest.raises(QualificationError):
        validate_patch(patch, "mlx_lm/evaluate.py")


def test_candidate_surface_rejects_hash_drift_and_non_python_files(tmp_path):
    root = tmp_path / "subject"
    (root / "mlx_lm").mkdir(parents=True)
    (root / "mlx_lm/evaluate.py").write_text("pass\n")
    (root / "vare_task_adapter.py").write_text("def run(): return 1\n")
    expected = {
        "mlx_lm/evaluate.py": hashlib.sha256(b"pass\n").hexdigest(),
        "vare_task_adapter.py": hashlib.sha256(b"def run(): return 1\n").hexdigest(),
    }

    assert verify_candidate_surface(root, expected) == expected
    (root / "mlx_lm/evaluate.py").write_text("changed\n")
    with pytest.raises(QualificationError, match="hash changed"):
        verify_candidate_surface(root, expected)
    (root / "mlx_lm/evaluate.py").write_text("pass\n")
    (root / "payload.json").write_text("{}")
    with pytest.raises(QualificationError, match="non-source"):
        verify_candidate_surface(root, expected)


def test_candidate_surface_rejects_target_symlink_and_protected_grader_tree(tmp_path):
    root = tmp_path / "subject"
    (root / "mlx_lm").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    (root / "mlx_lm/evaluate.py").symlink_to(outside)
    with pytest.raises(QualificationError, match="symlink"):
        verify_candidate_surface(root, {"mlx_lm/evaluate.py": sha256(b"pass\n")})

    (root / "mlx_lm/evaluate.py").unlink()
    (root / "mlx_lm/evaluate.py").write_text("pass\n")
    (root / "tests").mkdir()
    with pytest.raises(QualificationError, match="protected assets"):
        verify_candidate_surface(
            root,
            {
                "mlx_lm/evaluate.py": sha256(b"pass\n"),
                "tests/test_hidden.py": sha256(b"pass\n"),
            },
        )


def test_package_delta_counts_only_pinned_target_changes():
    base = {"mlx_lm/evaluate.py": b"a\n", "mlx_lm/other.py": b"b\n"}
    candidate = {
        "mlx_lm/evaluate.py": sha256(b"c\n"),
        "mlx_lm/other.py": sha256(b"b\n"),
        "vare_task_adapter.py": sha256(b"shim\n"),
    }

    assert package_delta(base, candidate) == ["mlx_lm/evaluate.py"]


def test_candidate_materialization_contains_the_selected_target_and_is_immutable(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def run(): return 1\n")
    destination = tmp_path / "subject"

    hashes = materialize_candidate(
        {"mlx_lm/evaluate.py": b"baseline\n", "mlx_lm/other.py": b"stable\n"},
        "mlx_lm/evaluate.py",
        b"fixed\n",
        adapter,
        destination,
    )

    assert (destination / "mlx_lm/evaluate.py").read_bytes() == b"fixed\n"
    assert (destination / "mlx_lm/other.py").read_bytes() == b"stable\n"
    assert hashes["mlx_lm/evaluate.py"] == sha256(b"fixed\n")
    with pytest.raises(PermissionError):
        (destination / "mlx_lm/evaluate.py").write_text("tampered\n")


def test_exclusive_atomic_artifact_write_never_overwrites(tmp_path):
    artifact = tmp_path / "record.json"

    digest = write_exclusive_atomic(artifact, b'{"ok":true}\n')

    assert artifact.read_bytes() == b'{"ok":true}\n'
    assert digest == sha256(artifact.read_bytes())
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(QualificationError, match="already exists"):
        write_exclusive_atomic(artifact, b'{"ok":false}\n')
    assert artifact.read_bytes() == b'{"ok":true}\n'
