from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_mlsys_vllm_64830_agent_grader_v1 import grade_candidate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_ROOT = (
    REPOSITORY_ROOT
    / "results/mlsys-fresh-vllm-64830-agent-attempt-v1/trusted"
)
TEST_FILE = TRUSTED_ROOT / "test_vllm_config_unsupported_features.py"
PYTEST_CONFIG = TRUSTED_ROOT / "pytest.ini"
METHOD_WORKER = (
    REPOSITORY_ROOT / "scripts/run_mlsys_vllm_64830_agent_method_worker_v1.py"
)
TEST_FILE_SHA256 = "c397c9407f3294460ce939fab3c43835da75844ec0e274e2b9dfc8f44832c28e"
PYTEST_CONFIG_SHA256 = "8250d20f42b419af40706490ead78aabab9feba93eab09a3f626e3de5256e1f4"
SOURCE_PATH = "vllm/config/vllm.py"
BASELINE_SOURCE = """class VllmConfig:
    def _get_v1_model_runner_unsupported_features(self):
        unsupported = []
        return unsupported
"""
FIXED_SOURCE = """class VllmConfig:
    def _get_v1_model_runner_unsupported_features(self):
        unsupported = []
        if self.attention_config.hisparse_config is not None:
            unsupported.append("HiSparse")
        if self.watermark_config is not None:
            unsupported.append("watermarking")
        return unsupported
"""
PARTIAL_SOURCE = """class VllmConfig:
    def _get_v1_model_runner_unsupported_features(self):
        unsupported = []
        if self.attention_config.hisparse_config is not None:
            unsupported.append("HiSparse")
        return unsupported
"""
SLOW_SOURCE = """class VllmConfig:
    def _get_v1_model_runner_unsupported_features(self):
        import time
        time.sleep(15)
        return []
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_workspace(root: Path) -> tuple[Path, str, Path]:
    workspace = root / "candidate"
    source = workspace / SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_text(BASELINE_SOURCE, encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "VARE grader test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(workspace), "add", SOURCE_PATH], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "--quiet", "-m", "baseline"],
        check=True,
    )
    baseline_revision = subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
    ).strip()
    return workspace, baseline_revision, source


def _grade(
    workspace: Path,
    baseline_revision: str,
    output_dir: Path,
    *,
    timeout_seconds: int = 30,
) -> dict:
    return grade_candidate(
        workspace=workspace,
        baseline_revision=baseline_revision,
        baseline_source_sha256=_sha256(BASELINE_SOURCE.encode("utf-8")),
        source_relative_path=SOURCE_PATH,
        test_file=TEST_FILE,
        test_file_sha256=TEST_FILE_SHA256,
        pytest_config=PYTEST_CONFIG,
        pytest_config_sha256=PYTEST_CONFIG_SHA256,
        method_worker=METHOD_WORKER,
        method_worker_sha256=_sha256(METHOD_WORKER.read_bytes()),
        method_profile=None,
        method_profile_sha256=None,
        python=Path(sys.executable),
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )


def test_baseline_executes_hidden_cases_and_fails_both(tmp_path: Path) -> None:
    workspace, revision, _ = _candidate_workspace(tmp_path)

    result = _grade(workspace, revision, tmp_path / "baseline-grade")

    assert result["classification"] == "TARGET_GATE_FAILED"
    assert result["grader_call_count"] == 1
    assert result["pytest_summary"] == {"failed": 2}


def test_known_fix_passes_both_cases(tmp_path: Path) -> None:
    workspace, revision, source = _candidate_workspace(tmp_path)
    source.write_text(FIXED_SOURCE, encoding="utf-8")

    result = _grade(workspace, revision, tmp_path / "fixed-grade")

    assert result["classification"] == "TARGET_GATE_PASSED"
    assert result["grader_call_count"] == 1
    assert result["pytest_summary"] == {"passed": 2}
    assert result["candidate_diff_bytes"] > 0


def test_partial_fix_is_classified_as_target_failure(tmp_path: Path) -> None:
    workspace, revision, source = _candidate_workspace(tmp_path)
    source.write_text(PARTIAL_SOURCE, encoding="utf-8")

    result = _grade(workspace, revision, tmp_path / "partial-grade")

    assert result["classification"] == "TARGET_GATE_FAILED"
    assert result["grader_call_count"] == 1
    assert result["pytest_summary"] == {"failed": 1, "passed": 1}


@pytest.mark.parametrize("bad_hash", ["0" * 64])
def test_trusted_test_hash_mismatch_fails_before_grading(
    tmp_path: Path, bad_hash: str
) -> None:
    workspace, revision, _ = _candidate_workspace(tmp_path)
    result = grade_candidate(
        workspace=workspace,
        baseline_revision=revision,
        baseline_source_sha256=_sha256(BASELINE_SOURCE.encode("utf-8")),
        source_relative_path=SOURCE_PATH,
        test_file=TEST_FILE,
        test_file_sha256=bad_hash,
        pytest_config=PYTEST_CONFIG,
        pytest_config_sha256=PYTEST_CONFIG_SHA256,
        method_worker=METHOD_WORKER,
        method_worker_sha256=_sha256(METHOD_WORKER.read_bytes()),
        method_profile=None,
        method_profile_sha256=None,
        python=Path(sys.executable),
        output_dir=tmp_path / "hash-mismatch-grade",
        timeout_seconds=30,
    )

    assert result["classification"] == "NOT_GRADED"
    assert result["grader_call_count"] == 0
    assert result["workspace_errors"] == ["trusted test overlay hash mismatch"]


def test_non_allowlisted_file_fails_before_grading(tmp_path: Path) -> None:
    workspace, revision, source = _candidate_workspace(tmp_path)
    source.write_text(FIXED_SOURCE, encoding="utf-8")
    (workspace / "unrelated.txt").write_text("untrusted", encoding="utf-8")

    result = _grade(workspace, revision, tmp_path / "allowlist-grade")

    assert result["classification"] == "NOT_GRADED"
    assert result["grader_call_count"] == 0
    assert result["changed_paths"] == ["unrelated.txt", SOURCE_PATH]


def test_candidate_syntax_error_is_not_a_semantic_grade(tmp_path: Path) -> None:
    workspace, revision, source = _candidate_workspace(tmp_path)
    source.write_text("class VllmConfig(:\n", encoding="utf-8")

    result = _grade(workspace, revision, tmp_path / "invalid-source-grade")

    assert result["classification"] == "INVALID_CANDIDATE_SOURCE"
    assert result["grader_call_count"] == 0


def test_timeout_is_inconclusive_and_bounded(tmp_path: Path) -> None:
    workspace, revision, source = _candidate_workspace(tmp_path)
    source.write_text(SLOW_SOURCE, encoding="utf-8")

    result = _grade(
        workspace,
        revision,
        tmp_path / "timeout-grade",
        timeout_seconds=1,
    )

    assert result["classification"] == "INCONCLUSIVE_TIMEOUT"
    assert result["grader_call_count"] == 1
