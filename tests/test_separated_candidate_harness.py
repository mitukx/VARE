from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
from vare.separated_candidate_harness import (
    PROTOCOL_PATH,
    SeparatedHarnessError,
    grade_function,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "benchmarks/mlsys_coding_v1/tasks/global_grad_clip"
TASK_ID = "mlsys-v1-global-grad-clip"
ENTRYPOINT = "global_grad_clip:clip_by_global_norm"


def _subject(root: Path, source: str) -> Path:
    subject = root / "subject"
    subject.mkdir(parents=True)
    (subject / "global_grad_clip.py").write_text(source, encoding="utf-8")
    return subject


def _request(subject: Path) -> dict:
    return {
        "workspace": str(subject),
        "entrypoint": ENTRYPOINT,
        "task_type": "correctness",
        "trusted_task_path": str(TASK_DIR / "task.json"),
        "nonce": "ab" * 16,
    }


def _pack_adapter():
    path = ROOT / "benchmarks/mlsys_coding_v1/harness_adapter.py"
    spec = importlib.util.spec_from_file_location("frozen_mlsys_harness_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_task_adapter_uses_isolated_shape_aware_grader(tmp_path: Path) -> None:
    module = _pack_adapter()
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    from mlsys_task_isolated_calls import _load_evaluator

    evaluator = _load_evaluator(
        TASK_DIR / "evaluator/test_hidden.py", TASK_DIR.parents[1], TASK_ID,
    )
    args, kwargs = evaluator._generator(0)
    expected = evaluator._reference(*args, **kwargs)
    # Baseline reproducer: the frozen nested zip checker accepts an empty
    # result without comparing either the outer or inner shape.
    evaluator._checker([], expected, list(args), {}, args, kwargs)

    request = module.HarnessRequest(
        workspace=_subject(
            tmp_path,
            "def clip_by_global_norm(grads, max_norm):\n    return []\n",
        ),
        entrypoint=ENTRYPOINT,
        task_type="correctness",
        trusted_task_path=TASK_DIR / "task.json",
        input_generator=lambda _index: ((), {}),
        reference=lambda: None,
        checker=lambda *_args: None,
        repeats=4,
        timeout_s=20,
    )

    report = module.run_harness(request)

    assert report.passed is False
    assert report.harness_id == "vare.separated_candidate_harness.grade_function"
    assert report.nonce_enforced is True
    assert report.error == "correctness_failed"


def test_frozen_oracle_passes_through_the_same_isolated_adapter(tmp_path: Path) -> None:
    module = _pack_adapter()
    subject = tmp_path / "subject"
    subject.mkdir()
    shutil.copy(
        TASK_DIR / "evaluator/oracle/global_grad_clip.py",
        subject / "global_grad_clip.py",
    )
    request = module.HarnessRequest(
        workspace=subject,
        entrypoint=ENTRYPOINT,
        task_type="correctness",
        trusted_task_path=TASK_DIR / "task.json",
        input_generator=lambda _index: ((), {}),
        reference=lambda: None,
        checker=lambda *_args: None,
        repeats=4,
        timeout_s=20,
    )

    report = module.run_harness(request)

    assert report.passed is True
    assert report.nonce_enforced is True
    assert report.harness_id == "vare.separated_candidate_harness.grade_function"


def test_task_descriptor_cannot_be_substituted(tmp_path: Path) -> None:
    forged = tmp_path / "task.json"
    forged.write_text(
        '{"id":"mlsys-v1-global-grad-clip","metadata":{"task_type":"correctness"}}\n',
        encoding="utf-8",
    )
    subject = _subject(tmp_path, "def clip_by_global_norm(*args):\n    return []\n")
    payload = _request(subject)
    payload["trusted_task_path"] = str(forged)

    with pytest.raises(SeparatedHarnessError, match="canonical pack path"):
        grade_function(payload)


def test_frozen_pack_manifest_detects_file_tampering(tmp_path: Path, monkeypatch) -> None:
    import vare.separated_candidate_harness as harness

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    pack = tmp_path / "benchmarks/mlsys_coding_v1"
    pack.mkdir(parents=True)
    (pack / "file.txt").write_text("original", encoding="utf-8")
    entry = {
        "bytes": len(b"original"),
        "path": "file.txt",
        "sha256": hashlib.sha256(b"original").hexdigest(),
    }
    manifest = json.dumps(
        {"files": [entry], "pack_id": "test"}, indent=2, sort_keys=True,
    ) + "\n"
    (pack / "manifest.json").write_text(manifest, encoding="utf-8")
    protocol["supported_packs"] = {
        pack.relative_to(tmp_path).as_posix(): {
            "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        },
    }

    harness._verify_pack(pack, protocol)
    (pack / "file.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(SeparatedHarnessError, match="file hash mismatch"):
        harness._verify_pack(pack, protocol)


def test_malformed_workspace_fails_closed(tmp_path: Path) -> None:
    task_path = TASK_DIR / "task.json"
    payload = _request(tmp_path)
    payload["trusted_task_path"] = str(task_path)
    payload["entrypoint"] = "global_grad_clip:other"
    with pytest.raises(SeparatedHarnessError, match="entrypoint"):
        grade_function(payload)


def test_dataset_standardization_fails_closed_without_mutation_observation(tmp_path: Path) -> None:
    from vare.separated_candidate_harness import PACK_ROOTS, _task_adapter

    dataset_task = PACK_ROOTS["mlsys-v1-"] / "tasks/dataset_standardize/task.json"
    assert dataset_task.is_file()
    with pytest.raises(SeparatedHarnessError, match="no isolated call materializer"):
        _task_adapter("mlsys-v1-dataset-standardize")


@pytest.mark.parametrize(
    ("task_id", "module", "function"),
    [
        ("mlsys-v3-ema-bias", "ema_bias", "bias_corrected_ema"),
        ("mlsys-v3-label-smooth", "label_smooth", "smooth_targets"),
    ],
)
def test_v3_oracle_controls_use_canonical_task_paths(
    tmp_path: Path, task_id: str, module: str, function: str,
) -> None:
    task_dir = ROOT / "benchmarks/mlsys_coding_v3/tasks" / task_id
    oracle_files = list((task_dir / "evaluator/oracle").glob("*.py"))
    assert len(oracle_files) == 1
    subject = tmp_path / "subject"
    subject.mkdir()
    shutil.copy(oracle_files[0], subject / f"{module}.py")
    report = grade_function({
        "workspace": str(subject),
        "entrypoint": f"{module}:{function}",
        "task_type": "correctness",
        "trusted_task_path": str(task_dir / "task.json"),
        "nonce": "cd" * 16,
    })
    assert report["passed"] is True
    assert report["nonce_enforced"] is True
