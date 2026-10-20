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
TASK_DIR = ROOT / "benchmarks/mlsys_coding_v1/tasks/causal_attention"
TASK_ID = "mlsys-v1-causal-attention"
ENTRYPOINT = "causal_attention:causal_attention_weights"


def _subject(root: Path, source: str) -> Path:
    subject = root / "subject"
    subject.mkdir(parents=True)
    (subject / "causal_attention.py").write_text(source, encoding="utf-8")
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
    with pytest.raises(AssertionError):
        evaluator._checker([], expected, list(args), {}, args, kwargs)

    request = module.HarnessRequest(
        workspace=_subject(
            tmp_path,
            "def causal_attention_weights(scores):\n    return []\n",
        ),
        entrypoint=ENTRYPOINT,
        task_type="correctness",
        trusted_task_path=TASK_DIR / "task.json",
        input_generator=evaluator._generator,
        reference=evaluator._reference,
        checker=evaluator._checker,
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
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    from mlsys_task_isolated_calls import _load_evaluator

    evaluator = _load_evaluator(
        TASK_DIR / "evaluator/test_hidden.py", TASK_DIR.parents[1], TASK_ID,
    )
    subject = tmp_path / "subject"
    subject.mkdir()
    shutil.copy(
        TASK_DIR / "evaluator/oracle/causal_attention.py",
        subject / "causal_attention.py",
    )
    request = module.HarnessRequest(
        workspace=subject,
        entrypoint=ENTRYPOINT,
        task_type="correctness",
        trusted_task_path=TASK_DIR / "task.json",
        input_generator=evaluator._generator,
        reference=evaluator._reference,
        checker=evaluator._checker,
        repeats=4,
        timeout_s=20,
    )

    report = module.run_harness(request)

    assert report.passed is True
    assert report.nonce_enforced is True
    assert report.harness_id == "vare.separated_candidate_harness.grade_function"


def test_candidate_stdout_does_not_corrupt_isolated_result_protocol(tmp_path: Path) -> None:
    import subprocess

    from vare.isolated_candidate import _CALLEE

    task_dir = ROOT / "benchmarks/mlsys_coding_v1/tasks/causal_attention"
    source = (task_dir / "evaluator/oracle/causal_attention.py").read_text(encoding="utf-8")
    source = source.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nprint('candidate import output')\n",
        1,
    )
    source = source.replace("import math\n", "import math\nimport os\n", 1)
    source = source.replace(
        "def causal_attention_weights(scores):\n",
        "def causal_attention_weights(scores):\n"
        "    print('candidate call output')\n"
        "    os.write(1, b'candidate fd output')\n",
        1,
    )
    subject = _subject(tmp_path, source)

    callee = _CALLEE.replace('"/subject"', repr(str(subject)), 1)
    request = {
        "args": [[[0.0]]],
        "kwargs": {},
        "tuple_paths": [],
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", callee],
        input=(
            json.dumps({
                "module": "causal_attention",
                "function": "causal_attention_weights",
                "session": False,
                "discard_candidate_stdout": True,
            }) + "\n" + json.dumps(request) + "\n"
        ).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == b""
    assert completed.stderr.splitlines() == [
        b"IMPORTING",
        b"READY",
        b'{"value":[[1.0]]}',
    ]


def test_isolated_grade_uses_trusted_task_wall_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vare.separated_candidate_harness as harness

    observed: dict[str, float | None] = {}

    def grade(_workspace: Path, _calls_path: Path, *, image: str, task_timeout_s: float | None):
        observed["timeout"] = task_timeout_s
        return {
            "decision": {"success": True, "reason": "correctness"},
            "grade": {"cases": []},
        }

    monkeypatch.setattr(harness, "grade_isolated_subject", grade)
    subject = _subject(tmp_path, "def causal_attention_weights(scores):\n    return []\n")

    report = harness.grade_function(_request(subject))

    assert observed["timeout"] == 20
    assert report["task_id"] == TASK_ID


def test_task_descriptor_cannot_be_substituted(tmp_path: Path) -> None:
    forged = tmp_path / "task.json"
    forged.write_text(
        '{"id":"mlsys-v1-causal-attention","metadata":{"task_type":"correctness"}}\n',
        encoding="utf-8",
    )
    subject = _subject(tmp_path, "def causal_attention_weights(*args):\n    return []\n")
    payload = _request(subject)
    payload["trusted_task_path"] = str(forged)

    with pytest.raises(SeparatedHarnessError, match="supported MLSys pack"):
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
    payload["entrypoint"] = "causal_attention:other"
    with pytest.raises(SeparatedHarnessError, match="entrypoint"):
        grade_function(payload)


@pytest.mark.parametrize(
    "task_id,task_name",
    [
        ("mlsys-v1-grad-accum-tail", "grad_accum_tail"),
        ("mlsys-v1-global-grad-clip", "global_grad_clip"),
        ("mlsys-v1-range-sums", "range_sums"),
        ("mlsys-v1-dataset-standardize", "dataset_standardize"),
    ],
)
def test_mutation_sensitive_tasks_fail_closed(
    task_id: str, task_name: str,
) -> None:
    from vare.separated_candidate_harness import PACK_ROOTS, _task_adapter

    pack = "mlsys-v1-" if task_id != "mlsys-v1-range-sums" else "mlsys-v1-"
    task_path = PACK_ROOTS[pack] / "tasks" / task_name / "task.json"
    assert task_path.is_file()
    with pytest.raises(SeparatedHarnessError, match="no isolated call materializer"):
        _task_adapter(task_id, task_path)


@pytest.mark.parametrize(
    "task_id,pack_key,task_name",
    [
        ("mlsys-v1-aging-scheduler", "mlsys-v1-", "aging_scheduler"),
        ("mlsys-v1-paired-shuffle-loader", "mlsys-v1-", "paired_shuffle_loader"),
        ("mlsys-v1-row-log-softmax", "mlsys-v1-", "row_log_softmax"),
        ("mlsys-v3-label-smooth", "mlsys-v3-", "mlsys-v3-label-smooth"),
    ],
)
def test_type_or_non_json_sensitive_tasks_fail_closed(
    task_id: str, pack_key: str, task_name: str,
) -> None:
    from vare.separated_candidate_harness import PACK_ROOTS, _task_adapter

    task_path = PACK_ROOTS[pack_key] / "tasks" / task_name / "task.json"
    assert task_path.is_file()
    with pytest.raises(SeparatedHarnessError, match="no isolated call materializer"):
        _task_adapter(task_id, task_path)


def test_v3_copy_of_supported_warmup_task_routes_from_canonical_path() -> None:
    from vare.separated_candidate_harness import PACK_ROOTS, _task_adapter

    task_path = PACK_ROOTS["mlsys-v3-"] / "tasks/warmup_cosine/task.json"
    task_id = json.loads(task_path.read_text(encoding="utf-8"))["id"]
    task_dir, _module, _function, _kind = _task_adapter(task_id, task_path)
    assert task_dir == task_path.parent


def test_v3_warmup_oracle_runs_through_the_isolated_grader(tmp_path: Path) -> None:
    task_dir = ROOT / "benchmarks/mlsys_coding_v3/tasks/warmup_cosine"
    subject = tmp_path / "subject"
    subject.mkdir()
    shutil.copy(
        task_dir / "evaluator/oracle/warmup_cosine.py",
        subject / "warmup_cosine.py",
    )
    report = grade_function({
        "workspace": str(subject),
        "entrypoint": "warmup_cosine:learning_rate",
        "task_type": "correctness",
        "trusted_task_path": str(task_dir / "task.json"),
        "nonce": "ef" * 16,
    })
    assert report["passed"] is True
    assert report["nonce_enforced"] is True


def test_supported_correctness_calls_preserve_frozen_timeout() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from mlsys_task_isolated_calls import materialize_isolated_calls

    calls, _provenance = materialize_isolated_calls(
        TASK_ID, TASK_DIR,
    )
    assert calls["timeout_s"] == 10


@pytest.mark.parametrize(
    ("task_id", "module", "function"),
    [
        ("mlsys-v3-ema-bias", "ema_bias", "bias_corrected_ema"),
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
