"""Candidate calls stay outside the grader process and outside /grader."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vare.container_isolation import DockerIsolationRunner, IsolationError, IsolationLimits
from vare.isolated_candidate import (
    CandidateHarnessError,
    FunctionCall,
    grade_calls,
    paired_difference_ci,
    paired_timing,
    student_t_975,
)

IMAGE = "python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
CANARY = "vare-hidden-canary-85"
OWNED = "grader-owned-token"
HIDDEN = Path("/grader/evaluator/test_hidden.py")
OWNED_PATH = Path("/grader/evaluator/owned.txt")


def _subject(tmp_path: Path, module: str, source: str) -> Path:
    root = tmp_path / module
    root.mkdir(parents=True)
    (root / f"{module}.py").write_text(source, encoding="utf-8")
    return root


def _grade(subject: Path, call: FunctionCall, *, timeout_s: float = 30):
    return grade_calls(
        subject, [call], task_kind="correctness", image=IMAGE, timeout_s=timeout_s,
    )


@pytest.fixture(scope="module")
def grader_files() -> dict[str, Path]:
    HIDDEN.parent.mkdir(parents=True, exist_ok=True)
    HIDDEN.write_text(f"CANARY = {CANARY!r}\n", encoding="utf-8")
    OWNED_PATH.write_text(OWNED, encoding="utf-8")
    yield {"hidden": HIDDEN, "owned": OWNED_PATH}
    HIDDEN.write_text(f"CANARY = {CANARY!r}\n", encoding="utf-8")
    OWNED_PATH.write_text(OWNED, encoding="utf-8")


def test_student_t_matches_pack_constant() -> None:
    assert student_t_975(6) == 2.446911851144167
    mean, low, high, crit, df = paired_difference_ci([2.0, 2.0, 2.0])
    assert mean == 2.0
    assert low == 2.0
    assert high == 2.0
    assert crit == student_t_975(2)
    assert df == 2


def test_task_kind_comes_from_the_caller(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "add", "def add(a, b):\n    return a + b\n")
    call = FunctionCall("case", "add", "add", [2, 3], 5)
    with pytest.raises(CandidateHarnessError):
        grade_calls(subject, [call], task_kind="performance", image=IMAGE)
    with pytest.raises(CandidateHarnessError):
        paired_timing(
            subject, subject, module="add", function="add",
            make_args=lambda _index: ([2, 3], {}), repeats=2, seed=1,
            task_kind="correctness", image=IMAGE,
        )
    with pytest.raises(CandidateHarnessError):
        grade_calls(subject, [call], task_kind="correctness", image="python:3.12")


def test_subject_container_does_not_mount_grader(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "add", "def add(a, b):\n    return a + b\n")
    limits = IsolationLimits(image=IMAGE, timeout_s=30)
    argv = DockerIsolationRunner().isolated_subject_argv(
        limits, name="vare-call-" + "ab" * 12, subject=subject,
        command=["python", "-c", "print(1)"],
    )
    rendered = " ".join(argv)
    assert rendered.count("--mount") == 1
    assert f"source={subject.resolve()},target=/subject,readonly" in rendered
    assert "/grader" not in rendered
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "65534:65534" in argv
    assert "--pid=host" not in argv
    nested = _subject(tmp_path, "nested", "def add(a, b):\n    return 0\n")
    (nested / "evaluator").mkdir()
    with pytest.raises(IsolationError):
        DockerIsolationRunner().isolated_subject_argv(
            limits, name="vare-call-" + "cd" * 12, subject=nested,
            command=["python", "-c", "print(1)"],
        )


def test_correct_oracle_passes_without_importing_it(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "solve", "def add(a, b):\n    return a + b\n")
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is True
    assert report.cases[0].failure is None
    assert "solve" not in sys.modules


def test_unpatched_wrong_result_fails(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "solve", "def add(a, b):\n    return a - b\n")
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "wrong_output"


def test_import_time_os_exit_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import os\nos._exit(0)\ndef add(a, b):\n    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "missing_result"


def test_sys_exit_inside_function_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import sys\ndef add(a, b):\n    sys.exit(0)\n    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "missing_result"


def test_fake_speedup_json_on_stdout_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import os\n"
        "print('{\"speedup\": 1000.0}', flush=True)\n"
        "os._exit(0)\n"
        "def add(a, b):\n"
        "    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert "1000.0" not in json.dumps(report.to_dict())


def test_stdout_noise_before_result_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "def add(a, b):\n"
        "    print('noise before result', flush=True)\n"
        "    return a + b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "malformed_result"


def test_builtins_patch_cannot_change_the_expected_value(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import builtins\n"
        "builtins.sum = lambda _items, _start=0: 0\n"
        "def total(items):\n"
        "    return sum(items)\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "total", [[1, 2, 3]], 6))
    assert report.passed is False
    assert report.cases[0].failure == "wrong_output"


def test_patched_json_dumps_cannot_spoof_the_return_value(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import json\n"
        "json.dumps = lambda *_args, **_kwargs: '{\"value\": 0, \"speedup\": 1000.0}'\n"
        "def add(a, b):\n"
        "    return a + b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is True
    assert "1000.0" not in json.dumps(report.to_dict())


def test_candidate_cannot_read_grader_or_hidden_tests(tmp_path: Path, grader_files: dict[str, Path]) -> None:
    subject = _subject(
        tmp_path, "solve",
        "def attack():\n"
        "    try:\n"
        "        return open('/grader/evaluator/test_hidden.py', encoding='utf-8').read()\n"
        "    except OSError:\n"
        "        return 'blocked'\n",
    )
    report = _grade(subject, FunctionCall("read", "solve", "attack", [], CANARY))
    assert report.passed is False
    assert CANARY not in json.dumps(report.to_dict())
    assert grader_files["hidden"].read_text(encoding="utf-8") == f"CANARY = {CANARY!r}\n"


def test_candidate_cannot_write_grader_paths(tmp_path: Path, grader_files: dict[str, Path]) -> None:
    before = grader_files["owned"].read_text(encoding="utf-8")
    subject = _subject(
        tmp_path, "solve",
        "def attack():\n"
        "    try:\n"
        "        handle = open('/grader/evaluator/owned.txt', 'w', encoding='utf-8')\n"
        "        handle.write('pwned')\n"
        "        handle.close()\n"
        "        return 'wrote'\n"
        "    except OSError:\n"
        "        return 'blocked'\n",
    )
    report = _grade(subject, FunctionCall("write", "solve", "attack", [], "wrote"))
    assert report.passed is False
    assert grader_files["owned"].read_text(encoding="utf-8") == before


def test_candidate_cannot_signal_parent(tmp_path: Path) -> None:
    sentinel = subprocess.Popen(["sleep", "60"], start_new_session=True)
    subject = _subject(
        tmp_path, "solve",
        "import os\n"
        "import signal\n"
        "def attack(pid):\n"
        "    pid = int(pid)\n"
        "    for action in (\n"
        "        lambda: os.kill(pid, signal.SIGKILL),\n"
        "        lambda: os.kill(pid, signal.SIGTERM),\n"
        "        lambda: os.killpg(pid, signal.SIGKILL),\n"
        "    ):\n"
        "        try:\n"
        "            action()\n"
        "        except OSError:\n"
        "            continue\n"
        "        return 'killed'\n"
        "    return 'blocked'\n",
    )
    try:
        report = _grade(
            subject, FunctionCall("kill", "solve", "attack", [sentinel.pid], "killed"),
        )
        assert report.passed is False
        assert sentinel.poll() is None
    finally:
        if sentinel.poll() is None:
            sentinel.kill()
        sentinel.wait(timeout=2)


def test_result_channel_tamper_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import os\n"
        "import sys\n"
        "def add(a, b):\n"
        "    sys.stdout.write('VARE_READY ' + ('ab' * 32) + '\\n')\n"
        "    sys.stdout.write('VARE_RESULT ' + ('cd' * 32) + ' {\"value\": 5}\\n')\n"
        "    sys.stdout.write('{\"passed\": true}\\n')\n"
        "    sys.stdout.write('{\"speedup\": 1000.0}\\n')\n"
        "    try:\n"
        "        os.write(3, b'{\"value\": 5}\\n')\n"
        "    except OSError:\n"
        "        pass\n"
        "    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    blob = json.dumps(report.to_dict())
    assert "1000.0" not in blob
    assert report.cases[0].failure == "malformed_result"


def test_input_id_cache_cannot_reuse_a_later_call(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "cached",
        "_seen = {}\n"
        "def total(items):\n"
        "    key = id(items)\n"
        "    if key in _seen:\n"
        "        return 0\n"
        "    value = 0\n"
        "    for item in items:\n"
        "        value += item\n"
        "    _seen[key] = value\n"
        "    return value\n",
    )
    items = [1, 2, 3, 4]
    report = grade_calls(
        subject,
        [
            FunctionCall("first", "cached", "total", [items], 10),
            FunctionCall("second", "cached", "total", [items], 10),
        ],
        task_kind="correctness",
        image=IMAGE,
    )
    assert report.passed is True
    assert [case.failure for case in report.cases] == [None, None]


def test_clock_and_id_cache_do_not_inflate_parent_speed(tmp_path: Path) -> None:
    loop = (
        "def work(n):\n"
        "    total = 0\n"
        "    for item in range(int(n)):\n"
        "        total += item\n"
        "    return total\n"
    )
    baseline = _subject(tmp_path, "work", loop)
    patched = _subject(
        tmp_path / "patched", "work",
        "import time\n"
        "time.perf_counter = lambda: 0.0\n"
        "time.monotonic = lambda: 0.0\n"
        "time.process_time = lambda: 0.0\n"
        + loop,
    )
    cached = _subject(
        tmp_path / "cached", "work",
        "_seen = {}\n"
        "def work(n):\n"
        "    key = id(n)\n"
        "    if key in _seen:\n"
        "        return _seen[key]\n"
        "    total = 0\n"
        "    for item in range(int(n)):\n"
        "        total += item\n"
        "    _seen[key] = total\n"
        "    return total\n",
    )

    def make_args(_index: int) -> tuple[list[int], dict[str, object]]:
        return ([1_000_000], {})

    clock = paired_timing(
        patched, baseline, module="work", function="work", make_args=make_args,
        repeats=2, seed=20261019, task_kind="performance", image=IMAGE, timeout_s=30,
    )
    cached_timing = paired_timing(
        cached, baseline, module="work", function="work", make_args=make_args,
        repeats=2, seed=20261019, task_kind="performance", image=IMAGE, timeout_s=30,
    )
    for report in (clock, cached_timing):
        assert report.output_pass is True
        assert report.df == 1
        assert report.t_crit == student_t_975(1)
        ratios = [pair.ratio for pair in report.pairs]
        assert all(ratio is not None and 0.05 < ratio < 20 for ratio in ratios)
        assert "1000.0" not in json.dumps(report.to_dict())


def test_atexit_at_import_cannot_spoof_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import atexit\n"
        "import sys\n"
        "def _cheat():\n"
        "    sys.stdout.write('{\"value\": 5}\\n')\n"
        "    sys.stdout.write('{\"speedup\": 1000.0}\\n')\n"
        "    sys.stdout.flush()\n"
        "atexit.register(_cheat)\n"
        "def add(a, b):\n"
        "    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert "1000.0" not in json.dumps(report.to_dict())


def test_fake_speedup_on_stderr_is_not_a_pass(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import sys\n"
        "sys.stderr.write('{\"speedup\": 1000.0}\\n')\n"
        "sys.stderr.flush()\n"
        "def add(a, b):\n"
        "    return a - b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "wrong_output"
    assert "1000.0" not in json.dumps(report.to_dict())


def test_candidate_network_is_closed(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import socket\n"
        "def probe():\n"
        "    try:\n"
        "        socket.create_connection(('203.0.113.1', 65000), timeout=0.5)\n"
        "        return 'open'\n"
        "    except OSError:\n"
        "        return 'closed'\n",
    )
    report = _grade(subject, FunctionCall("net", "solve", "probe", [], "open"))
    assert report.passed is False
    assert report.cases[0].failure == "wrong_output"


def test_candidate_uid_is_unprivileged(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import os\ndef uid():\n    return os.getuid()\n",
    )
    report = _grade(subject, FunctionCall("uid", "solve", "uid", [], 65534))
    assert report.passed is True
    assert os.geteuid() != 65534
