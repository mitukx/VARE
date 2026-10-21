"""Candidate calls stay outside the grader process and outside /grader."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import pytest

from vare.container_isolation import DockerIsolationRunner, IsolationError, IsolationLimits
from vare.isolated_candidate import (
    CandidateHarnessError,
    FunctionCall,
    _Lines,
    TimingPair,
    TimingReport,
    _candidate_temporary_directory,
    grade_calls,
    overhead_adjusted,
    paired_difference_ci,
    paired_timing,
    student_t_975,
    two_x_speedup_decision,
    values_close,
)

IMAGE = "python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"


def test_macos_candidate_temp_path_is_canonical_and_home_shared(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with _candidate_temporary_directory("candidate-test-") as folder:
        resolved = Path(folder).resolve(strict=True)
        assert resolved == Path(folder)
        assert resolved.parent == tmp_path / ".cache" / "vare-isolated-candidate"
CANARY = "vare-hidden-canary-85"
OWNED = "grader-owned-token"
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
def grader_files(tmp_path_factory) -> dict[str, Path]:
    evaluator = tmp_path_factory.mktemp("separate-grader") / "evaluator"
    evaluator.mkdir()
    hidden = evaluator / "test_hidden.py"
    owned = evaluator / "owned.txt"
    hidden.write_text(f"CANARY = {CANARY!r}\n", encoding="utf-8")
    owned.write_text(OWNED, encoding="utf-8")
    return {"hidden": hidden, "owned": owned}


def test_student_t_matches_pack_constant() -> None:
    assert student_t_975(6) == 2.446911851144167
    mean, low, high, crit, df = paired_difference_ci([2.0, 2.0, 2.0])
    assert mean == 2.0
    assert low == 2.0
    assert high == 2.0
    assert crit == student_t_975(2)
    assert df == 2


def test_numeric_checker_can_match_frozen_bool_as_number_semantics() -> None:
    assert values_close(True, 1.0, rtol=0, atol=0, bool_as_number=True)
    assert not values_close(True, 1.0, rtol=0, atol=0)


def test_task_kind_comes_from_the_caller(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "add", "def add(a, b):\n    return a + b\n")
    call = FunctionCall("case", "add", "add", [2, 3], 5)
    with pytest.raises(CandidateHarnessError):
        grade_calls(subject, [call], task_kind="performance", image=IMAGE)
    with pytest.raises(CandidateHarnessError):
        paired_timing(
            subject, subject, module="add", function="add",
            make_case=lambda _index: ([2, 3], {}, 5), repeats=2, seed=1,
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


def test_import_and_call_receive_separate_timeout_windows(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path,
        "slow",
        "import time\ntime.sleep(0.25)\n"
        "def slow():\n    time.sleep(0.25)\n    return 7\n",
    )
    call = FunctionCall("slow-case", "slow", "slow", [], 7)

    report = _grade(subject, call, timeout_s=0.4)

    assert report.passed is True


def test_grade_session_keeps_candidate_module_state_across_cases(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "counter",
        "count = 0\ndef next_value():\n    global count\n    count += 1\n    return count\n",
    )
    calls = [
        FunctionCall("first", "counter", "next_value", [], 1),
        FunctionCall("second", "counter", "next_value", [], 2),
    ]

    report = grade_calls(
        subject, calls, task_kind="correctness", image=IMAGE,
        preserve_session=True, timeout_s=5,
    )

    assert report.passed is True


def test_grade_session_rejects_stateful_candidate_mismatch(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "counter",
        "count = 0\ndef next_value():\n    global count\n    count += 1\n    return count\n",
    )
    calls = [
        FunctionCall("first", "counter", "next_value", [], 1),
        FunctionCall("second", "counter", "next_value", [], 1),
    ]

    report = grade_calls(
        subject, calls, task_kind="correctness", image=IMAGE,
        preserve_session=True, timeout_s=5,
    )

    assert report.passed is False
    assert [case.passed for case in report.cases] == [True, False]


def test_expected_exception_prefix_matches_frozen_adapter_contract(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "custom_error",
        "class ValueErrorSentinel(Exception):\n    pass\n"
        "def fail():\n    raise ValueErrorSentinel()\n",
    )

    report = grade_calls(
        subject,
        [FunctionCall(
            "invalid", "custom_error", "fail", [], None,
            expected_exception="ValueError", expected_exception_prefix="ValueError",
        )],
        task_kind="correctness", image=IMAGE, timeout_s=5,
    )

    assert report.passed is True


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
    hidden_path = repr(str(grader_files["hidden"]))
    subject = _subject(
        tmp_path, "solve",
        "def attack():\n"
        "    try:\n"
        "        return open(" + hidden_path + ", encoding='utf-8').read()\n"
        "    except OSError:\n"
        "        return 'blocked'\n",
    )
    report = _grade(subject, FunctionCall("read", "solve", "attack", [], CANARY))
    assert report.passed is False
    assert CANARY not in json.dumps(report.to_dict())
    assert grader_files["hidden"].read_text(encoding="utf-8") == f"CANARY = {CANARY!r}\n"


def test_candidate_cannot_write_grader_paths(tmp_path: Path, grader_files: dict[str, Path]) -> None:
    before = grader_files["owned"].read_text(encoding="utf-8")
    owned_path = repr(str(grader_files["owned"]))
    subject = _subject(
        tmp_path, "solve",
        "def attack():\n"
        "    try:\n"
        "        handle = open(" + owned_path + ", 'w', encoding='utf-8')\n"
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

    expected = sum(range(1_000_000))

    def make_case(_index: int) -> tuple[list[int], dict[str, object], int]:
        return ([1_000_000], {}, expected)

    clock = paired_timing(
        patched, baseline, module="work", function="work", make_case=make_case,
        repeats=2, seed=20261019, task_kind="performance", image=IMAGE, timeout_s=30,
    )
    cached_timing = paired_timing(
        cached, baseline, module="work", function="work", make_case=make_case,
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


def test_eof_after_value_line_is_not_pending() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'{"value":5}\n')
    os.close(write_fd)
    reader = _Lines(os.fdopen(read_fd, "rb"))
    try:
        assert reader.readline(time.monotonic() + 2) == b'{"value":5}'
        assert reader.has_pending() is False
    finally:
        reader.stream.close()


def test_trailing_bytes_after_value_line_stay_pending() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'{"value":5}\n{"speedup":1000}\n')
    os.close(write_fd)
    reader = _Lines(os.fdopen(read_fd, "rb"))
    try:
        assert reader.readline(time.monotonic() + 2) == b'{"value":5}'
        assert reader.has_pending() is True
    finally:
        reader.stream.close()


def test_trailing_stdout_after_a_value_line_fails(tmp_path: Path) -> None:
    subject = _subject(
        tmp_path, "solve",
        "import sys\n"
        "def add(a, b):\n"
        "    sys.stdout.write('{\"value\": 5}\\ntrailing\\n')\n"
        "    sys.stdout.flush()\n"
        "    return a + b\n",
    )
    report = _grade(subject, FunctionCall("sum", "solve", "add", [2, 3], 5))
    assert report.passed is False
    assert report.cases[0].failure == "malformed_result"


def test_repeated_honest_calls_have_no_eof_flake(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "solve", "def add(a, b):\n    return a + b\n")
    calls = [
        FunctionCall(f"add-{index}", "solve", "add", [2, 3], 5)
        for index in range(40)
    ]
    report = grade_calls(
        subject, calls, task_kind="correctness", image=IMAGE, timeout_s=30,
    )
    assert report.passed is True
    assert [case.failure for case in report.cases] == [None] * 40


def test_import_precompute_does_not_pass_a_two_x_gate(tmp_path: Path) -> None:
    limit = 20_000_000
    burn = (
        "def _burn():\n"
        "    total = 0\n"
        "    for item in range(20_000_000):\n"
        "        total += item\n"
        "    return total\n"
    )
    candidate = _subject(
        tmp_path / "candidate", "work",
        burn + "TABLE = _burn()\ndef work(n):\n    return TABLE\n",
    )
    baseline = _subject(
        tmp_path / "baseline", "work",
        burn + "def work(n):\n    return _burn()\n",
    )
    expected = (limit - 1) * limit // 2
    report = paired_timing(
        candidate, baseline, module="work", function="work",
        make_case=lambda _index: ([0], {}, expected), repeats=2, seed=1,
        task_kind="performance", image=IMAGE, timeout_s=60,
    )
    assert report.output_pass is True
    ratios = [pair.ratio for pair in report.pairs]
    assert all(ratio is not None and ratio < 2 for ratio in ratios)
    assert all(
        pair.candidate_s is not None and pair.candidate_s > 0.5 for pair in report.pairs
    )
    assert all(
        pair.baseline_s is not None and pair.baseline_s > 0.5 for pair in report.pairs
    )
    assert report.mean_adjusted_ratio is not None
    assert report.ci95_low_adjusted_ratio is not None
    assert report.ci95_high_adjusted_ratio is not None
    assert report.mean_adjusted_difference_s == pytest.approx(report.mean_difference_s)
    for pair in report.pairs:
        assert set(pair.order.split("-")) == {"baseline", "candidate", "overhead"}
        assert pair.overhead_s is not None and 0 < pair.overhead_s < pair.candidate_s
        assert pair.adjusted_ratio is not None
        assert pair.adjusted_difference_s == pytest.approx(pair.difference_s)
    decision = two_x_speedup_decision(report)
    assert decision["success"] is False
    if report.workload_sufficient:
        assert decision["reason"] == "below_threshold"
    else:
        assert decision["reason"] == "workload_below_empty_control"


def test_paired_timing_rejects_equal_wrong_returns(tmp_path: Path) -> None:
    source = "def add(a, b):\n    return 0\n"
    candidate = _subject(tmp_path / "candidate", "addmod", source)
    baseline = _subject(tmp_path / "baseline", "addmod", source)
    report = paired_timing(
        candidate, baseline, module="addmod", function="add",
        make_case=lambda _index: ([2, 3], {}, 5), repeats=2, seed=1,
        task_kind="performance", image=IMAGE,
    )
    assert report.output_pass is False
    assert report.mean_difference_s is None
    assert report.ci95_low_difference_s is None
    assert report.ci95_high_difference_s is None
    assert [pair.failure for pair in report.pairs] == ["wrong_output", "wrong_output"]
    assert [pair.values_match for pair in report.pairs] == [False, False]


def test_overhead_adjustment_restores_a_call_ratio_hidden_by_startup() -> None:
    raw_baseline = 1.40
    raw_candidate = 1.20
    overhead = 1.0
    assert raw_baseline / raw_candidate < 2
    adjusted_baseline, adjusted_candidate, difference, ratio = overhead_adjusted(
        raw_baseline, raw_candidate, overhead,
    )
    assert adjusted_baseline == pytest.approx(0.40)
    assert adjusted_candidate == pytest.approx(0.20)
    assert difference == pytest.approx(raw_baseline - raw_candidate)
    assert ratio == pytest.approx(2.0)
    mean, low, high, _crit, df = paired_difference_ci([2.0, 2.0])
    assert mean == 2.0
    assert low == 2.0
    assert high == 2.0
    assert df == 1


def test_hardcoded_answer_fails_when_each_repetition_differs(tmp_path: Path) -> None:
    rng = random.Random(20261020)
    drawn: list[int] = []

    def make_case(_index: int) -> tuple[list[int], dict[str, object], int]:
        value = rng.randrange(1_000_000, 9_000_000)
        drawn.append(value)
        return ([value], {}, value * 3 + 1)

    candidate = _subject(
        tmp_path / "candidate", "work",
        "ANS = {2000000: 6000001}\n"
        "def work(n):\n"
        "    return ANS.get(n, 0)\n",
    )
    baseline = _subject(
        tmp_path / "baseline", "work",
        "def work(n):\n"
        "    return n * 3 + 1\n",
    )
    report = paired_timing(
        candidate, baseline, module="work", function="work", make_case=make_case,
        repeats=2, seed=7, task_kind="performance", image=IMAGE,
    )
    assert len(drawn) == 2
    assert len(set(drawn)) == 2
    assert 2000000 not in drawn
    assert report.output_pass is False
    assert report.mean_difference_s is None
    assert report.mean_adjusted_ratio is None
    assert [pair.failure for pair in report.pairs] == ["wrong_output", "wrong_output"]
    assert [pair.values_match for pair in report.pairs] == [False, False]


def test_float_tolerance_is_per_task_and_exact_by_default() -> None:
    noisy = 0.1 + 0.2
    assert values_close([noisy], [0.3], rtol=1e-6, atol=0.0) is True
    assert values_close([noisy], [0.3], rtol=0.0, atol=0.0) is False
    assert values_close(True, 1, rtol=1e-6, atol=1e-6) is False
    assert values_close({"row": [noisy, 1]}, {"row": [0.3, 1]}, rtol=1e-5, atol=1e-8) is True


def test_grade_calls_applies_task_float_tolerance(tmp_path: Path) -> None:
    subject = _subject(tmp_path, "solve", "def score():\n    return 0.1 + 0.2\n")
    exact = grade_calls(
        subject, [FunctionCall("near", "solve", "score", [], 0.3)],
        task_kind="correctness", image=IMAGE, rtol=0.0, atol=0.0,
    )
    close = grade_calls(
        subject, [FunctionCall("near", "solve", "score", [], 0.3)],
        task_kind="correctness", image=IMAGE, rtol=1e-6, atol=0.0,
    )
    assert exact.passed is False
    assert exact.cases[0].failure == "wrong_output"
    assert close.passed is True


def _timing_pair(baseline: float, candidate: float, overhead: float) -> TimingPair:
    return TimingPair(
        "baseline-candidate-overhead", baseline, candidate, baseline - candidate,
        baseline / candidate, True, None, overhead,
    )


def test_two_x_refuses_when_startup_dominates() -> None:
    report = TimingReport(
        "performance", True, 2, 1,
        (_timing_pair(1.0, 0.2, 0.6), _timing_pair(1.0, 0.2, 0.6)),
        0.8, 0.8, 0.8, student_t_975(1), 1,
        workload_sufficient=True, oracle_subject_ratio=5.0,
    )
    decision = two_x_speedup_decision(report)
    assert decision["success"] is False
    assert decision["workload_sufficient"] is False
    assert decision["reason"] == "workload_below_empty_control"
    assert decision["oracle_subject_ratio"] == pytest.approx(5.0)


def test_two_x_accepts_only_a_sufficient_workload() -> None:
    report = TimingReport(
        "performance", True, 2, 1,
        (_timing_pair(4.0, 1.5, 0.4), _timing_pair(4.0, 1.5, 0.4)),
        None, None, None, None, None,
    )
    decision = two_x_speedup_decision(report)
    assert decision["success"] is True
    assert decision["workload_sufficient"] is True
    assert decision["reason"] == "performance"
    assert decision["oracle_subject_ratio"] == pytest.approx(4.0 / 1.5)
