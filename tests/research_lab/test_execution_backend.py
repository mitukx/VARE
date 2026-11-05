"""Runner limits, recovery, and fail-closed boundary tests.

These tests check the execution harness. They do not measure a model.
"""
from __future__ import annotations

from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

from vare.research_lab import (
    CommandStep,
    ExperimentRunner,
    ExperimentSpec,
    MetricContract,
    RunLimits,
    load_spec,
)
from vare.research_lab.headroom import decide_accelerator_headroom
from vare.research_lab.jsonutil import strict_load
from vare.research_lab.store import DuplicateCommitError, ResumeStore
from vare.research_lab.spec import SpecError


ROOT = Path(__file__).resolve().parents[2]
IMAGE = "example@sha256:" + ("0" * 64)


def _limits(**kwargs) -> RunLimits:
    values = dict(wall_time_s=5, cpu_time_s=30, memory_mb=256, max_output_bytes=100_000, max_file_bytes=1_000_000)
    values.update(kwargs)
    return RunLimits(**values)


def _spec(tmp: Path, argv: tuple[str, ...], **kwargs) -> ExperimentSpec:
    run_id = kwargs.get("run_id", "run1")
    workspace = tmp / run_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "model.py"
    if not source.exists():
        source.write_text("value = 1\n", encoding="utf-8")
    protected_dir = tmp / "evaluator"
    protected_dir.mkdir(exist_ok=True)
    protected = protected_dir / "hidden.py"
    if not protected.exists():
        protected.write_text("SECRET = 1\n", encoding="utf-8")
    contract = kwargs.pop("metric_contract", MetricContract(relative_path="metrics.json", required_keys=("loss",), maxima={"loss": 1.0}))
    return ExperimentSpec(
        run_id=kwargs.pop("run_id", "run1"),
        task_id=kwargs.pop("task_id", "task1"),
        workspace=workspace,
        steps=kwargs.pop("steps", (CommandStep("main", argv),)),
        limits=kwargs.pop("limits", _limits()),
        source_files=kwargs.pop("source_files", ("model.py",)),
        protected_paths=kwargs.pop("protected_paths", (protected,)),
        metric_contract=contract,
        config=kwargs.pop("config", {"seed": 1}),
        **kwargs,
    )


def _runner(tmp: Path, spec: ExperimentSpec) -> ExperimentRunner:
    runner = ExperimentRunner(tmp / "state.sqlite", tmp / "runs")
    runner.prepare(spec)
    return runner


def _write_metrics(code: str) -> str:
    return (
        "import json,pathlib; pathlib.Path('metrics.json').write_text(json.dumps({'loss': 0.2})); "
        + code
    )


class ContractTests(unittest.TestCase):
    def test_timeout_is_not_incorrect_unless_the_contract_says_so(self) -> None:
        from vare.research_lab.classify import judge

        plain = judge(step_failure="wall_timeout", metrics_status="ok", protected_ok=True, source_ok=True,
                      timeout_counts_as_incorrect=False)
        marked = judge(step_failure="wall_timeout", metrics_status="ok", protected_ok=True, source_ok=True,
                       timeout_counts_as_incorrect=True)
        self.assertFalse(plain.accepted)
        self.assertFalse(plain.candidate_incorrect)
        self.assertEqual("wall_timeout", plain.failure_class)
        self.assertFalse(marked.accepted)
        self.assertTrue(marked.candidate_incorrect)

    def test_self_report_does_not_override_a_missing_metric(self) -> None:
        from vare.research_lab.classify import read_metrics

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "metrics.json").write_text('{"passed": true, "accepted": true}', encoding="utf-8")
            read = read_metrics(root, MetricContract("metrics.json", ("loss",)))
        self.assertEqual("forged_success", read.status)

    def test_metric_path_cannot_escape(self) -> None:
        with self.assertRaises(SpecError) as raised:
            MetricContract("../metrics.json", ("loss",))
        self.assertEqual("path_traversal", raised.exception.failure_class)

    def test_headroom_refuses_low_memory_and_competing_jobs_without_signaling(self) -> None:
        low = decide_accelerator_headroom(available_bytes=100, min_available_bytes=1024, cmdlines=[])
        busy = decide_accelerator_headroom(
            available_bytes=10_000, min_available_bytes=1024, cmdlines=["python grpo_mlx.py"],
        )
        clear = decide_accelerator_headroom(available_bytes=10_000, min_available_bytes=1024, cmdlines=["python other.py"])
        self.assertFalse(low.allowed)
        self.assertEqual("insufficient_memory", low.reason)
        self.assertFalse(busy.allowed)
        self.assertEqual("competing_accelerator_job", busy.reason)
        self.assertTrue(clear.allowed)

    def test_unpinned_docker_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(SpecError):
                _spec(Path(name), ("python", "-c", "print(1)"), isolation="docker", docker_image="ubuntu:latest",
                      network_policy="deny")


class PatchTests(unittest.TestCase):
    def test_invalid_patch_and_traversal_leave_the_workspace_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", "print(1)"))
            runner = _runner(tmp, spec)
            before = (spec.workspace / "model.py").read_bytes()
            rejected = runner.apply_patch(spec.run_id, "this is not a diff\n")
            self.assertFalse(rejected.applied)
            self.assertEqual("invalid_patch", rejected.failure_class)
            escaped = runner.apply_patch(spec.run_id, textwrap.dedent("""\
                diff --git a/../secret b/../secret
                --- a/../secret
                +++ b/../secret
                @@ -1 +1 @@
                -x
                +y
                """))
            self.assertFalse(escaped.applied)
            self.assertEqual("path_traversal", escaped.failure_class)
            link = spec.workspace / "linked.py"
            link.symlink_to(spec.protected_paths[0])
            linked = runner.apply_patch(spec.run_id, textwrap.dedent("""\
                diff --git a/linked.py b/linked.py
                --- a/linked.py
                +++ b/linked.py
                @@ -1 +1 @@
                -SECRET = 1
                +SECRET = 2
                """))
            self.assertFalse(linked.applied)
            self.assertEqual("symlink_escape", linked.failure_class)
            self.assertEqual(before, (spec.workspace / "model.py").read_bytes())
            self.assertEqual("SECRET = 1\n", spec.protected_paths[0].read_text(encoding="utf-8"))

    def test_matching_patch_updates_only_the_enrolled_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", "print(1)"))
            runner = _runner(tmp, spec)
            applied = runner.apply_patch(spec.run_id, textwrap.dedent("""\
                diff --git a/model.py b/model.py
                --- a/model.py
                +++ b/model.py
                @@ -1 +1 @@
                -value = 1
                +value = 2
                """))
            self.assertTrue(applied.applied)
            self.assertEqual("value = 2\n", (spec.workspace / "model.py").read_text(encoding="utf-8"))
            mismatched = runner.apply_patch(spec.run_id, textwrap.dedent("""\
                diff --git a/model.py b/model.py
                --- a/model.py
                +++ b/model.py
                @@ -1 +1 @@
                -value = 1
                +value = 3
                """))
            self.assertFalse(mismatched.applied)
            self.assertEqual("value = 2\n", (spec.workspace / "model.py").read_text(encoding="utf-8"))


class RunnerTests(unittest.TestCase):
    def test_success_records_a_manifest_and_does_not_import_workspace_code(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", _write_metrics("print('ok')")))
            (spec.workspace / "evil.py").write_text("open('imported','w').write('x')\n", encoding="utf-8")
            view = _runner(tmp, spec).execute(spec.run_id)
            self.assertTrue(view.accepted)
            self.assertEqual("completed", view.failure_class)
            self.assertFalse(view.duplicate)
            self.assertFalse((spec.workspace / "imported").exists())
            manifest = strict_load((tmp / "runs" / spec.run_id / "manifest.json").read_bytes())
            self.assertEqual(view.to_dict()["spec_sha256"], manifest["spec_sha256"])
            self.assertTrue((tmp / "runs" / "events.jsonl").is_file())
            again = ExperimentRunner(tmp / "state.sqlite", tmp / "runs").execute(spec.run_id)
            self.assertTrue(again.duplicate)
            self.assertTrue(again.accepted)

    def test_wall_timeout_kills_the_group_and_is_not_incorrectness(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(
                tmp,
                ("python", "-c", "while True:\n pass"),
                limits=_limits(wall_time_s=0.6, cpu_time_s=None, memory_mb=256),
            )
            started = time.monotonic()
            view = _runner(tmp, spec).execute(spec.run_id)
            self.assertLess(time.monotonic() - started, 5)
            self.assertFalse(view.accepted)
            self.assertFalse(view.candidate_incorrect)
            self.assertEqual("wall_timeout", view.failure_class)

    def test_timeout_contract_marks_incorrect_without_accepting(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(
                tmp,
                ("python", "-c", "while True:\n pass"),
                limits=_limits(wall_time_s=0.6, cpu_time_s=None),
                metric_contract=MetricContract("metrics.json", ("loss",), timeout_counts_as_incorrect=True),
            )
            view = _runner(tmp, spec).execute(spec.run_id)
            self.assertFalse(view.accepted)
            self.assertTrue(view.candidate_incorrect)
            self.assertEqual("wall_timeout", view.failure_class)

    def test_cpu_memory_and_output_limits_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            cpu = _spec(
                tmp,
                ("python", "-c", "while True:\n pass"),
                run_id="cpu",
                limits=_limits(wall_time_s=8, cpu_time_s=1, memory_mb=512),
            )
            cpu_view = _runner(tmp, cpu).execute(cpu.run_id)
            self.assertEqual("cpu_limit", cpu_view.failure_class)
            self.assertFalse(cpu_view.accepted)
            memory = _spec(
                tmp,
                ("python", "-c", "x=bytearray(256*1024*1024); import time; time.sleep(2)"),
                run_id="mem",
                limits=_limits(wall_time_s=5, cpu_time_s=None, memory_mb=96),
            )
            memory_view = _runner(tmp, memory).execute(memory.run_id)
            self.assertEqual("memory_limit", memory_view.failure_class)
            self.assertFalse(memory_view.accepted)
            flood = _spec(
                tmp,
                ("python", "-c", "print('x'*100000)"),
                run_id="out",
                limits=_limits(max_output_bytes=128, cpu_time_s=None),
            )
            flood_view = _runner(tmp, flood).execute(flood.run_id)
            self.assertEqual("output_limit", flood_view.failure_class)
            self.assertFalse(flood_view.accepted)

    def test_cancel_preserves_no_acceptance_and_stops_the_child(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", "import time; time.sleep(30)"), limits=_limits(wall_time_s=20, cpu_time_s=None))
            runner = _runner(tmp, spec)
            holder: dict[str, object] = {}

            def _run() -> None:
                holder["view"] = runner.execute(spec.run_id)

            thread = threading.Thread(target=_run)
            thread.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                row = runner.store.get_run(spec.run_id)
                if row and row["child_pid"] and row["state"] == "running":
                    break
                time.sleep(0.02)
            cancelled = runner.cancel(spec.run_id)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual("cancelled", cancelled.failure_class)
            self.assertFalse(cancelled.accepted)
            resumed = runner.resume(spec.run_id)
            self.assertTrue(resumed.duplicate)
            self.assertEqual("cancelled", resumed.failure_class)

    def test_resume_after_kill_does_not_duplicate_a_committed_step(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            first = "from pathlib import Path; Path('counter').open('a').write('A')"
            second = (
                "from pathlib import Path; import json,time; "
                "Path('counter').open('a').write('B'); "
                "Path('metrics.json').write_text(json.dumps({'loss': 0.2})); "
                "p=Path('hold'); \n"
                "import time\n"
                "for _ in range(200):\n"
                "    if not p.exists():\n"
                "        break\n"
                "    time.sleep(0.05)\n"
            )
            spec = _spec(
                tmp,
                ("python", "-c", first),
                steps=(
                    CommandStep("first", ("python", "-c", first)),
                    CommandStep("second", ("python", "-c", second)),
                ),
                limits=_limits(wall_time_s=20, cpu_time_s=None),
            )
            runner = _runner(tmp, spec)
            workspace = spec.workspace
            (workspace / "hold").write_text("1", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys; from vare.research_lab import ExperimentRunner; "
                 "ExperimentRunner(sys.argv[1], sys.argv[2]).execute(sys.argv[3])",
                 str(tmp / "state.sqlite"), str(tmp / "runs"), spec.run_id],
                env=env,
            )
            deadline = time.monotonic() + 8
            child_pid = None
            while time.monotonic() < deadline:
                steps = runner.store.get_steps(spec.run_id)
                row = runner.store.get_run(spec.run_id)
                if (len(steps) == 2 and steps[0]["state"] == "committed" and steps[1]["state"] == "running"
                        and row and row["child_pid"]):
                    child_pid = int(row["child_pid"])
                    break
                time.sleep(0.05)
            self.assertIsNotNone(child_pid)
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
            gone = time.monotonic() + 3
            while time.monotonic() < gone and Path(f"/proc/{child_pid}").exists():
                time.sleep(0.05)
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
            mid = runner.status(spec.run_id)
            self.assertFalse(mid.accepted)
            self.assertFalse(mid.to_dict()["owner_alive"])
            self.assertEqual("committed", mid.to_dict()["steps"][0]["state"])
            first_hash = mid.to_dict()["steps"][0]["record_sha256"]
            (workspace / "hold").unlink()
            resumed = runner.resume(spec.run_id)
            counter = (workspace / "counter").read_text(encoding="utf-8")
            self.assertEqual(1, counter.count("A"))
            self.assertIn("B", counter)
            self.assertEqual(first_hash, runner.status(spec.run_id).to_dict()["steps"][0]["record_sha256"])
            self.assertTrue(resumed.accepted)
            duplicate = runner.execute(spec.run_id)
            self.assertTrue(duplicate.duplicate)
            self.assertEqual(counter, (workspace / "counter").read_text(encoding="utf-8"))
            with self.assertRaises(DuplicateCommitError):
                runner.store.finalize(
                    spec.run_id, state="failed_closed", failure_class="internal_error", accepted=False,
                    candidate_incorrect=False, metrics=None, notes=["other"], timeout_counts_as_incorrect=False,
                )

    def test_stale_source_and_protected_writes_and_bad_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            stale = _spec(tmp, ("python", "-c", "open('ran','w').write('x')"), run_id="stale")
            runner = _runner(tmp, stale)
            (stale.workspace / "model.py").write_text("value = 9\n", encoding="utf-8")
            view = runner.execute(stale.run_id)
            self.assertEqual("stale_source", view.failure_class)
            self.assertFalse((stale.workspace / "ran").exists())
            protected = _spec(
                tmp,
                ("python", "-c", f"open({str(tmp / 'evaluator' / 'hidden.py')!r}, 'w').write('hacked')"),
                run_id="protected",
            )
            protected_view = _runner(tmp, protected).execute(protected.run_id)
            self.assertEqual("protected_path_violation", protected_view.failure_class)
            self.assertFalse(protected_view.accepted)
            forged = _spec(
                tmp,
                ("python", "-c", "open('metrics.json','w').write('{\"passed\": true, \"accepted\": true}')"),
                run_id="forged",
            )
            forged_view = _runner(tmp, forged).execute(forged.run_id)
            self.assertEqual("forged_success", forged_view.failure_class)
            self.assertFalse(forged_view.candidate_incorrect)
            corrupt = _spec(tmp, ("python", "-c", "open('metrics.json','w').write('{')"), run_id="corrupt")
            self.assertEqual("corrupt_metrics", _runner(tmp, corrupt).execute(corrupt.run_id).failure_class)
            missing = _spec(tmp, ("python", "-c", "print(1)"), run_id="missing")
            self.assertEqual("missing_metrics", _runner(tmp, missing).execute(missing.run_id).failure_class)
            rejected = _spec(
                tmp,
                ("python", "-c", "open('metrics.json','w').write('{\"loss\": 5}')"),
                run_id="rejected",
            )
            rejected_view = _runner(tmp, rejected).execute(rejected.run_id)
            self.assertEqual("metric_rejected", rejected_view.failure_class)
            self.assertTrue(rejected_view.candidate_incorrect)
            self.assertFalse(rejected_view.accepted)

    def test_symlink_metrics_blocked_program_and_network_and_fork_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            outside = tmp / "outside.json"
            outside.write_text('{"loss": 0.1}', encoding="utf-8")
            spec = _spec(tmp, ("python", "-c", "print(1)"), run_id="link")
            (spec.workspace / "metrics.json").symlink_to(outside)
            view = _runner(tmp, spec).execute(spec.run_id)
            self.assertEqual("symlink_escape", view.failure_class)
            self.assertFalse(view.accepted)
            blocked = _spec(tmp, ("bash", "-c", "echo hi"), run_id="bash")
            blocked_view = _runner(tmp, blocked).execute(blocked.run_id)
            self.assertEqual("executable_not_allowed", blocked_view.failure_class)
            self.assertFalse((blocked.workspace / "ran-bash").exists())
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(0.2)
            port = server.getsockname()[1]
            network = _spec(
                tmp,
                ("python", "-c",
                 "import socket,sys; s=socket.socket(); s.settimeout(0.5)\n"
                 f"s.connect(('127.0.0.1', {port}))\n"
                 "sys.exit(0)\n"),
                run_id="net",
                limits=_limits(cpu_time_s=None),
            )
            network_view = _runner(tmp, network).execute(network.run_id)
            accepted_connection = False
            try:
                conn, _addr = server.accept()
                accepted_connection = True
                conn.close()
            except OSError:
                accepted_connection = False
            server.close()
            self.assertFalse(accepted_connection)
            self.assertFalse(network_view.accepted)
            fork = _spec(
                tmp,
                ("python", "-c",
                 "import subprocess,sys\n"
                 "try:\n"
                 "    subprocess.run([sys.executable, '-c', 'print(1)'], check=True)\n"
                 "except Exception:\n"
                 "    sys.exit(3)\n"
                 "sys.exit(0)\n"),
                run_id="fork",
                limits=_limits(max_child_processes=0, cpu_time_s=None),
            )
            fork_view = _runner(tmp, fork).execute(fork.run_id)
            self.assertFalse(fork_view.accepted)
            self.assertNotEqual("completed", fork_view.failure_class)

    def test_manifest_and_log_tamper_fail_closed_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", _write_metrics("open('counter','a').write('A')")))
            runner = _runner(tmp, spec)
            self.assertTrue(runner.execute(spec.run_id).accepted)
            manifest_path = tmp / "runs" / spec.run_id / "manifest.json"
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace("true", "false", 1), encoding="utf-8")
            tampered = runner.status(spec.run_id)
            self.assertFalse(tampered.accepted)
            self.assertEqual("manifest_tamper", tampered.failure_class)
            again = runner.resume(spec.run_id)
            self.assertFalse(again.accepted)
            self.assertEqual("A", (spec.workspace / "counter").read_text(encoding="utf-8"))
            log = tmp / "runs" / "events.jsonl"
            log.write_text("{\"tamper\": true}\n", encoding="utf-8")
            logged = ExperimentRunner(tmp / "state.sqlite", tmp / "runs").status(spec.run_id)
            self.assertFalse(logged.accepted)
            self.assertEqual("log_tamper", logged.failure_class)

    def test_mlx_gate_does_not_start_a_step(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", "open('ran','w').write('x')"), backend="mlx", run_id="mlx-low")
            runner = ExperimentRunner(
                tmp / "state.sqlite", tmp / "runs",
                memory_probe=lambda: 10, process_probe=lambda: [],
            )
            runner.prepare(spec)
            refused = runner.execute(spec.run_id)
            self.assertEqual("mlx_refused_headroom", refused.failure_class)
            self.assertFalse((spec.workspace / "ran").exists())
            competing = _spec(tmp, ("python", "-c", "open('ran','w').write('x')"), backend="mlx", run_id="mlx-busy")
            busy = ExperimentRunner(
                tmp / "state.sqlite", tmp / "runs",
                memory_probe=lambda: 10**12, process_probe=lambda: ["python -m grpo_mlx"],
            )
            busy.prepare(competing)
            self.assertEqual("mlx_refused_headroom", busy.execute(competing.run_id).failure_class)
            clear = _spec(tmp, ("python", "-c", "open('ran','w').write('x')"), backend="mlx", run_id="mlx-clear")
            opened = ExperimentRunner(
                tmp / "state.sqlite", tmp / "runs",
                memory_probe=lambda: 10**12, process_probe=lambda: ["python train.py"],
            )
            opened.prepare(clear)
            available = opened.execute(clear.run_id)
            self.assertEqual("mlx_unavailable", available.failure_class)
            self.assertFalse((spec.workspace / "ran").exists())

    def test_docker_failure_does_not_fall_back_to_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(
                tmp,
                ("python", "-c", "open('ran','w').write('x')"),
                isolation="docker",
                docker_image=IMAGE,
                network_policy="deny",
            )
            calls: list[list[str]] = []

            def explode(argv, _limits):
                calls.append(list(argv))
                from vare.container_isolation import IsolationError
                raise IsolationError("docker missing")

            runner = ExperimentRunner(tmp / "state.sqlite", tmp / "runs", docker_execute=explode)
            runner.prepare(spec)
            view = runner.execute(spec.run_id)
            self.assertEqual("isolation_unavailable", view.failure_class)
            self.assertFalse(view.accepted)
            self.assertFalse((spec.workspace / "ran").exists())
            self.assertTrue(calls)
            self.assertIn("--network=none", calls[0])
            (spec.workspace / "metrics.json").write_text('{"loss": 0.2}', encoding="utf-8")

            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            def succeed(argv, _limits):
                calls.append(list(argv))
                return _Result()

            spec_ok = _spec(
                tmp,
                ("python", "-c", "open('ran','w').write('x')"),
                run_id="docker-ok",
                isolation="docker",
                docker_image=IMAGE,
                network_policy="deny",
            )
            (spec_ok.workspace / "metrics.json").write_text('{"loss": 0.2}', encoding="utf-8")
            ok_runner = ExperimentRunner(tmp / "state.sqlite", tmp / "runs", docker_execute=succeed)
            ok_runner.prepare(spec_ok)
            ok_view = ok_runner.execute(spec_ok.run_id)
            self.assertTrue(ok_view.accepted)
            self.assertFalse((spec_ok.workspace / "ran").exists())

    def test_spec_roundtrip_and_script_status(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            spec = _spec(tmp, ("python", "-c", _write_metrics("print(1)")))
            payload = spec.workspace.parent / "spec.json"
            # load_spec resolves the workspace; dumping through prepare is enough.
            runner = _runner(tmp, spec)
            self.assertTrue(runner.execute(spec.run_id).accepted)
            from vare.research_lab.spec import spec_to_dict
            from vare.research_lab.jsonutil import canonical_bytes
            payload.write_bytes(canonical_bytes(spec_to_dict(spec)))
            loaded = load_spec(payload)
            self.assertEqual(spec.run_id, loaded.run_id)
            self.assertEqual(spec.workspace, loaded.workspace)
            script = ROOT / "scripts" / "research_lab_exec.py"
            completed = subprocess.run(
                [sys.executable, str(script), "--store", str(tmp / "state.sqlite"), "--runs", str(tmp / "runs"),
                 "status", "--run-id", spec.run_id],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            body = strict_load(completed.stdout)
            self.assertTrue(body["accepted"])
            self.assertEqual(1, ResumeStore(tmp / "state.sqlite").audit()["runs"])


if __name__ == "__main__":
    unittest.main()
