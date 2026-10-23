import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_coding_agent_dev_study_v1.py")


class CodingAgentDevStudyRunnerTests(unittest.TestCase):
    def test_historical_protocol_rejects_changed_runner_source(self):
        with self.assertRaisesRegex(ValueError, "locked implementation changed: scripts/run_local_agent.py"):
            runner.load_frozen()

    def test_historical_protocol_and_raw_evidence_remain_frozen(self):
        protocol_bytes = (runner.STUDY / "protocol-v1.json").read_bytes()
        lock_bytes = (runner.STUDY / "protocol-v1.lock.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        lock = json.loads(lock_bytes)
        self.assertEqual(protocol["protocol_id"], lock["protocol_id"])
        self.assertEqual(runner.sha256_bytes(protocol_bytes), lock["protocol_sha256"])
        self.assertTrue(lock_bytes)

    def test_run_records_exact_runner_source_used_for_inference(self):
        run_dir = ROOT / "results/phase-a-v6/protocol-v1/run-2"
        manifest = json.loads((run_dir / "run-manifest.json").read_text())
        snapshot = run_dir / "runner.snapshot.py"
        self.assertEqual(runner.sha256_file(snapshot), manifest["runner_sha256"])

    def test_run_two_raw_evidence_manifest_still_verifies(self):
        run_dir = ROOT / "results/phase-a-v6/protocol-v1/run-2"
        runner.verify_sha256_manifest(run_dir, "SHA256SUMS")

    def test_candidate_workspaces_use_private_colima_shared_home_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            workspace = runner.create_docker_shared_workspace(
                "task-123", home=home,
            )
            self.assertTrue(
                workspace.is_relative_to(
                    home.resolve() / ".cache" / "vare" / "candidate-workspaces"
                )
            )
            self.assertFalse(workspace.is_symlink())
            self.assertFalse(workspace.exists())
            self.assertEqual(workspace.parent.stat().st_mode & 0o777, 0o700)

    def test_materializer_creates_allocated_workspace_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_repo = root / "source"
            source_repo.mkdir()
            subprocess.run(
                ["git", "init", str(source_repo)], check=True, capture_output=True
            )
            (source_repo / "src").mkdir()
            (source_repo / "src" / "module.py").write_text("value = 1\n")
            subprocess.run(
                ["git", "-C", str(source_repo), "add", "src/module.py"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_repo),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-m",
                    "baseline",
                ],
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            task_pack = root / "tasks"
            task_dir = task_pack / "task-123"
            task_dir.mkdir(parents=True)
            (task_dir / "test_public.py").write_text("def test_placeholder(): pass\n")
            original_task_pack = runner.TASK_PACK
            runner.TASK_PACK = task_pack
            try:
                with tempfile.TemporaryDirectory() as home_dir:
                    destination = runner.create_docker_shared_workspace(
                        "task-123", home=Path(home_dir),
                    )
                    runner.materialize_workspace(
                        source_repo,
                        "task-123",
                        destination,
                        {
                            "source_revision": revision,
                            "directories": {"src": 0o755},
                            "files": {
                                "src/module.py": {
                                    "mode": 0o644,
                                    "sha256": runner.sha256_file(
                                        source_repo / "src/module.py"
                                    ),
                                },
                                "test_public.py": {
                                    "mode": 0o644,
                                    "sha256": runner.sha256_file(task_dir / "test_public.py"),
                                },
                            },
                        },
                    )
                    self.assertEqual(
                        (destination / "src/module.py").read_text(), "value = 1\n"
                    )
                    self.assertTrue((destination / "test_public.py").is_file())
                    self.assertEqual(destination.stat().st_mode & 0o777, 0o755)
            finally:
                runner.TASK_PACK = original_task_pack

    def test_candidate_workspace_root_rejects_symlinked_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            outside = home / "outside"
            outside.mkdir()
            (home / ".cache").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                runner.create_docker_shared_workspace("task-123", home=home)

    def test_historical_public_test_results_separate_launch_timeout_and_assertion(self):
        self.assertEqual(
            runner.classify_public_test_result({"passed": False, "returncode": 125}),
            "container_launch_failure",
        )
        for returncode in (126, 127):
            self.assertEqual(
                runner.classify_public_test_result(
                    {"passed": False, "returncode": returncode}
                ),
                "container_launch_failure",
            )
        self.assertEqual(
            runner.classify_public_test_result({"passed": False, "returncode": 1}),
            "test_failure",
        )
        self.assertEqual(
            runner.classify_public_test_result({
                "passed": False,
                "gradeable": True,
                "returncode": 1,
                "classification": "assertion_failure",
            }),
            "assertion_failure",
        )
        self.assertEqual(
            runner.classify_public_test_result({
                "passed": False,
                "gradeable": True,
                "returncode": 1,
                "classification": "test_execution_error",
            }),
            "test_execution_error",
        )
        self.assertFalse(runner.is_unavailable_public_evaluator({
            "passed": False, "gradeable": True, "classification": "assertion_failure",
        }))
        self.assertFalse(runner.is_unavailable_public_evaluator({
            "passed": False, "gradeable": True, "classification": "test_execution_error",
        }))
        for malformed in (
            {"passed": True, "gradeable": False, "classification": "assertion_failure"},
            {"passed": False, "gradeable": False, "classification": "assertion_failure"},
            {"passed": False, "gradeable": True, "classification": "passed"},
            {"passed": False, "gradeable": True, "classification": "container_launch_failure"},
        ):
            self.assertEqual(
                runner.classify_public_test_result(malformed), "runner_integrity_error"
            )
            self.assertTrue(runner.is_unavailable_public_evaluator(malformed))
        self.assertEqual(
            runner.classify_public_test_result({
                "passed": True, "gradeable": True, "classification": "passed",
            }),
            "passed",
        )
        self.assertEqual(
            runner.classify_public_test_result({
                "passed": False,
                "error": "IsolationError: container exceeded wall-time limit",
            }),
            "candidate_timeout",
        )
        self.assertEqual(runner.classify_public_test_result(None), "not_run")
        self.assertFalse(runner.is_unavailable_public_evaluator(None))
        self.assertTrue(runner.is_unavailable_public_evaluator("malformed"))
        self.assertTrue(
            runner.is_unavailable_public_evaluator({"passed": False, "returncode": 125})
        )
        self.assertFalse(
            runner.is_unavailable_public_evaluator({"passed": False, "returncode": 1})
        )

    def test_evidence_manifest_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "raw.json"
            payload.write_text("{}\n", encoding="utf-8")
            (root / "SHA256SUMS").write_text(
                f"{runner.sha256_file(payload)}  raw.json\n", encoding="utf-8"
            )
            runner.verify_sha256_manifest(root, "SHA256SUMS")
            payload.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence manifest mismatch"):
                runner.verify_sha256_manifest(root, "SHA256SUMS")

    def test_workspace_snapshot_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            before = runner.workspace_snapshot_hash(root)
            (root / "outside.py").write_text("value = 2\n", encoding="utf-8")
            (root / "link.py").symlink_to(root / "outside.py")
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                runner.workspace_snapshot_hash(root)
            (root / "link.py").unlink()
            self.assertNotEqual(before, runner.workspace_snapshot_hash(root))

    def test_diff_is_deterministic_and_keeps_target_path(self):
        diff = runner.candidate_diff(b"return 1\n", b"return 2\n", "src/module.py")
        self.assertIn("--- a/src/module.py", diff)
        self.assertIn("+++ b/src/module.py", diff)
        self.assertIn("-return 1", diff)
        self.assertIn("+return 2", diff)


if __name__ == "__main__":
    unittest.main()
