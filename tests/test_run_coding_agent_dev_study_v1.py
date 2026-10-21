import importlib.util
import json
from pathlib import Path
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
            self.assertEqual(workspace.resolve(), workspace)
            self.assertEqual(workspace.parent.stat().st_mode & 0o777, 0o700)

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
        self.assertEqual(
            runner.classify_public_test_result({"passed": False, "returncode": 1}),
            "test_failure",
        )
        self.assertEqual(
            runner.classify_public_test_result({
                "passed": False,
                "error": "IsolationError: container exceeded wall-time limit",
            }),
            "candidate_timeout",
        )
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
