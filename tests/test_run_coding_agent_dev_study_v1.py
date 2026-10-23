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
    def test_frozen_protocol_and_implementation_hashes_validate(self):
        protocol, lock, protocol_bytes, lock_bytes = runner.load_frozen()
        self.assertEqual(protocol["protocol_id"], lock["protocol_id"])
        self.assertEqual(runner.sha256_bytes(protocol_bytes), lock["protocol_sha256"])
        self.assertTrue(lock_bytes)

    def test_run_records_exact_runner_source_used_for_inference(self):
        run_dir = ROOT / "results/phase-a-v6/protocol-v1/run-2"
        manifest = json.loads((run_dir / "run-manifest.json").read_text())
        snapshot = run_dir / "runner.snapshot.py"
        self.assertEqual(runner.sha256_file(snapshot), manifest["runner_sha256"])

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
