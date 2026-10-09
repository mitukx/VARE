import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_synthetic_preference_robustness_bundle_independent as verifier


class IndependentRobustnessVerifierTests(unittest.TestCase):
    def setUp(self):
        self.bundle = ROOT / "results/synthetic-preference-robustness-v1/confirmation"

    def test_replays_retained_bundle_without_runner_import(self):
        result = verifier.audit(self.bundle)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["seed_count"], 10)
        self.assertTrue(result["acceptance_passed"])
        self.assertFalse(result["external_human_reproduction"])
        self.assertIn("not_performed", result["design_generation_replay"])

    def test_rejects_metric_tampering_even_if_manifest_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "bundle"
            shutil.copytree(self.bundle, altered)
            seed_file = altered / "seeds/seed-503.json"
            seed_record = json.loads(seed_file.read_text(encoding="utf-8"))
            seed_record["metrics"]["clean"]["base_teacher"]["heldout_preference_nll"] += 0.1
            seed_file.write_text(json.dumps(seed_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                 encoding="utf-8")

            files = {}
            for path in sorted(altered.rglob("*")):
                if path.is_file() and path.name != "manifest.json":
                    files[path.relative_to(altered).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            (altered / "manifest.json").write_text(
                json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metric mismatch"):
                verifier.audit(altered)


if __name__ == "__main__":
    unittest.main()
