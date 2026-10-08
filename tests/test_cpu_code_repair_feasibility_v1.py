import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_script("generate_cpu_code_repair_feasibility_v1.py")
auditor = load_script("audit_cpu_code_repair_feasibility_v1.py")
sandbox = load_script("code_repair_sandbox_v1.py")
grader = load_script("grade_cpu_code_repair_feasibility_v1.py")
runner = load_script("run_cpu_code_repair_feasibility_v1.py")


class CpuCodeRepairFeasibilityTests(unittest.TestCase):
    def test_protocol_lock_matches_all_source_hashes(self):
        spec, digest = runner.load_locked_spec()
        self.assertEqual(spec["protocol_id"], "vare-cpu-code-repair-feasibility-v1")
        self.assertEqual(len(digest), 64)

    def test_pilot_is_deterministic_and_independently_regenerated(self):
        first = generator.generate_rows()
        second = auditor.independently_generate()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual({family: sum(row["family"] == family for row in first)
                          for family in {row["family"] for row in first}},
                         {"boundary": 8, "sequence": 8,
                          "default_aggregation": 8, "ordered_transformation": 8})

    def test_every_bug_is_detected_and_every_known_fix_passes(self):
        auditor.validate_mutants(generator.generate_rows())

    def test_candidate_source_is_interpreted_not_executed(self):
        for source in (
            "import os\ndef f(x):\n    return x\n",
            "def f(x):\n    return x.__class__\n",
            "def f(x):\n    return open('file')\n",
            "def f(x):\n    return [y for y in x for z in x]\n",
        ):
            with self.assertRaises(Exception):
                sandbox.evaluate_function(source, "f", {"x": [1]})

    def test_short_circuit_matches_python_without_running_candidate_code(self):
        self.assertFalse(sandbox.evaluate_function("def f(x):\n    return x and 1 / 0\n", "f", {"x": 0}))
        self.assertTrue(sandbox.evaluate_function("def f(x):\n    return x or 1 / 0\n", "f", {"x": 1}))

    def test_bool_and_integer_results_are_not_equal(self):
        self.assertTrue(grader.typed_equal([True], [True]))
        self.assertFalse(grader.typed_equal(True, 1))
        self.assertFalse(grader.typed_equal([True], [1]))

    def test_tool_arguments_have_exact_names_types_and_lengths(self):
        self.assertEqual(runner._decode_arguments({"name": "read_file", "arguments": {"path": "src/solution.py"}}),
                         ("read_file", {"path": "src/solution.py"}))
        with self.assertRaises(ValueError):
            runner._decode_arguments({"name": "read_file", "arguments": {"path": 7}})
        with self.assertRaises(ValueError):
            runner._decode_arguments({"name": "read_file", "arguments": {"path": "src/solution.py", "extra": "x"}})
        with self.assertRaises(ValueError):
            runner._decode_arguments({"name": "not_a_tool", "arguments": {}})

    def test_nested_tool_json_and_markup_inside_strings_parse_whole(self):
        payload = {"name": "edit_file", "arguments": {"path": "src/solution.py",
            "old_text": "return {'x': 1}", "new_text": "return '</tool_call>'"}}
        output = "reasoning <tool_call>" + json.dumps(payload) + "</tool_call> done"
        parsed = runner.extract_tool_payloads(output)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(json.loads(parsed[0]), payload)
        self.assertEqual(auditor.extract_calls_independently(output), parsed)

    def test_bounded_episode_replays_edit_test_and_finish(self):
        import torch

        row = generator.generate_rows()[0]
        fixed_source = auditor.known_fix(row)
        def call(name, arguments):
            return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"

        actions = [
            call("list_files", {}) + call("read_file", {"path": "src/solution.py"}),
            call("run_visible_tests", {}),
            call("edit_file", {"path": "src/solution.py", "old_text": row["source"], "new_text": fixed_source}),
            call("run_visible_tests", {}),
            call("finish", {"summary": "Repaired the function and verified visible cases."}),
        ]

        class FakeTokenizer:
            eos_token = "<eos>"
            eos_token_id = 0
            is_fast = True
            def apply_chat_template(self, *args, **kwargs): return torch.tensor([[1]])
            def decode(self, generated, **kwargs): return actions[int(generated[0])]

        class FakeModel:
            def __init__(self): self.index = 0
            def generate(self, input_ids, **kwargs):
                result = torch.cat([input_ids, torch.tensor([[self.index]])], dim=-1)
                self.index += 1
                return result

        spec = json.loads((ROOT / "protocols/cpu_code_repair_feasibility_v1.json").read_text())
        record = runner.run_episode(FakeModel(), FakeTokenizer(), row, spec, grader, torch)
        self.assertTrue(record["episode_pass"])
        self.assertTrue(record["finished"])
        self.assertEqual(record["tool_metrics"]["attempted"], 6)
        self.assertEqual(record["tool_metrics"]["accepted_edits"], 1)
        self.assertEqual(record["tool_metrics"]["visible_test_runs"], 2)

    def test_path_traversal_and_symlinks_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src/solution.py").write_text("def f(x):\n    return x\n")
            self.assertTrue(runner._path_for(root, "../outside", {"src/solution.py"})[1])
            (root / "link.py").symlink_to(root / "src/solution.py")
            self.assertTrue(runner._path_for(root, "link.py", {"link.py"})[1])


if __name__ == "__main__":
    unittest.main()
