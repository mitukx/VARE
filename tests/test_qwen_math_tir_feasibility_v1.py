import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_qwen_math_tir_feasibility_data_v1 as generator
import run_cpu_qwen_math_tir_feasibility_v1 as runner
import audit_cpu_qwen_math_tir_feasibility_v1 as auditor


class QwenMathTIRFeasibilityTests(unittest.TestCase):
    def test_generator_has_balanced_deterministic_exact_integer_rows(self):
        first = generator.generate_rows()
        second = generator.generate_rows()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({row["task_id"] for row in first}), 96)
        counts = {}
        for row in first:
            counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertIs(type(row["gold_answer"]), int)
        self.assertEqual(counts, {"inventory": 24, "event_revenue": 24,
                                 "factory_output": 24, "equal_distribution": 24})
        self.assertEqual(first, auditor.regenerate_independently())

    def test_boxed_answer_parser_uses_last_integer(self):
        self.assertEqual(runner.parse_boxed_integer("work \\boxed{12}, final \\boxed{-7}"), -7)
        self.assertIsNone(runner.parse_boxed_integer("answer: 7"))

    def test_restricted_calculator_accepts_integer_arithmetic(self):
        self.assertEqual(runner.safe_integer_expression("(12 + 3) * 4 - 5"), 55)
        self.assertEqual(auditor.safe_integer_expression("(12 + 3) * 4 - 5"), 55)

    def test_restricted_calculator_rejects_code_and_unsafe_nodes(self):
        for expression in ("__import__('os').system('true')", "2 ** 100", "4 // 0", "3.5 + 1"):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, SyntaxError)):
                    runner.safe_integer_expression(expression)
                with self.assertRaises((ValueError, SyntaxError)):
                    auditor.safe_integer_expression(expression)

    def test_tool_call_schema_is_parsed_strictly(self):
        call = '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
        self.assertEqual(runner.parse_tool_call(call), "2+3")
        self.assertIsNone(runner.parse_tool_call("no call"))
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            runner.parse_tool_call('<tool_call>{"name":"shell","arguments":{}}</tool_call>')


if __name__ == "__main__":
    unittest.main()
