from __future__ import annotations

import unittest

from mlx_lm.tool_parsers import qwen3_coder


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string"},
                },
            },
        },
    }]


class TestQwen3CoderNestedParameters(unittest.TestCase):
    def test_nested_function_markup_is_preserved_and_sibling_arguments_continue(self):
        value = "<function=demo>\n<parameter=a>\nv\n</parameter>\n</function>"
        call = (
            "<function=write_file>\n"
            "<parameter=path>\nexample.txt\n</parameter>\n"
            f"<parameter=content>\n{value}\n</parameter>\n"
            "<parameter=mode>\nappend\n</parameter>\n"
            "</function>"
        )

        result = qwen3_coder.parse_tool_call(call, _tools())

        self.assertEqual(result["arguments"], {
            "path": "example.txt", "content": value, "mode": "append",
        })

    def test_unbalanced_nested_opening_tag_remains_literal_and_sibling_parses(self):
        value = "Wrap each argument in <parameter=NAME> and close it."
        call = (
            "<function=write_file>\n"
            "<parameter=path>\nREADME.md\n</parameter>\n"
            f"<parameter=content>\n{value}\n</parameter>\n"
            "<parameter=mode>\nappend\n</parameter>\n"
            "</function>"
        )

        result = qwen3_coder.parse_tool_call(call, _tools())

        self.assertEqual(result["arguments"]["content"], value)
        self.assertEqual(result["arguments"]["mode"], "append")
        self.assertNotIn("NAME", result["arguments"])


if __name__ == "__main__":
    unittest.main()
