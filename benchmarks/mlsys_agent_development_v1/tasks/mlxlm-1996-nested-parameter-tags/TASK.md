# Task: preserve nested markup inside Qwen3 Coder parameter values

In `mlx_lm/tool_parsers/qwen3_coder.py`, fix parsing of a parameter value that contains nested function or parameter markup. Preserve the full value and continue parsing later sibling parameters. A nested opening marker that does not balance should remain literal content when a closing tag is present.

Only edit `mlx_lm/tool_parsers/qwen3_coder.py`. The candidate agent will not receive the trusted grader or the upstream repair.
