"""Public tests visible to the proposer (adapted from tests/test_learning_pipeline_v1.py)."""
from edit_format import apply_blocks, parse_blocks, reference_fix_blocks, render_blocks

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 def add(a, b):
-    return a + b
+    return a - b
 
 
"""
CLEAN = "def add(a, b):\n    return a + b\n\n\nX = 1\n"
BUGGY = "def add(a, b):\n    return a - b\n\n\nX = 1\n"


def test_reference_fix_blocks_restore_clean_file():
    path, blocks = reference_fix_blocks(PATCH)
    assert path == "pkg/mod.py"
    out = apply_blocks(BUGGY, blocks)
    assert out.ok and out.new_text == CLEAN
    again = apply_blocks(BUGGY, parse_blocks(render_blocks(blocks)))
    assert again.ok and again.new_text == CLEAN


def test_apply_blocks_fail_closed():
    cases = [
        ("no blocks here", "no_edit_block"),
        ("<<<<<<< SEARCH\nnot in file\n=======\nx\n>>>>>>> REPLACE", "search_not_found"),
        ("<<<<<<< SEARCH\n    return a - b\n=======\n    return a - b\n>>>>>>> REPLACE", "no_op_edit"),
    ]
    for text, reason in cases:
        out = apply_blocks(BUGGY, parse_blocks(text))
        assert not out.ok and out.reason.startswith(reason)


def test_ambiguous_search_rejected():
    blocks = parse_blocks("<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
    out = apply_blocks("x = 1\nx = 1\n", blocks)
    assert not out.ok and out.reason.startswith("ambiguous")


def test_think_section_is_ignored():
    text = "<think><<<<<<< SEARCH\nX = 1\n=======\nX = 9\n>>>>>>> REPLACE</think>" \
           "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    out = apply_blocks(BUGGY, parse_blocks(text))
    assert out.ok and out.new_text == CLEAN


def test_trailing_whitespace_tolerated():
    text = "a = 1   \nb = 2\n"
    out = apply_blocks(text, parse_blocks("<<<<<<< SEARCH\na = 1\t\nb = 2\n=======\nc\n>>>>>>> REPLACE"))
    assert out.ok and out.new_text == "c\n"
