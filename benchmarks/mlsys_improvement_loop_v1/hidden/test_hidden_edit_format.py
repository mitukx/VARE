"""Held-out behaviour tests for the edit-apply target (grader-owned).

These run inside the sandbox through the v2 trusted runner. They are not part
of the proposer view. Every test is a plain function (no parametrisation), so
the grader can list the exact test ids it requires.
"""
from edit_format import Block, apply_blocks, parse_blocks, render_blocks


def _blk(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def test_exact_single_block():
    out = apply_blocks("a = 1\nb = 2\n", parse_blocks(_blk("b = 2", "b = 3")))
    assert (out.ok, out.reason, out.new_text) == (True, "applied", "a = 1\nb = 3\n")


def test_relaxed_trailing_whitespace_in_file():
    text = "def f():  \n    return 1   \n\nx = 2\n"
    out = apply_blocks(text, parse_blocks(_blk("def f():\n    return 1", "def f():\n    return 2")))
    assert (out.ok, out.new_text) == (True, "def f():\n    return 2\n\nx = 2\n")


def test_relaxed_trailing_whitespace_in_search():
    text = "a = 1\nb = 2\nc = 3\n"
    out = apply_blocks(text, parse_blocks(_blk("a = 1  \nb = 2\t", "a = 0\nb = 0")))
    assert (out.ok, out.new_text) == (True, "a = 0\nb = 0\nc = 3\n")


def test_relaxed_ambiguous_is_not_found():
    text = "x = 1  \ny = 2\nx = 1\ny = 2\t\n"
    out = apply_blocks(text, parse_blocks(_blk("x = 1\t\ny = 2 ", "z = 0")))
    assert (out.ok, out.reason) == (False, "search_not_found_block_0")


def test_relaxed_overlapping_windows_are_ambiguous():
    text = "a  \na\na \nb\n"
    out = apply_blocks(text, parse_blocks(_blk("a\t\na\t", "c")))
    assert (out.ok, out.reason) == (False, "search_not_found_block_0")


def test_relaxed_match_at_file_end_without_newline():
    text = "a = 1\nb = 2  "
    out = apply_blocks(text, parse_blocks(_blk("b = 2\t", "b = 5")))
    assert (out.ok, out.new_text) == (True, "a = 1\nb = 5")


def test_exact_match_is_substring_based():
    out = apply_blocks("a = 1\nb = 2  ", parse_blocks(_blk("b = 2", "b = 5")))
    assert (out.ok, out.new_text) == (True, "a = 1\nb = 5  ")


def test_relaxed_needle_longer_than_file():
    out = apply_blocks("a \n", parse_blocks(_blk("a\nb\nc\nd", "x")))
    assert (out.ok, out.reason) == (False, "search_not_found_block_0")


def test_relaxed_leading_whitespace_is_significant():
    text = "    a = 1 \nb = 2\n"
    out = apply_blocks(text, parse_blocks(_blk("a = 1\nb = 2", "c")))
    assert (out.ok, out.reason) == (False, "search_not_found_block_0")


def test_relaxed_blank_lines_in_needle():
    text = "a = 1\n   \n\nb = 2 \nc\n"
    out = apply_blocks(text, parse_blocks(_blk("a = 1\n\n\nb = 2", "a = 9")))
    assert (out.ok, out.new_text) == (True, "a = 9\nc\n")


def test_relaxed_first_line_repeats_but_window_unique():
    text = "x \ny = 1\nx\ny = 2\nx\ny = 3 \n"
    out = apply_blocks(text, parse_blocks(_blk("x\ny = 2 ", "x\ny = 20")))
    assert (out.ok, out.new_text) == (True, "x \ny = 1\nx\ny = 20\nx\ny = 3 \n")


def test_exact_ambiguous_rejected():
    out = apply_blocks("x = 1\nx = 1\n", parse_blocks(_blk("x = 1", "x = 2")))
    assert (out.ok, out.reason) == (False, "ambiguous_search_block_0")


def test_empty_search_rejected():
    out = apply_blocks("x = 1\n", [Block("  \n", "y")])
    assert (out.ok, out.reason) == (False, "empty_search_block_0")


def test_no_blocks_rejected():
    out = apply_blocks("x = 1\n", parse_blocks("no edit here"))
    assert (out.ok, out.reason) == (False, "no_edit_block")


def test_no_op_rejected():
    out = apply_blocks("x = 1\n", parse_blocks(_blk("x = 1", "x = 1")))
    assert (out.ok, out.reason) == (False, "no_op_edit")


def test_blocks_apply_sequentially():
    text = "a = 1\nb = 2\n"
    comp = _blk("a = 1", "a = 2") + "\n" + _blk("a = 2\nb = 2", "c = 3")
    out = apply_blocks(text, parse_blocks(comp))
    assert (out.ok, out.new_text) == (True, "c = 3\n")


def test_second_block_failure_reports_index():
    comp = _blk("a = 1", "a = 2") + "\n" + _blk("missing", "x")
    out = apply_blocks("a = 1\n", parse_blocks(comp))
    assert (out.ok, out.reason, out.new_text) == (False, "search_not_found_block_1", None)


def test_think_section_dropped_and_marker_lengths():
    comp = "<think>" + _blk("a = 1", "a = 7") + "</think>" \
        "<<<<<<<<< SEARCH extra\na = 1\n=========\na = 3\n>>>>> REPLACE"
    blocks = parse_blocks(comp)
    assert blocks == [Block("a = 1", "a = 3")]
    assert apply_blocks("a = 1\n", blocks).new_text == "a = 3\n"


def test_render_parse_roundtrip():
    blocks = [Block("a\nb", "c"), Block("d", "e\nf")]
    assert parse_blocks(render_blocks(blocks)) == blocks


def test_crlf_lines_relaxed():
    text = "a = 1\r\nb = 2\r\n"
    out = apply_blocks(text, parse_blocks(_blk("a = 1\nb = 2", "z")))
    assert (out.ok, out.new_text) == (True, "z\n")


def test_large_file_relaxed_unique():
    lines = [f"v{i} = {i}" for i in range(5000)]
    lines[4321] += "   "
    out = apply_blocks("\n".join(lines), parse_blocks(_blk("v4321 = 4321\nv4322 = 4322", "w = 0")))
    assert out.ok and out.new_text.split("\n")[4321] == "w = 0" and len(out.new_text.split("\n")) == 4999
