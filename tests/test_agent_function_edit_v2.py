from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from vare.agent_function_edit_v1 import FunctionEditLeaseStoreV2


def _store(tmp_path: Path, source: str | bytes) -> FunctionEditLeaseStoreV2:
    path = tmp_path / "module.py"
    path.write_bytes(source if isinstance(source, bytes) else source.encode("utf-8"))
    serial = iter(f"lease-{index}" for index in range(1, 100))
    return FunctionEditLeaseStoreV2(tmp_path, {"module.py"}, token_factory=serial.__next__)


def _read(store: FunctionEditLeaseStoreV2, function: str = "route") -> dict:
    value = store.read_function("module.py", function)
    assert value["status"] == "read_complete"
    return value


def test_v2_commits_one_line_and_preserves_indentation(tmp_path: Path) -> None:
    store = _store(tmp_path, "def route(value):\n    return value + 1\n")
    read = _read(store)

    result = store.replace_function_body_lines_transactional_v2(
        read["lease"], ["    return value * 2"],
    )

    assert result["applied"] is True
    assert result["line_ending"] == "LF"
    assert (tmp_path / "module.py").read_bytes() == b"def route(value):\n    return value * 2\n"


def test_v2_commits_multiline_body_and_blank_lines(tmp_path: Path) -> None:
    source = "def route(value):\n    return value + 1\n"
    store = _store(tmp_path, source)
    read = _read(store)
    lines = ["    if value:", "        result = value * 2", "", "    else:", "        result = 0", "    return result"]

    result = store.replace_function_body_lines_transactional_v2(read["lease"], lines)

    assert result["applied"] is True
    written = (tmp_path / "module.py").read_text(encoding="utf-8")
    assert "        result = value * 2\n\n    else:" in written
    assert written.endswith("    return result\n")
    ast.parse(written)


def test_v2_preserves_crlf_and_original_final_newline_state(tmp_path: Path) -> None:
    source = b"def route(value):\r\n    return value + 1\r\n"
    store = _store(tmp_path, source)
    read = _read(store)

    result = store.replace_function_body_lines_transactional_v2(
        read["lease"], ["    return value + 2", "    # trailing blank line", ""],
    )

    assert result["applied"] is True
    updated = (tmp_path / "module.py").read_bytes()
    assert b"\n" not in updated.replace(b"\r\n", b"")
    assert updated.endswith(b"\r\n\r\n")
    assert result["final_newline_preserved"] is True

    no_newline = _store(tmp_path, b"def route(value):\n    return value\n")
    # The function is not at EOF in this fixture; use a direct EOF source with no final newline.
    (tmp_path / "module.py").write_bytes(b"def route(value):\n    return value")
    no_newline = FunctionEditLeaseStoreV2(tmp_path, {"module.py"}, token_factory=lambda: "no-newline")
    eof_read = _read(no_newline)
    eof_result = no_newline.replace_function_body_lines_transactional_v2(
        eof_read["lease"], ["    return value + 1"],
    )
    assert eof_result["applied"] is True
    assert (tmp_path / "module.py").read_bytes() == b"def route(value):\n    return value + 1"
    assert hashlib.sha256((tmp_path / "module.py").read_bytes()).hexdigest() == eof_result["after_sha256"]


def test_v2_preserves_unicode_and_does_not_strip_chat_markers(tmp_path: Path) -> None:
    store = _store(tmp_path, 'def route(value):\n    """café"""\n    return value\n')
    read = _read(store)
    result = store.replace_function_body_lines_transactional_v2(
        read["lease"], ["    # <|im_start|><|im_start|> 保持文字", "    return value + 1"],
    )

    assert result["applied"] is True
    written = (tmp_path / "module.py").read_text(encoding="utf-8")
    assert '"""café"""' in written
    assert "<|im_start|><|im_start|> 保持文字" in written


def test_v2_rejects_unknown_consumed_and_invalid_line_values(tmp_path: Path) -> None:
    store = _store(tmp_path, "def route(value):\n    return value\n")
    assert store.replace_function_body_lines_transactional_v2("forged", ["    return 0"])["error"] == "unknown_or_consumed_lease"
    read = _read(store)
    invalid = store.replace_function_body_lines_transactional_v2(read["lease"], ["    return 1\n"])
    assert invalid["error"] == "invalid_replacement_lines"
    assert store.replace_function_body_lines_transactional_v2(read["lease"], ["    return 1"])["error"] == "unknown_or_consumed_lease"


def test_v2_rejects_stale_source_hash_without_writing(tmp_path: Path) -> None:
    source = "def route(value):\n    return value\n"
    store = _store(tmp_path, source)
    read = _read(store)
    (tmp_path / "module.py").write_text(source + "# concurrent\n", encoding="utf-8")

    result = store.replace_function_body_lines_transactional_v2(read["lease"], ["    return value + 1"])

    assert result["error"] == "stale_preimage"
    assert (tmp_path / "module.py").read_text(encoding="utf-8").endswith("# concurrent\n")


def test_v2_rejects_invalid_ast_then_recovers_with_fresh_lease(tmp_path: Path) -> None:
    original = "def route(value):\n    return value\n"
    store = _store(tmp_path, original)
    first = _read(store)

    rejected = store.replace_function_body_lines_transactional_v2(first["lease"], ["    if value:"])

    assert rejected["status"] == "candidate_rejected_syntax"
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == original
    second = _read(store)
    accepted = store.replace_function_body_lines_transactional_v2(
        second["lease"], ["    return value + 1"],
    )
    assert accepted["applied"] is True
    assert (tmp_path / "module.py").read_text(encoding="utf-8").endswith("    return value + 1\n")


def test_v2_rejects_mixed_file_eol_instead_of_normalizing(tmp_path: Path) -> None:
    store = _store(tmp_path, b"def route(value):\r\n    return value\n")
    result = store.read_function("module.py", "route")
    assert result["status"] == "valid_call_rejected_by_policy"
    assert result["error"] == "mixed_source_line_endings_unsupported"


@pytest.mark.parametrize("raw_lines", [None, "    return 1", [1], ["x\x00"]])
def test_v2_rejects_wrong_array_shapes(tmp_path: Path, raw_lines: object) -> None:
    store = _store(tmp_path, "def route(value):\n    return value\n")
    read = _read(store)
    result = store.replace_function_body_lines_transactional_v2(read["lease"], raw_lines)
    assert result["status"] == "invalid_json_or_schema"
