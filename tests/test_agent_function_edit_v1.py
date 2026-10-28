from __future__ import annotations

import hashlib
from pathlib import Path

from vare.agent_function_edit_v1 import FunctionEditLeaseStoreV1


def _store(tmp_path: Path, source: str) -> FunctionEditLeaseStoreV1:
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    return FunctionEditLeaseStoreV1(tmp_path, {"module.py"}, token_factory=iter(["lease-1", "lease-2", "lease-3", "lease-4"]).__next__)


def test_read_function_preserves_signature_and_docstring_in_immutable_prefix(tmp_path: Path):
    source = (
        "def route(value: int) -> int:\n"
        '    """Keep this contract text exactly."""\n'
        "    return value + 1\n"
    )
    store = _store(tmp_path, source)

    listed = store.list_functions("module.py")
    assert [item["qualified_name"] for item in listed["functions"]] == ["route"]
    read = store.read_function("module.py", "route")

    assert read["status"] == "read_complete"
    assert '"""Keep this contract text exactly."""' in read["preserved_prefix"]
    assert read["body"] == "    return value + 1\n"
    assert read["body_indent_prefix"] == "    "


def test_function_body_commit_is_hash_bound_and_retains_docstring(tmp_path: Path):
    original = "def route(value):\n    \"\"\"Stable docstring.\"\"\"\n    return value + 1\n"
    store = _store(tmp_path, original)
    lease = store.read_function("module.py", "route")["lease"]

    result = store.replace_function_body_transactional(lease, "    return value * 2\n")

    assert result["applied"] is True
    assert result["syntax_valid"] is True
    current = (tmp_path / "module.py").read_text(encoding="utf-8")
    assert current == "def route(value):\n    \"\"\"Stable docstring.\"\"\"\n    return value * 2\n"
    assert result["after_sha256"] == hashlib.sha256(current.encode()).hexdigest()
    assert store.replace_function_body_transactional(lease, "    return 9\n")["applied"] is False


def test_class_method_body_edit_parses_and_commits_in_full_file_context(tmp_path: Path):
    original = (
        "class Router:\n"
        "    def route(self, value):\n"
        '        """Route one value."""\n'
        "        return value + 1\n"
    )
    store = _store(tmp_path, original)
    read = store.read_function("module.py", "Router.route")
    assert read["status"] == "read_complete"
    assert read["body"] == "        return value + 1\n"
    assert read["body_indent_prefix"] == "        "

    result = store.replace_function_body_transactional(
        read["lease"], "        return value * 2\n",
    )

    assert result["applied"] is True
    current = (tmp_path / "module.py").read_text(encoding="utf-8")
    assert '"""Route one value."""' in current
    assert current.endswith("        return value * 2\n")


def test_function_body_edit_rejects_syntax_and_signature_changes_without_writing(tmp_path: Path):
    original = "def route(value):\n    return value + 1\n"
    store = _store(tmp_path, original)
    syntax_lease = store.read_function("module.py", "route")["lease"]
    syntax_result = store.replace_function_body_transactional(syntax_lease, "    return (\n")
    assert syntax_result["status"] == "candidate_rejected_syntax"
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == original

    signature_lease = store.read_function("module.py", "route")["lease"]
    signature_result = store.replace_function_body_transactional(
        signature_lease, "def route(other):\n    return other\n",
    )
    # A complete new definition cannot occupy the body slot; it is rejected as
    # invalid syntax within the immutable original signature.
    assert signature_result["status"] == "candidate_rejected_syntax"
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == original


def test_explicit_fence_normalization_is_recorded_not_implicit(tmp_path: Path):
    original = "def route(value):\n    return value + 1\n"
    store = _store(tmp_path, original)
    lease = store.read_function("module.py", "route")["lease"]
    fenced = "```python\n    return value + 2\n```\n"

    rejected = store.replace_function_body_transactional(lease, fenced)
    assert rejected["status"] == "candidate_rejected_syntax"
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == original

    lease = store.read_function("module.py", "route")["lease"]
    accepted = store.replace_function_body_transactional(
        lease, fenced, normalization="strip_single_python_fence",
    )
    assert accepted["applied"] is True
    assert accepted["normalization"] == "strip_single_python_fence"
    assert accepted["raw_replacement_sha256"] == hashlib.sha256(fenced.encode()).hexdigest()


def test_stale_function_lease_and_unsupported_functions_fail_closed(tmp_path: Path):
    source = (
        "@staticmethod\n"
        "def decorated(value):\n"
        "    return value\n\n"
        "def route(value):\n"
        "    return value + 1\n"
        "\n"
        "def compact(value): return value\n"
    )
    store = _store(tmp_path, source)
    names = [item["qualified_name"] for item in store.list_functions("module.py")["functions"]]
    assert names == ["route"]
    lease = store.read_function("module.py", "route")["lease"]
    (tmp_path / "module.py").write_text(source + "# concurrent change\n", encoding="utf-8")

    result = store.replace_function_body_transactional(lease, "    return value * 2\n")
    assert result["error"] == "stale_preimage"
    assert (tmp_path / "module.py").read_text(encoding="utf-8").endswith("# concurrent change\n")


def test_single_line_mode_lists_only_one_line_return_functions(tmp_path: Path):
    source = (
        "def simple(value):\n"
        "    return value\n\n"
        "def multiple(value):\n"
        "    if value:\n"
        "        return value\n"
        "    return 0\n"
    )
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    store = FunctionEditLeaseStoreV1(tmp_path, {"module.py"}, single_line_only=True)

    listed = store.list_functions("module.py")
    assert [item["qualified_name"] for item in listed["functions"]] == ["simple"]
    assert store.read_function("module.py", "multiple")["error"] == "single_line_return_function_required"

    lease = store.read_function("module.py", "simple")["lease"]
    rejected = store.replace_function_body_transactional(lease, "    if value:\n        return value\n")
    assert rejected["error"] == "single_line_return_replacement_required"
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == source
