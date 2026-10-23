from __future__ import annotations

import importlib


agent = importlib.import_module("scripts.run_local_agent")


def test_opaque_structured_edit_schema_allows_host_resolved_handles() -> None:
    tools = agent.tools_for("structured", public_tests=True, opaque_handles=True)
    by_name = {item["function"]["name"]: item["function"] for item in tools}
    for name in ("replace_text", "insert_text"):
        parameters = by_name[name]["parameters"]
        assert "workspace_revision_handle" in parameters["properties"]
        assert "source_preimage_handle" in parameters["properties"]
        assert "workspace_revision_handle" not in parameters["required"]
        assert "source_preimage_handle" not in parameters["required"]
    assert "run_public_tests" in by_name
