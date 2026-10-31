"""Model-visible prompt. Identical for every arm and every split.

Visible: repository name, issue text, target file path and its current
(buggy) contents. Never visible: tests, test names, the bug patch, the
reference fix, grader output.
"""
from __future__ import annotations

from typing import Any, Mapping

SYSTEM = (
    "You are a careful software engineer. You fix bugs in Python repositories by "
    "editing a single source file. Reply with one or more SEARCH/REPLACE blocks and "
    "nothing else."
)

FORMAT_HELP = """Edit format (repeat the block for several edits):
<<<<<<< SEARCH
exact, contiguous lines copied from the current file
=======
the corrected lines
>>>>>>> REPLACE
Rules: SEARCH text must match the file exactly and occur exactly once. Keep blocks short. Do not edit tests."""

PROMPT_VERSION = "v1"


def build_messages(task: Mapping[str, Any], file_text: str) -> list[dict[str, str]]:
    user = (
        f"Repository: {task['upstream_repo']}\n\n"
        f"Issue:\n{task['_problem'].strip()}\n\n"
        f"The bug is in `{task['target_file']}`. Current contents:\n"
        f"```python\n{file_text}\n```\n\n"
        f"{FORMAT_HELP}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
