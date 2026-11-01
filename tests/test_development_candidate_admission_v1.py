import ast

import pytest

from scripts import grade_vare_development_candidate_v1 as grader


@pytest.mark.parametrize("task", grader.TASKS)
def test_pinned_baseline_is_admitted(task):
    asset = grader.TASKS[task][0]
    grader.admit(task, (grader.ASSETS / asset).read_text())


@pytest.mark.parametrize("body", [
    "print('forged success')\nreturn [], []",
    "import os\nos._exit(0)",
    "return manifest.__class__",
    "return globals()",
    "json.dumps = str\nreturn [], []",
    "return getattr(manifest, 'keys')()",
    "return (lambda: 1)()",
])
def test_interference_rejected_before_execution(body):
    baseline = (grader.ASSETS / "manifest-baseline.py.txt").read_text()
    tree = ast.parse(baseline)
    function = grader.select(tree, None, "manifest_entries")
    function.body = ast.parse(body).body
    with pytest.raises(ValueError):
        grader.admit("manifest-nonempty", ast.unparse(tree))


def test_forged_module_completion_is_never_imported():
    baseline = (grader.ASSETS / "manifest-baseline.py.txt").read_text()
    with pytest.raises(ValueError, match="outside"):
        grader.admit("manifest-nonempty", "print('forged')\n" + baseline)


def test_other_function_edits_rejected():
    baseline = (grader.ASSETS / "navigation-baseline.py.txt").read_text()
    with pytest.raises(ValueError, match="outside"):
        grader.admit("navigation-overflow", baseline.replace("return self._index_sha256", "return 'fake'"))
