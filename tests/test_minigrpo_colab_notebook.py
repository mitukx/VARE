from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "minigrpo_selfimprove_t4.ipynb"


def test_notebook_runs_both_scripts_from_drive() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in document["cells"])
    assert "drive.mount(\"/content/drive\")" in source
    assert "HF_HOME" in source
    assert "scripts/run_minigrpo.py" in source
    assert "scripts/run_selfimprove.py" in source
    assert "--check" in source
    assert "--reward\",\n        \"correctness\"" in source or "--reward correctness" in source or '"correctness"' in source
    assert "cursor/minigrpo-colab-af75" in source
    assert "7ae557604adf67be50417f59c2c2f167def9a775" in source
