from pathlib import Path
import subprocess

import pytest

from scripts.reproduce_trajectory_worker_sqlite import require_clean_checkout


def test_reproducer_rejects_tracked_and_untracked_checkout_changes(tmp_path: Path):
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Evidence Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "evidence@example.invalid"], check=True
    )
    source = checkout / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "source"], check=True)

    require_clean_checkout(checkout)
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checkout is dirty"):
        require_clean_checkout(checkout)

    source.write_text("value = 1\n", encoding="utf-8")
    (checkout / "untracked.py").write_text("value = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checkout is dirty"):
        require_clean_checkout(checkout)
