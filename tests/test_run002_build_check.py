"""Run-002 R7 build check: report hygiene (no network, no model)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import platform
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("run002_build_check", ROOT / "scripts" / "run002_build_check.py")
assert _spec and _spec.loader
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)
from run002_macos_sandbox import MacosSeatbeltRepoEnv


def test_summarize_drops_test_names_and_output() -> None:
    grade = {"passed": False, "reason": "tests_failed", "f2p_passed": 0, "f2p_total": 2,
             "p2p_passed": 3, "p2p_total": 3, "pytest_exit": 1, "stdout_sha256": "0" * 64,
             "seconds": 1.0, "pytest_seconds": 0.9, "missing": ["tests/x.py::secret_name"],
             "tests": {"tests/x.py::other_name": "FAILED"}, "stdout": "secret output",
             "apply_stderr": "secret"}
    out = bc.summarize(grade)
    text = json.dumps(out)
    assert "secret" not in text and "::" not in text
    assert out["missing_count"] == 1 and out["f2p_total"] == 2


def test_sanitize_removes_host_paths(tmp_path: Path) -> None:
    cache, data = tmp_path / "cache", tmp_path / "data"
    text = f"error in {cache}/repos/x and {data}/shard and {Path.home()}/y"
    out = bc.sanitize(text, cache, data)
    assert str(tmp_path) not in out and str(Path.home()) not in out


def test_report_settings_mismatch_refuses_with_exit_2(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    argv = ["--report", str(report), "--cache", str(tmp_path / "c"), "--dataset-cache", str(tmp_path / "d")]
    args = bc.build_parser().parse_args(argv)
    checker = bc.Checker(args)
    rep = checker.load_report({"a": "1"}, {"tasks_per_repo": 8})
    checker.save(rep)
    with pytest.raises(SystemExit) as exc:
        checker.load_report({"a": "1"}, {"tasks_per_repo": 7})
    assert exc.value.code == 2


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS Seatbelt integration probe")
def test_run002_seatbelt_denies_home_reads_and_external_writes(tmp_path: Path, monkeypatch) -> None:
    from vare.learning_pipeline.repo_env import RepoEnv

    fake_home = tmp_path / "operator-home"
    fake_home.mkdir()
    secret = fake_home / "private.txt"
    secret.write_text("do not disclose", encoding="utf-8")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    env = RepoEnv("probe/synthetic.abc123", "probe/synthetic", "0" * 40,
                  fake_home / ".cache")
    env.path.mkdir(parents=True)
    env.venv.mkdir(parents=True)
    (env.path / "allowed.txt").write_text("source", encoding="utf-8")
    (env.path / ".git").mkdir()
    (env.path / ".git" / "HEAD").write_text("protected reference metadata", encoding="utf-8")
    outside_write = tmp_path / "outside-write.txt"
    isolation = MacosSeatbeltRepoEnv(env)
    probe = """
from pathlib import Path
import getpass, os, subprocess
print("source=" + Path("allowed.txt").read_text())
print("login=" + getpass.getuser())
for label, action in (
    ("home", lambda: Path(%r).read_text()),
    ("outside_write", lambda: Path(%r).write_text("escape")),
    ("repo_write", lambda: Path("allowed-output.txt").write_text("temporary")),
    ("venv_write", lambda: Path(%r).write_text("escape")),
    ("git_write", lambda: Path(".git/HEAD").write_text("overwrite")),
    ("git_payload", lambda: Path(%r).read_text()),
):
    try:
        action()
    except OSError:
        print(label + "=BLOCKED")
    else:
        print(label + "=ALLOWED")
print("git_stub_visible=" + str(Path(".git").exists()))
show = subprocess.run(["/usr/bin/git", "show", "HEAD:allowed.txt"], capture_output=True, text=True)
print("git_objects=" + ("BLOCKED" if show.returncode else "EXPOSED"))
scratch = Path(os.environ["TMPDIR"]) / "scratch.txt"
scratch.write_text("allowed")
print("scratch=" + scratch.read_text())
Path("/dev/null").write_text("discarded")
print("null=allowed")
""" % (str(secret), str(outside_write), str(env.venv / "forbidden.txt"),
       str(isolation.protected_git_dir / "HEAD"))

    result = isolation.run_untrusted([sys.executable, "-c", probe], phase="grader",
                                     cwd=env.path, timeout=15)

    assert result.returncode == 0, result.stderr
    assert "source=source" in result.stdout
    assert "login=r7-sandbox" in result.stdout
    assert "home=BLOCKED" in result.stdout
    assert "outside_write=BLOCKED" in result.stdout
    assert "repo_write=ALLOWED" in result.stdout
    assert "venv_write=BLOCKED" in result.stdout
    assert "git_write=BLOCKED" in result.stdout
    assert "git_payload=BLOCKED" in result.stdout
    assert "git_stub_visible=True" in result.stdout
    assert "git_objects=BLOCKED" in result.stdout
    assert "scratch=allowed" in result.stdout
    assert "null=allowed" in result.stdout
    assert not outside_write.exists()
    (env.path / "allowed-output.txt").unlink()
    assert (env.path / ".git" / "HEAD").read_text(encoding="utf-8") == "protected reference metadata"
    assert not isolation.protected_git_dir.exists()
    assert secret.read_text(encoding="utf-8") == "do not disclose"


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS Seatbelt integration probe")
def test_run002_install_cannot_write_git_config_or_trigger_host_hooks(tmp_path: Path) -> None:
    from vare.learning_pipeline.repo_env import RepoEnv

    env = RepoEnv("probe/synthetic.abc123", "probe/synthetic", "0" * 40, tmp_path / "cache")
    env.path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=env.path, check=True)
    (env.path / "tracked.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-c", "user.name=probe", "-c", "user.email=probe@invalid",
                    "add", "tracked.txt"], cwd=env.path, check=True)
    subprocess.run(["git", "-c", "user.name=probe", "-c", "user.email=probe@invalid",
                    "commit", "-q", "-m", "probe"], cwd=env.path, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=env.path, capture_output=True,
                            text=True, check=True).stdout.strip()
    hooks = env.path / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    marker = tmp_path / "host-hook-ran"
    hook = hooks / "post-checkout"
    hook.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\n"
                    f"Path({str(marker)!r}).write_text('ran')\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(["git", "config", "core.hooksPath", str(hooks)], cwd=env.path, check=True)
    env.commit = commit
    isolation = MacosSeatbeltRepoEnv(env)
    probe = """
from pathlib import Path
try:
    Path('.git/config').write_text('unsafe = true')
except OSError:
    print('git_config_write=BLOCKED')
else:
    print('git_config_write=ALLOWED')
Path('install-output.txt').write_text('allowed')
print('repo_write=ALLOWED')
"""

    with isolation:
        result = isolation.run_untrusted([sys.executable, "-c", probe], phase="install",
                                         cwd=env.path, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "git_config_write=BLOCKED" in result.stdout
        assert "repo_write=ALLOWED" in result.stdout
        (env.path / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        env.reset()

    assert (env.path / "tracked.txt").read_text(encoding="utf-8") == "source\n"
    assert not marker.exists()
