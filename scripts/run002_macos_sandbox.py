"""Per-process macOS Seatbelt boundary for the Run-002 R7 checker.

This deliberately lives in ``scripts/``. The Run-001 ``RepoEnv`` module is
source-pinned and remains byte-identical; the Run-002 checker temporarily wraps
its subprocess helper only while building or grading a repository.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
from typing import Sequence

R7_EXTRA_TEST_DEPS: dict[str, tuple[str, ...]] = {
    # The frozen Run-001 extras omit Click, but this repository's pytest
    # conftest imports click.testing during collection.
    "theskumar/python-dotenv": ("click",),
}


class MacosSeatbeltRepoEnv:
    """Run repository-controlled pip and pytest processes inside Seatbelt."""

    def __init__(self, repo_env):
        self.env = repo_env
        self._repo_env_module = None
        self._original_run = None
        self.install_profile_sha256: str | None = None
        self.grader_profile_sha256: str | None = None

    @property
    def root(self) -> Path:
        return self.env.cache / "r7-seatbelt" / self.env.name

    @property
    def protected_git_dir(self) -> Path:
        return self.env.cache.parent / f".r7-protected-git-{self.env.name}"

    def _paths(self) -> tuple[Path, Path, Path]:
        if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
            raise RuntimeError("macOS sandbox unavailable")
        root = self.root.resolve()
        home, tmp = root / "home", root / "tmp"
        for path in (home, tmp, root / "pip-cache"):
            path.mkdir(parents=True, exist_ok=True)
        return root, home, tmp

    @staticmethod
    def _sbpl_path(path: Path) -> str:
        value = str(path.resolve())
        if any(char in value for char in ('"', "\\", "\n", "\r")):
            raise RuntimeError("sandbox path contains unsupported characters")
        return json.dumps(value)

    def _profile_text(self, phase: str) -> tuple[str, Path]:
        if phase not in {"install", "grader"}:
            raise ValueError(f"unsupported Run-002 sandbox phase: {phase}")
        root, _sandbox_home, _tmp = self._paths()
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow file-read*)",
            f"(deny file-read* (subpath {self._sbpl_path(Path.home())}))",
            "(allow file-read* (literal \"/dev/null\"))",
            "(allow file-write* (literal \"/dev/null\"))",
        ]
        # The order is deliberate: repo, venv and task-local temporary data are
        # the only operator-home exceptions after denying the home subtree.
        for path in (self.env.cache, root):
            lines.append(f"(allow file-read* (subpath {self._sbpl_path(path)}))")
        if phase == "install":
            for path in (self.env.path, self.env.venv):
                lines.append(f"(allow file-write* (subpath {self._sbpl_path(path)}))")
            lines.append("(allow network-outbound)")
        else:
            # Some upstream test suites write fixtures into their checkout. The
            # clone is disposable and RepoEnv.reset() runs after every grade.
            # Keep Git metadata outside the sandbox and deny its recreation.
            lines.append(f"(allow file-write* (subpath {self._sbpl_path(self.env.path)}))")
        # Repository-controlled setup and tests must not be able to add hooks
        # or alter local Git config that the trusted host invokes during reset.
        lines.append(f"(deny file-write* (subpath {self._sbpl_path(self.env.path / '.git')}))")
        lines.append(f"(allow file-write* (subpath {self._sbpl_path(root)}))")
        return "\n".join(lines) + "\n", root

    def profile_sha256(self, phase: str) -> str:
        text, _root = self._profile_text(phase)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _child_environment(self, root: Path) -> dict[str, str]:
        home, tmp = root / "home", root / "tmp"
        ca_file = ssl.get_default_verify_paths().cafile
        if not ca_file or not Path(ca_file).is_file():
            raise RuntimeError("Python CA bundle unavailable for sandboxed package install")
        pip_config = root / "pip.conf"
        pip_config.write_text(
            "[global]\n"
            f"cert = {ca_file}\n"
            "use-deprecated = legacy-certs\n"
            "disable-pip-version-check = true\n"
            "no-input = true\n"
            "no-cache-dir = true\n",
            encoding="utf-8",
        )
        return {
            "PATH": os.pathsep.join((str(self.env.python.parent), "/usr/bin", "/bin",
                                     "/usr/sbin", "/sbin")),
            "HOME": str(home),
            "USER": "r7-sandbox",
            "LOGNAME": "r7-sandbox",
            "LNAME": "r7-sandbox",
            "USERNAME": "r7-sandbox",
            "TMPDIR": str(tmp),
            "PIP_CACHE_DIR": str(root / "pip-cache"),
            "PIP_CONFIG_FILE": str(pip_config),
            "PIP_CERT": ca_file,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_ADDOPTS": "",
            "VIRTUAL_ENV": str(self.env.venv),
            "SSL_CERT_FILE": ca_file,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "TZ": "UTC",
        }

    def _run_untrusted(self, cmd: Sequence[str], *, phase: str, cwd: Path | None,
                       timeout: float):
        text, root = self._profile_text(phase)
        profile = root / f"profile-{phase}.sb"
        profile.write_text(text, encoding="utf-8")
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-f", str(profile), *cmd],
            cwd=cwd or self.env.path,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=self._child_environment(root),
            stdin=subprocess.DEVNULL,
        )

    def _git_environment(self) -> dict[str, str]:
        _root, home, tmp = self._paths()
        return {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
        }

    def _restore_git_metadata(self) -> None:
        hidden = self.protected_git_dir
        target = self.env.path / ".git"
        if hidden.exists():
            if target.exists():
                raise RuntimeError("sandbox git metadata recovery collision")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(hidden, target)

    def _write_git_stub(self, git_dir: Path) -> dict[str, str]:
        files = {
            "HEAD": f"{self.env.commit}\n",
            "config": "[core]\n\trepositoryformatversion = 0\n\tfilemode = false\n\tbare = false\n",
        }
        git_dir.mkdir()
        for rel, content in files.items():
            (git_dir / rel).write_text(content, encoding="utf-8")
        return files

    @staticmethod
    def _git_stub_unchanged(git_dir: Path, expected: dict[str, str]) -> bool:
        if not git_dir.is_dir():
            return False
        actual_files = {p.relative_to(git_dir).as_posix(): p.read_text(encoding="utf-8")
                        for p in git_dir.rglob("*") if p.is_file()}
        return actual_files == expected

    def run_untrusted(self, cmd: Sequence[str], *, phase: str,
                      cwd: Path | None = None, timeout: float = 600):
        if phase == "install":
            result = self._run_untrusted(cmd, phase=phase, cwd=cwd, timeout=timeout)
            self.install_profile_sha256 = self.profile_sha256("install")
            return result
        if phase != "grader":
            raise ValueError(f"unsupported Run-002 sandbox phase: {phase}")

        self._restore_git_metadata()
        git_dir, hidden = self.env.path / ".git", self.protected_git_dir
        moved_git = git_dir.is_dir()
        expected_stub: dict[str, str] | None = None
        if moved_git:
            if hidden.exists():
                raise RuntimeError("sandbox git metadata collision")
            hidden.parent.mkdir(parents=True, exist_ok=True)
            os.replace(git_dir, hidden)
            expected_stub = self._write_git_stub(git_dir)
        try:
            result = self._run_untrusted(cmd, phase=phase, cwd=cwd, timeout=timeout)
        finally:
            unexpected_git = moved_git and not self._git_stub_unchanged(git_dir, expected_stub or {})
            if git_dir.is_dir():
                shutil.rmtree(git_dir, ignore_errors=True)
            else:
                git_dir.unlink(missing_ok=True)
            if moved_git:
                os.replace(hidden, git_dir)
            if unexpected_git:
                raise RuntimeError("sandbox recreated protected git metadata")
        self.grader_profile_sha256 = self.profile_sha256("grader")
        return result

    def installed_distributions(self) -> list[dict[str, str]]:
        script = (
            "import importlib.metadata as m, json; "
            "rows = sorted({(d.metadata.get('Name'), d.version) for d in m.distributions() "
            "if d.metadata.get('Name') and d.version}); "
            "print(json.dumps([{'name': n, 'version': v} for n, v in rows]))"
        )
        result = self.run_untrusted(
            [str(self.env.python), "-c", script], phase="grader",
            cwd=self.env.path, timeout=120,
        )
        if result.returncode:
            raise RuntimeError("dependency inventory failed")
        rows = json.loads(result.stdout)
        if not rows:
            raise RuntimeError("dependency inventory empty")
        return rows

    def __enter__(self):
        from vare.learning_pipeline import repo_env as repo_env_module

        self._repo_env_module = repo_env_module
        self._original_run = repo_env_module._run
        original = self._original_run

        def wrapped(cmd, cwd=None, timeout=600, env=None):
            command = list(cmd)
            if command and command[0] == "git":
                command[1:1] = ["-c", "core.hooksPath=/dev/null"]
                return original(command, cwd=cwd, timeout=timeout, env=self._git_environment())
            if len(command) >= 3 and command[1:3] == ["-m", "pip"]:
                if "install" in command:
                    at = command.index("install") + 1
                    command[at:at] = ["--use-deprecated", "legacy-certs"]
                    command.extend(R7_EXTRA_TEST_DEPS.get(self.env.upstream, ()))
                return self.run_untrusted(command, phase="install", cwd=cwd, timeout=timeout)
            if len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
                return self.run_untrusted(command, phase="grader", cwd=cwd, timeout=timeout)
            return original(command, cwd=cwd, timeout=timeout, env=env)

        repo_env_module._run = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._repo_env_module is not None and self._original_run is not None:
            self._repo_env_module._run = self._original_run
        self._repo_env_module = None
        self._original_run = None
