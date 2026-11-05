from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vare import rsi_cli


def _complete_repo(root: Path) -> Path:
    for relative in rsi_cli.MODEL_EXECUTION_SCRIPTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return root


def _complete_model(root: Path) -> Path:
    snapshot = (
        root
        / f"models--{rsi_cli.MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / rsi_cli.MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    return snapshot


def test_model_snapshot_requires_configuration_and_weight_files(tmp_path):
    snapshot = _complete_model(tmp_path)
    (snapshot / "model.safetensors").unlink()

    path, ready = rsi_cli._model_snapshot(str(snapshot))

    assert path == snapshot
    assert ready is False


def test_model_snapshot_rejects_arbitrary_local_model_path(tmp_path):
    arbitrary_model = tmp_path / "different-model"
    arbitrary_model.mkdir()
    (arbitrary_model / "config.json").write_text("{}\n", encoding="utf-8")
    (arbitrary_model / "model.safetensors").write_bytes(b"fixture")

    _, ready = rsi_cli._model_snapshot(str(arbitrary_model))

    assert ready is False


def test_docker_probe_finds_available_local_unix_context(monkeypatch):
    docker = "/usr/bin/docker"
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: docker)
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv == [docker, "context", "show"]:
            return SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        if argv == [docker, "info", "--format", "{{.ServerVersion}}"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="unavailable")
        if argv == [docker, "context", "ls", "--format", "{{json .}}"]:
            rows = [
                {"Name": "remote", "DockerEndpoint": "tcp://example.invalid:2376"},
                {"Name": "default", "DockerEndpoint": "unix:///var/run/docker.sock"},
                {
                    "Name": "colima",
                    "DockerEndpoint": "unix:///Users/user/.colima/default/docker.sock",
                },
            ]
            return SimpleNamespace(
                returncode=0,
                stdout="".join(json.dumps(row) + "\n" for row in rows),
                stderr="",
            )
        if argv == [
            docker,
            "--context",
            "colima",
            "info",
            "--format",
            "{{.ServerVersion}}",
        ]:
            return SimpleNamespace(returncode=0, stdout="29.5.2\n", stderr="")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(rsi_cli.subprocess, "run", run)

    version, reason, context = rsi_cli._docker_server_version()

    assert (version, reason, context) == ("29.5.2", "docker_daemon_available", "colima")
    assert not any("remote" in call for call in calls)


def test_docker_probe_does_not_accept_remote_only_context(monkeypatch):
    docker = "/usr/bin/docker"
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: docker)

    def run(argv, **_kwargs):
        if argv == [docker, "context", "show"]:
            return SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        if argv == [docker, "info", "--format", "{{.ServerVersion}}"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="unavailable")
        if argv == [docker, "context", "ls", "--format", "{{json .}}"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"Name": "remote", "DockerEndpoint": "tcp://example.invalid:2376"}
                )
                + "\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(rsi_cli.subprocess, "run", run)

    version, reason, context = rsi_cli._docker_server_version()

    assert version is None
    assert reason == "docker_daemon_unavailable"
    assert context is None


def test_macos_resource_parsers_read_memory_pressure_and_swap(monkeypatch):
    outputs = {
        "vm_stat": (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 100.\nPages inactive: 200.\nPages speculative: 10.\n"
        ),
        "memory_pressure": "System-wide memory free percentage: 25%\n",
        "sysctl": "total = 2048.00M used = 512.00M free = 1536.00M\n",
    }
    monkeypatch.setattr(
        rsi_cli,
        "_run_readonly",
        lambda argv: outputs[Path(argv[0]).name if argv[0] == "vm_stat" else argv[0]],
    )

    assert rsi_cli._available_memory_bytes() == 16384 * 310
    assert rsi_cli._memory_free_pct() == 25.0
    assert rsi_cli._swap_usage() == {
        "total_gib": 2.0,
        "used_gib": 0.5,
        "fraction": 0.25,
    }


def test_preflight_reports_runtime_blockers_without_capability_claims(
    tmp_path, monkeypatch
):
    repo = _complete_repo(tmp_path / "repo")
    monkeypatch.setattr(rsi_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rsi_cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(rsi_cli, "_module_available", lambda _name: False)
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: None)

    result = rsi_cli.build_preflight(repo_root=repo, run_root=repo, min_free_disk_gib=0)

    assert result["runtime_status"] == "blocked"
    assert "python_module:mlx" in result["blockers"]
    assert "python_module:mlx_lm" in result["blockers"]
    assert "docker_candidate_worker" in result["blockers"]
    assert result["scope"] == "local_runtime_only"
    assert result["task_allocation"] == "not_assessed"
    assert result["policy_update"] == "not_run"
    assert result["model_loaded"] is False
    assert result["model_downloaded"] is False
    assert result["candidate_executed"] is False


def test_preflight_ready_means_runtime_only(tmp_path, monkeypatch):
    repo = _complete_repo(tmp_path / "repo")
    model = _complete_model(tmp_path / "model")
    monkeypatch.setattr(rsi_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rsi_cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(rsi_cli, "_module_available", lambda _name: True)
    monkeypatch.setattr(rsi_cli, "_available_memory_bytes", lambda: 12 * rsi_cli.GIB)
    monkeypatch.setattr(rsi_cli, "_memory_free_pct", lambda: 30.0)
    monkeypatch.setattr(
        rsi_cli,
        "_swap_usage",
        lambda: {"total_gib": 3.0, "used_gib": 0.5, "fraction": 1 / 6},
    )
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        rsi_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="29.0.1\n", stderr=""
        ),
    )

    result = rsi_cli.build_preflight(
        repo_root=repo,
        run_root=repo,
        model_path=str(model),
        min_free_disk_gib=0,
    )

    assert result["runtime_status"] == "ready"
    assert result["task_allocation"] == "not_assessed"
    assert result["verifier_qualification"] == "not_assessed"
    assert result["capability_evaluation"] == "not_run"


def test_preflight_blocks_resource_pressure_at_pinned_runner_limits(
    tmp_path, monkeypatch
):
    repo = _complete_repo(tmp_path / "repo")
    monkeypatch.setattr(rsi_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rsi_cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(rsi_cli, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        rsi_cli, "_model_snapshot", lambda _path: (tmp_path / "model", True)
    )
    monkeypatch.setattr(rsi_cli, "_available_memory_bytes", lambda: 7 * rsi_cli.GIB)
    monkeypatch.setattr(rsi_cli, "_memory_free_pct", lambda: 14.0)
    monkeypatch.setattr(
        rsi_cli,
        "_swap_usage",
        lambda: {"total_gib": 3.0, "used_gib": 2.82, "fraction": 0.94},
    )
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        rsi_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="29.0.1\n", stderr=""
        ),
    )

    result = rsi_cli.build_preflight(
        repo_root=repo,
        run_root=repo,
        model_path="unused-by-fixture",
        min_free_disk_gib=0,
    )

    assert result["runtime_status"] == "blocked"
    assert {"available_memory", "memory_pressure", "swap_pressure"}.issubset(
        set(result["blockers"])
    )


def test_preflight_cli_writes_json_and_returns_blocked_status(
    tmp_path, monkeypatch, capsys
):
    repo = _complete_repo(tmp_path / "repo")
    monkeypatch.setattr(rsi_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rsi_cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(rsi_cli, "_module_available", lambda _name: False)
    monkeypatch.setattr(rsi_cli.shutil, "which", lambda _name: None)
    output = tmp_path / "audit" / "preflight.json"

    code = rsi_cli.main(
        [
            "preflight",
            "--repo-root",
            str(repo),
            "--run-root",
            str(repo),
            "--min-free-disk-gib",
            "0",
            "--output",
            str(output),
        ]
    )

    assert code == 2
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["runtime_status"] == "blocked"
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_preflight_rejects_nonfinite_disk_floor(tmp_path):
    try:
        rsi_cli.build_preflight(
            repo_root=tmp_path,
            run_root=tmp_path,
            min_free_disk_gib=float("nan"),
        )
    except ValueError as exc:
        assert "finite and non-negative" in str(exc)
    else:
        raise AssertionError("non-finite disk floor was accepted")
