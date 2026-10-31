from pathlib import Path

from scripts.run_mlsys_vllm_64830_qualification_arm import build_pytest_command


def test_pytest_internal_log_is_written_inside_the_arm_output_directory(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "baseline.sb"
    python = tmp_path / "python"
    test_file = tmp_path / "test_case.py"
    pytest_config = tmp_path / "pytest.ini"
    output_dir = tmp_path / "baseline-attempt-002"
    basetemp = output_dir / "pytest-tmp"
    log_file = output_dir / "pytest-internal.log"
    test_ids = ["test_case[one]", "test_case[two]"]

    command = build_pytest_command(
        profile,
        python,
        test_file,
        pytest_config,
        basetemp,
        log_file,
        test_ids,
    )

    assert command[command.index("-o") + 1] == f"log_file={log_file}"
    assert command[command.index("--basetemp") + 1] == str(basetemp)
    assert command[-2:] == [
        f"{test_file}::test_case[one]",
        f"{test_file}::test_case[two]",
    ]
    assert "/dev/null" not in command
