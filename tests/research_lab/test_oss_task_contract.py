from pathlib import Path

import pytest

from vare.agent_task_contract import assert_split_member, assert_train_only, load_agent_task
from vare.environments.spec import (
    CommandSpec,
    EngineeringTaskSpec,
    ProtectedPath,
    RepositorySource,
    ResourceLimits,
)


def test_engineering_task_spec_is_the_existing_oss_agent_contract(tmp_path: Path) -> None:
    spec = EngineeringTaskSpec(
        id="oss-metric-regression",
        title="Pinned repository regression",
        prompt="Repair the selected behavior without changing protected files.",
        family="model-training-infrastructure",
        source=RepositorySource(
            clone_url="https://github.com/example/model-training.git",
            base_revision="0123456789abcdef0123456789abcdef01234567",
        ),
        tests=(
            CommandSpec(
                argv=("python", "-m", "pytest", "tests/test_regression.py"),
                cwd_base="evaluator",
            ),
        ),
        protected_paths=(ProtectedPath("src/trainer.py", "a" * 64),),
        limits=ResourceLimits(wall_time_s=30, cpu_time_s=20, memory_mb=1024),
        metadata={
            "editable_files": ["src/model/trainer.py"],
            "split": "development",
        },
    )
    spec_path = tmp_path / "task.json"
    spec.write(spec_path)

    contract = load_agent_task(tmp_path)

    assert contract.schema == "engineering_task_spec"
    assert contract.task_id == spec.id
    assert contract.base_revision == spec.source.base_revision
    assert contract.split == "development"
    assert contract.allowed_files == ("src/model/trainer.py",)
    assert contract.prompt_text == spec.prompt
    assert "prompt_sha256" in contract.input_hashes
    assert all(not path.startswith("evaluator/") for path in contract.allowed_files)
    assert_split_member(contract, {"development": [spec.id], "heldout": []}, "development")
    with pytest.raises(ValueError, match="outside the frozen train split"):
        assert_train_only(contract, {"train": [spec.id], "heldout": []})
