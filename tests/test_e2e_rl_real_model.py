"""Independent falsifiers for the real-model systems benchmark.

Default CI stays dependency-free. The optional model cases are run in the
isolated benchmark job with a revision-pinned checkpoint.
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from scripts import benchmark_e2e_rl_real_model as bench


def test_task_namespaces_and_percentiles():
    train = bench.tasks_for(1701, 32)
    evaluation = bench.tasks_for(1701, 32, True)
    assert train == bench.tasks_for(1701, 32)
    assert not ({t.prompt for t in train} & {t.prompt for t in evaluation})
    assert not ({t.id for t in train} & {t.id for t in evaluation})
    assert bench.percentile([1, 2, 3], 0.95) == pytest.approx(2.9)


@pytest.fixture(scope="module")
def backend():
    location = os.environ.get("VARE_E2E_MODEL_DIR")
    if not location:
        pytest.skip("set VARE_E2E_MODEL_DIR for actual model regression tests")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("psutil")
    # This setup is in a fresh pytest process, before inter-op work.
    bench.torch_setup()
    model_dir = Path(location).resolve()
    pool, _ = bench.pool_start(model_dir, 1)
    evaluation, _ = bench.pool_start(model_dir, 2)
    yield model_dir, pool, evaluation
    pool.shutdown(wait=True, cancel_futures=True)
    evaluation.shutdown(wait=True, cancel_futures=True)


def run_case(backend, tmp_path, mode, fault="none"):
    model_dir, pool, evaluation = backend
    return asyncio.run(bench.execute_case(model_dir, tmp_path / f"{mode}-{fault}", pool,
                                         evaluation, mode, 1701, 1, 8, 8, fault))


def test_same_real_update_and_checkpoint_under_optional_guards(backend, tmp_path):
    rows = [run_case(backend, tmp_path, mode) for mode in ("minimal", "guarded")]
    updates = [json.loads((tmp_path / f"{mode}-none/optimizer_update.json").read_text())
               for mode in ("minimal", "guarded")]
    for row, update in zip(rows, updates):
        assert row["optimizer_steps"] == 1
        assert row["state_integrity"] and row["promotion_correct"]
        assert update["parameters_before_sha256"] != update["parameters_after_sha256"]
        assert update["trainable_parameters"] == 2200
        assert len(update["rewards"]) == 8
    # Identical numerical updates demonstrate that the comparison did not
    # quietly change learning workload or algorithm when guards were enabled.
    assert updates[0] == updates[1]


@pytest.mark.parametrize("fault", ["reward_tamper", "replay_stale", "forged_eval", "checkpoint_corrupt"])
def test_real_backend_rejects_injected_unsound_evidence(backend, tmp_path, fault):
    minimal = run_case(backend, tmp_path, "minimal", fault)
    guarded = run_case(backend, tmp_path, "guarded", fault)
    assert bench.fault_oracle(minimal)
    assert bench.fault_oracle(guarded)
    assert minimal["optimizer_steps"] == 1
    assert guarded["active_policy"] == "policy-0"
    assert guarded["state_integrity"]


@pytest.mark.parametrize("fault", ["optimizer_exception", "evaluation_cancel", "publication_crash"])
def test_optimizer_and_publisher_failure_recover_exact_state(backend, tmp_path, fault):
    row = run_case(backend, tmp_path, "guarded", fault)
    assert bench.fault_oracle(row)
    assert row["optimizer_steps"] == 1
    assert row["journal"]["pending_tx_id"] is None
    assert row["journal"]["round_owner_id"] is None
    if fault == "publication_crash":
        assert json.loads((tmp_path / "guarded-publication_crash/publisher_exit.json").read_text())["exitcode"] == 74
