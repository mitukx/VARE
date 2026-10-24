"""Independent offline audit for the verifier-drift GRPO bundle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v6/protocol.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare_pinned_imports(rvl_source: Path) -> None:
    """Load audited code from the frozen worktree, never an editable install."""
    import site
    for entry in site.getsitepackages():
        if entry not in sys.path:
            sys.path.append(entry)
    for finder in list(sys.meta_path):
        mapping = getattr(finder, "MAPPING", None)
        if isinstance(mapping, dict) and ({"vare", "rvl_systems"} & set(mapping)):
            sys.meta_path.remove(finder)
    for name in ("vare", "rvl_systems"):
        module = sys.modules.get(name)
        if module is not None:
            expected = (ROOT / "src/vare/__init__.py" if name == "vare"
                        else rvl_source / "src/rvl_systems/__init__.py")
            require(Path(module.__file__).resolve() == expected.resolve(),
                    f"preloaded {name} came from outside pinned source")
    sys.path.insert(0, str(rvl_source / "src"))
    sys.path.insert(0, str(ROOT / "src"))


def verify_import_origins(rvl_source: Path) -> None:
    from importlib import import_module
    expected = {
        "vare": ROOT / "src/vare/__init__.py",
        "vare.trajectory_store": ROOT / "src/vare/trajectory_store.py",
        "vare.trajectory_admission": ROOT / "src/vare/trajectory_admission.py",
        "vare.trajectory_learner": ROOT / "src/vare/trajectory_learner.py",
        "vare.promotion": ROOT / "src/vare/promotion.py",
        "rvl_systems": rvl_source / "src/rvl_systems/__init__.py",
        "rvl_systems.hf_backend": rvl_source / "src/rvl_systems/hf_backend.py",
        "rvl_systems.hf_trainer": rvl_source / "src/rvl_systems/hf_trainer.py",
        "rvl_systems.grpo": rvl_source / "src/rvl_systems/grpo.py",
    }
    for name, path in expected.items():
        module = import_module(name)
        require(Path(module.__file__).resolve() == path.resolve(),
                f"import origin mismatch for {name}: {module.__file__}")


def trusted_oracle(version: int, response: str) -> int:
    """Independent literal implementation of the frozen synthetic rubrics."""
    if version == 0:
        return int(any(character in response for character in ".!?"))
    if version == 1:
        return int(re.search(r"(?<![A-Za-z0-9_])ground(?![A-Za-z0-9_])",
                             response, re.IGNORECASE) is not None)
    raise ValueError(f"unknown verifier version {version}")


def max_error(actual, expected) -> float:
    import torch
    errors = []
    def walk(a, b):
        if isinstance(a, torch.Tensor):
            require(isinstance(b, torch.Tensor) and a.shape == b.shape,
                    "independent replay tensor shape mismatch")
            errors.append(float((a.detach().cpu() - b.detach().cpu()).abs().max())
                          if a.numel() else 0.0)
        elif isinstance(a, dict):
            require(isinstance(b, dict) and a.keys() == b.keys(),
                    "independent replay state keys mismatch")
            for key in a:
                walk(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            require(isinstance(b, type(a)) and len(a) == len(b),
                    "independent replay sequence mismatch")
            for x, y in zip(a, b):
                walk(x, y)
        else:
            require(a == b, "independent replay scalar mismatch")
    walk(actual, expected)
    return max(errors, default=0.0)


def verify_manifest(run: Path) -> None:
    manifest = json.loads((run / "SHA256.json").read_text())
    for relative, digest in manifest.items():
        path = run / relative
        require(path.is_file() and sha(path) == digest,
                f"artifact digest mismatch: {relative}")


def verify_groups(run: Path, seed: int) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    require(summary["seed"] == seed, "seed identity mismatch")
    groups = {arm: json.loads((run / arm / "rollouts.json").read_text())
              for arm in ("sync", "naive_async", "vare_async")}
    expected = {
        "sync": {"audit": [1] * 8, "fresh": [1] * 8},
        "naive_async": {"stale": [0] * 4 + [1] * 4, "fresh": [1] * 8},
        "vare_async": {"stale": [0] * 4 + [1] * 4, "fresh": [1] * 8},
    }
    scores = {}
    for arm, expected_groups in expected.items():
        require(set(groups[arm]) == set(expected_groups),
                f"unexpected group set in {arm}")
        for group_id, expected_versions in expected_groups.items():
            group = groups[arm][group_id]
            require(len(group["members"]) == 8, f"wrong group size: {arm}/{group_id}")
            require(group["versions"] == expected_versions,
                    f"verifier drift schedule differs: {arm}/{group_id}")
            independently_scored = []
            for member_index, member in enumerate(group["members"]):
                attempt = member["attempt"]
                gen = attempt["metadata"]["rvl_generation"]
                response = attempt["output"]
                require(response == gen["response"], "response/token trace mismatch")
                token_meta = gen["metadata"]
                require(len(token_meta["response_token_ids"]) == gen["token_count"]
                        == len(token_meta["response_token_logprobs"]),
                        "token budget/logprob alignment mismatch")
                require(member["verifier_version"] == expected_versions[member_index],
                        "member verifier identity mismatch")
                expected_score = trusted_oracle(member["verifier_version"], response)
                require(float(member["score"]) == float(expected_score),
                        "recorded reward differs from independent verifier")
                independently_scored.append(expected_score)
            require(independently_scored == group["scores"], "group score summary mismatch")
            require(len(set(independently_scored)) > 1,
                    f"degenerate group retained in results: {arm}/{group_id}")
            scores[f"{arm}/{group_id}"] = independently_scored
    # The synchronously-scored and VARE fresh groups are the same seeded model
    # behavior and must match token-for-token before their identical update.
    sync_fresh = groups["sync"]["fresh"]["members"]
    vare_fresh = groups["vare_async"]["fresh"]["members"]
    for left, right in zip(sync_fresh, vare_fresh, strict=True):
        a = left["attempt"]["metadata"]["rvl_generation"]
        b = right["attempt"]["metadata"]["rvl_generation"]
        require(a["response"] == b["response"]
                and a["metadata"]["response_token_ids"] == b["metadata"]["response_token_ids"]
                and a["metadata"]["response_token_logprobs"] == b["metadata"]["response_token_logprobs"],
                "sync and VARE fresh matched trajectories differ")
    disagreements = {}
    for arm in ("naive_async", "vare_async"):
        false_accepts = false_rejects = 0
        for member in groups[arm]["stale"]["members"]:
            recorded = int(member["score"])
            authoritative = trusted_oracle(1, member["attempt"]["output"])
            false_accepts += int(recorded == 1 and authoritative == 0)
            false_rejects += int(recorded == 0 and authoritative == 1)
        disagreements[arm] = {
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "total_disagreements": false_accepts + false_rejects,
        }
        require(summary["arms"][arm]["stale_reward_disagreements"] == disagreements[arm],
                f"recorded reward disagreement summary differs from trusted oracle: {arm}")
    return {"summary": summary, "groups": groups, "scores": scores,
            "stale_reward_disagreements": disagreements}


def replay_update(run: Path, arm: str, group: dict, model_dir: Path,
                  rvl_source: Path) -> dict:
    """Recompute an optimizer transaction offline from saved behavior traces."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(rvl_source / "src"))
    import torch
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from vare.types import Attempt, Experience, Task, Verification
    from validate_grpo_e2e_cpu import reference_update

    torch.set_num_threads(1)
    backend = HFLocalBackend(str(model_dir), max_new_tokens=3,
                             device="cpu", precision="fp32")
    model = backend.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("transformer.wpe.")
                                 or name.startswith("transformer.ln_f."))
    config = HFTTrainerConfig(learning_rate=1e-4)
    trainer = HFCausalLMGRPOTrainer(model, config=config)
    initial = torch.load(run / arm / "checkpoints/policy-0.pt", weights_only=True)
    candidate = torch.load(run / arm / "checkpoints/policy-1.pt", weights_only=True)
    trainer.restore_training_state(initial)
    reference_model = copy.deepcopy(model)
    batch = []
    for index, member in enumerate(group["members"]):
        raw = member["attempt"]
        raw_task = raw["task"]
        task = Task(**raw_task)
        group_id = str(raw_task["id"])
        task.metadata = {**task.metadata, "vare_rollout_group": group_id,
                         "vare_rollout_group_size": 8}
        metadata = {**raw["metadata"], "vare_rollout_group": group_id,
                    "vare_rollout_group_size": 8,
                    "rvl_generation": copy.deepcopy(raw["metadata"]["rvl_generation"])}
        attempt = Attempt(
            task=task, output=raw["output"], policy_id=raw["policy_id"],
            policy_version=raw["policy_version"], created_step=index,
            logprob=raw["logprob"], latency_ms=raw["latency_ms"], metadata=metadata)
        verification = Verification(
            float(member["score"]), bool(member["score"]), 1.0,
            int(member["verifier_version"]), "landing", trusted=True)
        batch.append(Experience(attempt, verification, 0,
                                1 - verification.verifier_version, 0.0))
    oracle = reference_update(reference_model, initial, batch, config)
    model_error = max_error(candidate["model"], oracle["model"])
    optimizer_error = max_error(candidate["optimizer"], oracle["optimizer"])
    require(max(model_error, optimizer_error) < 2e-6,
            f"offline GRPO replay mismatch for {arm}: {model_error}/{optimizer_error}")
    return {"model_max_abs": model_error, "optimizer_max_abs": optimizer_error,
            "loss": oracle["loss"], "advantages": oracle["advantages"],
            "behavior_alignment_error": oracle["max_behavior_alignment_error"]}


def wilson(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_paired_sign_test(naive: list[int], vare: list[int]) -> dict:
    differences = [left - right for left, right in zip(naive, vare, strict=True)]
    non_ties = [difference for difference in differences if difference != 0]
    positive = sum(difference > 0 for difference in non_ties)
    n = len(non_ties)
    tail = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n) if n else 1.0
    return {"positive_pairs": positive, "negative_pairs": n - positive,
            "ties": len(differences) - n, "non_ties": n,
            "exact_one_sided_p_naive_greater": tail,
            "pairs": differences}


def exact_state_equal(left, right) -> bool:
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            exact_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            exact_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def audit_sigkill_recovery(source_run: Path, crash_run: Path) -> dict:
    """Independently inspect all five killed processes and reconstructed state."""
    import torch
    from vare.trajectory_store import audit_trajectory_store

    verify_manifest(crash_run)
    summary = json.loads((crash_run / "summary.json").read_text())
    seed = json.loads((source_run / "summary.json").read_text())["seed"]
    require(summary["seed"] == seed, "crash run seed differs from matched source run")
    require(Path(summary["source_run"]).resolve() == source_run.resolve(),
            "crash run source identity mismatch")
    require(len(summary["cutpoints"]) == 5 and summary["process_kills"] == 5,
            "all five declared learner processes were not killed")
    source_rollouts = source_run / "sync/rollouts.json"
    source_state = torch.load(source_run / "sync/checkpoints/policy-1.pt",
                              weights_only=True)
    cutpoints = []
    expected_attempts = {
        "rollout_complete_before_admission": 1,
        "admission_before_optimizer": 1,
        "optimizer_computed_before_commit": 2,
        "commit_before_acknowledgment": 1,
        "evaluation_before_promotion": 2,
    }
    for row in summary["cutpoints"]:
        name = row["cutpoint"]
        case = crash_run / name
        require(name in expected_attempts, f"unknown recovery cutpoint: {name}")
        proof = json.loads((case / "supervisor-proof.json").read_text())
        marker = json.loads((case / "cutpoint.json").read_text())
        worker = json.loads((case / "worker-summary.json").read_text())
        require(proof["cutpoint"] == marker["cutpoint"] == name,
                f"cutpoint marker mismatch: {name}")
        require(proof["signal_returncode"] == -9
                and proof["restart_returncode"] == 0
                and not proof["old_process_alive"],
                f"real SIGKILL/restart identity failed: {name}")
        require(sha(case / "rollouts.json") == sha(source_rollouts),
                f"durable rollout payload changed: {name}")
        admission = json.loads((case / "admission.json").read_text())
        require(admission["admitted"] and admission["members"] == 8,
                f"durable complete group failed admission: {name}")
        require(worker["phase"] == "acknowledged"
                and worker["effective_consumed_trajectories"] == 8
                and worker["committed_updates"] == 1,
                f"recovery did not finish one effective transaction: {name}")
        require(worker["optimizer_attempts"] == expected_attempts[name],
                f"optimizer recompute count differs from protocol: {name}")
        attempts = [json.loads(line) for line in
                    (case / "optimizer-attempts.jsonl").read_text().splitlines()]
        require(len(attempts) == expected_attempts[name]
                and all(item["optimizer_calls"] == 1 for item in attempts),
                f"actual optimizer step count differs from recovery contract: {name}")
        store_audit = audit_trajectory_store(case / "trajectory-snapshot.sqlite")
        require(store_audit.get("pass") is True,
                f"trajectory-store audit failed after recovery: {name}")
        require(worker["store_counts"].get("consumed", 0) == 8
                and worker["store_counts"].get("quarantined", 0) == 0,
                f"effective trajectory acknowledgment count differs: {name}")
        journal = worker["promotion_journal"]
        require(journal["active_policy_id"] == "policy-1"
                and journal["pending_tx_id"] is None,
                f"promotion journal is not clean after recovery: {name}")
        with sqlite3.connect(case / "promotion.sqlite") as db:
            transactions = db.execute(
                "SELECT tx_id,state FROM transactions ORDER BY tx_id").fetchall()
        require(len(transactions) == 1 and transactions[0][1] == "committed",
                f"recovery produced duplicate or noncommitted transactions: {name}")
        active = json.loads((case / "active.json").read_text())
        require(active == {"policy_id": "policy-1",
                           "checkpoint_sha256": journal["active_checkpoint_sha256"]},
                f"promotion pointer/journal mismatch: {name}")
        recovered = torch.load(case / "checkpoints/policy-1.pt", weights_only=True)
        state_fields = ("model", "optimizer", "cpu_rng", "module_modes")
        require(all(field in recovered and field in source_state for field in state_fields),
                f"checkpoint omitted required model/Adam/RNG state: {name}")
        exact_fields = {field: exact_state_equal(recovered[field], source_state[field])
                        for field in state_fields}
        require(all(exact_fields.values()),
                f"recovered training state differs exactly from sync reference: {name}/{exact_fields}")
        require(worker["peak_rss_bytes"] <= 8 * 1024**3,
                f"recovery process exceeded RSS limit: {name}")
        cutpoints.append({
            "cutpoint": name,
            "actual_sigkill": True,
            "restart_exit_code": proof["restart_returncode"],
            "recovery_wall_s": proof["recovery_wall_s"],
            "optimizer_attempts": len(attempts),
            "optimizer_calls": sum(item["optimizer_calls"] for item in attempts),
            "committed_transactions": len(transactions),
            "effective_consumed_trajectories": worker["effective_consumed_trajectories"],
            "exact_state_match": exact_fields,
            "candidate_sha256": sha(case / "checkpoints/policy-1.pt"),
            "trajectory_store_audit": store_audit,
            "peak_rss_bytes": worker["peak_rss_bytes"],
        })
    return {
        "status": "PASS",
        "cutpoints": cutpoints,
        "all_model_adam_rng_states_exact": all(
            all(row["exact_state_match"].values()) for row in cutpoints),
        "effective_consumed_trajectories_total_across_isolated_cases": sum(
            row["effective_consumed_trajectories"] for row in cutpoints),
        "scope": "Five independent SIGKILL transactions reuse one held-fixed, independently verified TinyStories rollout group. Exact restoration is established by fieldwise comparison to the matched synchronous candidate. This is crash-recovery evidence, not independent task generalization.",
    }


def main() -> None:
    require(sys.flags.no_site == 1,
            "launch the independent auditor with python -S to disable site startup hooks")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="one seed bundle; repeat for paired seeds")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--crash-run", type=Path,
                        help="optional five-cutpoint actual-process SIGKILL bundle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    require(sha(Path(__file__)) == protocol["auditor_sha256"],
            "auditor source differs from frozen protocol")
    require(sha(ROOT / "scripts/benchmark_verifier_consistent_async_grpo.py")
            == protocol["runner_sha256"], "runner source differs from frozen protocol")
    require(sha(ROOT / "scripts/benchmark_verifier_consistent_grpo_sigkill.py")
            == protocol["crash_runner_sha256"], "crash runner differs from frozen protocol")
    for relative, digest in protocol["source_sha256"].items():
        require(sha(ROOT / relative) == digest,
                f"VARE source differs from frozen protocol: {relative}")
    require(sha(args.rvl_source / "src/rvl_systems/hf_backend.py")
            == protocol["budgets"]["rvl_patched_backend_sha256"],
            "pinned RVL backend mismatch")
    prepare_pinned_imports(args.rvl_source)
    verify_import_origins(args.rvl_source)
    seeds = [json.loads((path / "summary.json").read_text())["seed"] for path in args.run]
    require(len(set(seeds)) == len(seeds), "duplicate seed bundle")
    if not args.allow_partial:
        require(sorted(seeds) == sorted(protocol["budgets"]["seeds"]),
                "complete prespecified seed set required")
    per_seed = []
    for run, seed in zip(args.run, seeds, strict=True):
        verify_manifest(run)
        checked = verify_groups(run, seed)
        summary = checked["summary"]
        arms = summary["arms"]
        require(arms["naive_async"].get("training_path")
                == protocol["naive_control"]["training_path"],
                "naive arm did not use the preregistered provenance-unaware RVL path")
        require(all(arms[name]["rollouts"] == 16
                    and arms[name]["verifier_calls"] == 16
                    and arms[name]["optimizer_calls"] == 1
                    for name in ("sync", "naive_async", "vare_async")),
                "matched rollout/verifier/optimizer budget mismatch")
        require(len({arms[name]["initial_model_tensor_sha256"]
                     for name in arms}) == 1,
                "arms did not share identical starting model tensors")
        stale_false_accepts = checked["stale_reward_disagreements"][
            "naive_async"]["false_accepts"]
        vare = arms["vare_async"]
        require(vare["trajectory_counts"].get("quarantined", 0) == 8
                and vare["trajectory_counts"].get("consumed", 0) == 8,
                "VARE did not quarantine the stale group and consume the fresh group")
        require(all(row["state"] == "quarantined"
                    and row["reason"] == "quarantined"
                    for row in vare["admission_decisions"]),
                "VARE stale group lacks explicit quarantine evidence")
        require(arms["sync"]["journal"]["active_policy_id"] != "policy-0"
                and vare["journal"]["active_policy_id"] != "policy-0"
                and arms["naive_async"]["journal"]["active_policy_id"] != "policy-0",
                "one or more model candidates were not committed")
        require(arms["naive_async"]["invalid_committed_updates"] == int(
                    checked["stale_reward_disagreements"]["naive_async"][
                        "total_disagreements"] > 0)
                and vare["invalid_committed_updates"] == 0,
                "independent reward oracle disagrees with invalid update counts")
        require(arms["naive_async"]["naive_promotion"]["accepted"]
                and not arms["naive_async"]["strict_stale_promotion"]["accepted"],
                "stale evaluator control decision differs from protocol")
        for arm, expected_version in (("sync", 1), ("vare_async", 1)):
            reports = json.loads((run / arm / "evaluations.json").read_text())
            require(len(reports) == 2 and all(
                entry["report"]["metadata"]["verifier_version"] == expected_version
                for entry in reports), "current-version evaluator provenance missing")
        stale_reports = json.loads((run / "naive_async/evaluations.json").read_text())
        require(len(stale_reports) == 2 and all(
            entry["report"]["metadata"]["verifier_version"] == 0
            for entry in stale_reports), "naive stale evaluation metadata missing")
        # Recompute one actual update from each saved raw batch and verify the
        # final model and Adam state against an offline numerical reference.
        replay = {}
        selected = {"sync": "fresh", "naive_async": "stale", "vare_async": "fresh"}
        for arm, group_id in selected.items():
            replay[arm] = replay_update(run, arm,
                checked["groups"][arm][group_id], args.model_dir, args.rvl_source)
            update_records = arms[arm]["updates"]
            require(len(update_records) == 1
                    and update_records[0]["optimizer_calls"] == 1
                    and update_records[0]["batch_size"] == 8
                    and max(update_records[0]["errors"].values()) < 2e-6,
                    f"runner numerical update record invalid: {arm}")
        sync_state = __import__("torch").load(run / "sync/checkpoints/policy-1.pt", weights_only=True)
        vare_state = __import__("torch").load(run / "vare_async/checkpoints/policy-1.pt", weights_only=True)
        require(max_error(sync_state["model"], vare_state["model"]) == 0.0,
                "synchronous and VARE fresh-group model commits diverged")
        # Independently audit the VARE trajectory database snapshot.
        sys.path.insert(0, str(ROOT / "src"))
        from vare.trajectory_store import audit_trajectory_store
        store_audit = audit_trajectory_store(run / "vare_async/snapshot.sqlite")
        require(store_audit.get("pass") is True,
                "offline VARE trajectory-store audit failed")
        per_seed.append({"seed": seed, "stale_false_accepts": stale_false_accepts,
                         "stale_reward_disagreements": checked[
                             "stale_reward_disagreements"],
                         "invalid_commit_naive": arms["naive_async"]["invalid_committed_updates"],
                         "invalid_commit_vare": vare["invalid_committed_updates"],
                         "stale_eval_naive_accept": arms["naive_async"]["naive_promotion"]["accepted"],
                         "stale_eval_strict_accept": arms["naive_async"]["strict_stale_promotion"]["accepted"],
                         "group_scores": checked["scores"], "update_replay": replay,
                         "trajectory_store_audit": store_audit})
    invalid_naive = sum(row["invalid_commit_naive"] for row in per_seed)
    invalid_vare = sum(row["invalid_commit_vare"] for row in per_seed)
    paired_sign = exact_paired_sign_test(
        [row["invalid_commit_naive"] for row in per_seed],
        [row["invalid_commit_vare"] for row in per_seed])
    pilot_signal = bool(per_seed and per_seed[0]["seed"] == protocol["stopping"]["pilot_seed"]
                        and per_seed[0]["stale_reward_disagreements"][
                            "naive_async"]["total_disagreements"] > 0)
    report = {
        "protocol_sha256": sha(PROTOCOL),
        "auditor_sha256": sha(Path(__file__)),
        "runs": [str(path) for path in args.run],
        "seeds": per_seed,
        "primary": {
            "invalid_committed_updates": {
                "naive_async": {"successes": invalid_naive, "trials": len(per_seed),
                                "wilson_95": wilson(invalid_naive, len(per_seed))},
                "vare_async": {"successes": invalid_vare, "trials": len(per_seed),
                               "wilson_95": wilson(invalid_vare, len(per_seed))},
            },
            "paired_sign_test": paired_sign,
            "multiplicity": "One prespecified primary contrast; no adjustment across secondary systems metrics.",
            "scope_limit": "Seed bundles are the independent units; this selected synthetic systems fixture does not estimate benchmark-wide contamination or coding capability.",
        },
        "decision": ("PILOT_SIGNAL" if pilot_signal else "NO_GO_NO_FALSE_ACCEPTANCE"),
        "expand_after_pilot": pilot_signal,
        "full_seed_set": sorted(seeds) == sorted(protocol["budgets"]["seeds"]),
    }
    if args.crash_run is not None:
        require(len(args.run) >= 1, "crash audit requires its matched model run")
        source = next((path for path in args.run
                       if json.loads((path / "summary.json").read_text())["seed"]
                       == json.loads((args.crash_run / "summary.json").read_text())["seed"]), None)
        require(source is not None, "no matched seed run supplied for crash bundle")
        report["crash_recovery"] = audit_sigkill_recovery(source, args.crash_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
