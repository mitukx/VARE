"""Recompute retained metrics and optionally score actual checkpoint tensors.

HMAC keys are ephemeral benchmark trust roots; offline audit instead rechecks
the published bytes, arithmetic, independent scores and declared fault oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vare.checkpoint_artifacts import CheckpointIntegrityError, TrustedCheckpointStore


def check(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, model_dir=None):
    directory = Path(directory)
    sums = json.loads((directory / "SHA256SUMS.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS.json"}
    check(actual == set(sums), "artifact inventory mismatch")
    for name, expected in sums.items():
        check(digest(directory / name) == expected, f"changed artifact: {name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        check(digest(directory / "sources" / name) == expected, "source snapshot mismatch")
    rows = json.loads((directory / "progress.json").read_text())
    if model_dir:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        choices = [tokenizer.encode(" A")[0], tokenizer.encode(" B")[0]]
        for name, expected in manifest["model_files"].items():
            check(digest(Path(model_dir) / name) == expected, "model revision bytes differ")
    for row in rows:
        run = directory / row["directory"]
        trace = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
        check(row["rollout_tokens"] == sum(t["stage"] == "rollout" for t in trace), "token count mismatch")
        check(row["tasks_s"] == row["rollout_tokens"] / row["wall_s"], "throughput mismatch")
        check(row["tokens_s"] == row["tasks_s"], "one-token throughput mismatch")
        samples = [json.loads(line) for line in (run / "resources.jsonl").read_text().splitlines()]
        check(row["resources"]["sampled_tree_peak_rss_bytes"] == max(r["rss_bytes"] for r in samples), "RSS mismatch")
        expected_state = row["candidate_state_sha256"] if row["active_policy"] == "policy-1" else row["initial_state_sha256"]
        check(row["recovered_state_sha256"] == expected_state, "recovery state mismatch")
        train = json.loads((run / "training_tasks.json").read_text())
        evaluation = json.loads((run / "evaluation_tasks.json").read_text())
        check(not ({t["prompt"] for t in train} & {t["prompt"] for t in evaluation}), "training/evaluation overlap")
        if row["optimizer_steps"]:
            update = json.loads((run / "optimizer_update.json").read_text())
            check(update["parameters_before_sha256"] != update["parameters_after_sha256"], "mock/no-op update")
            check(update["trainable_parameters"] == 2200 and update["total_parameters"] == 102714, "model size mismatch")
        fault, guarded = row["fault"], row["mode"] == "guarded"
        if fault in ("policy_stale", "replay_stale", "reward_tamper"):
            check(row["optimizer_steps"] == (0 if guarded else 1), "unsafe replay reached optimizer")
        elif fault == "forged_eval":
            check(row["active_policy"] == ("policy-0" if guarded else "policy-1"), "forged promotion result")
        elif fault == "checkpoint_corrupt":
            check(bool(row["error"]) == guarded, "corrupt checkpoint not rejected")
        elif fault in ("evaluation_cancel", "optimizer_exception"):
            check(row["optimizer_steps"] == 1 and row["active_policy"] == "policy-0" and row["error"], "failure rollback mismatch")
        elif fault == "worker_crash":
            check(row["optimizer_steps"] == 0 and row["worker_restart_s"] > 0 and row["error"], "worker restart mismatch")
        elif fault == "publication_crash":
            check(row["active_policy"] == "policy-1" and row["error"], "publication recovery mismatch")
            check(json.loads((run / "publisher_exit.json").read_text())["exitcode"] == 74, "publisher did not exit")
        else:
            check(row["error"] is None and row["optimizer_steps"] == 1, "normal case incomplete")
        if guarded:
            check(row["journal"]["pending_tx_id"] is None and row["journal"]["round_owner_id"] is None, "unreconciled journal")
        # Recompute the normal paired workload equivalence from raw updates.
        if fault == "none":
            other_mode = "minimal" if guarded else "guarded"
            other = directory / f"w{row['workers']}-s{row['seed']}-{other_mode}" / "optimizer_update.json"
            check(json.loads((run / "optimizer_update.json").read_text()) == json.loads(other.read_text()), "unmatched learning workload")
        for report_path in run.glob("evaluation-policy-*.json"):
            record = json.loads(report_path.read_text())
            check(len(record["scores"]) == len(evaluation), "evaluation cardinality mismatch")
            if fault != "forged_eval" or record["policy_id"] == "policy-0":
                check(statistics.fmean(record["scores"].values()) == record["report"]["primary"], "evaluation mean mismatch")
            if model_dir:
                if guarded:
                    raw, _ = TrustedCheckpointStore(run / "checkpoints", read_only=True).read_verified(record["policy_id"])
                else:
                    raw = (run / f"{record['policy_id']}.pt").read_bytes()
                state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"])
                with torch.inference_mode():
                    for task in evaluation:
                        ids = torch.tensor([tokenizer.encode(task["prompt"])])
                        probability = float(model(ids).logits[0, -1, choices].softmax(-1)[0])
                        check(abs(probability - record["scores"][task["id"]]) < 1e-7, "checkpoint evaluation does not replay")
        if guarded and fault == "checkpoint_corrupt":
            try:
                TrustedCheckpointStore(run / "checkpoints", read_only=True).digest("policy-1")
            except CheckpointIntegrityError:
                pass
            else:
                raise ValueError("corrupted checkpoint unexpectedly verified")
    aggregate = json.loads((directory / "summary.json").read_text())
    check(aggregate["case_count"] == aggregate["oracle_passes"] == len(rows), "campaign pass count mismatch")
    for group in aggregate["normal"]:
        matched = [r for r in rows if r["mode"] == group["mode"] and r["workers"] == group["workers"] and r["fault"] == "none"]
        check(group["n"] == len(matched), "repetition count mismatch")
        for metric, record in group["metrics"].items():
            values = [r[metric] for r in matched]
            check(record["mean"] == statistics.fmean(values) and record["min"] == min(values) and record["max"] == max(values), "aggregate mismatch")
            if len(values) > 1:
                check(record["sd"] == statistics.stdev(values), "variance mismatch")
    return {"pass": True, "cases": len(rows), "checkpoint_scores_replayed": bool(model_dir), "artifacts_hashed": len(sums)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.model_dir), sort_keys=True))
