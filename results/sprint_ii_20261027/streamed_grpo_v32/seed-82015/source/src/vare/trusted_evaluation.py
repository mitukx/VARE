"""Bounded CPU reference evaluator over *actual* registered checkpoint bytes.

Only a protocol-integration fixture: the scoring model is a frozen linear
classifier, not a neural-model evaluator. For hostile agents, run this worker
as a separate, less-accessible OS identity or VM with secret + dataset mounts
unavailable to the candidate and to its tools.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any

from .attestation import EvidenceTrustRoot
from .checkpoint_artifacts import TrustedCheckpointStore
from .evidence import canonical_json
from .types import EvaluationReport


def _strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _real(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("expected a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite evaluation value")
    return number


def evaluate_cpu_fixture(
    store: TrustedCheckpointStore,
    policy_id: str,
    frozen_tasks_path: Path,
    signer: EvidenceTrustRoot,
) -> EvaluationReport:
    """Compute scores from checkpoint bytes, then attest the *same* bytes.

    Unlike signing caller-submitted numbers, this trusted process owns the
    task pack, score calculation, checkpoint read, and HMAC key. The interpreter
    and all its dependencies must also be controlled by the evaluator owner.
    """
    if not signer.strict_checkpoint_binding:
        raise ValueError("trusted reference evaluator requires strict checkpoint binding")
    if frozen_tasks_path.is_symlink():
        raise ValueError("refuse symlinked evaluation set")
    if frozen_tasks_path.stat().st_size > 2_000_000:
        raise ValueError("oversized evaluation set")
    tasks_raw = frozen_tasks_path.read_bytes()
    tasks_sha = hashlib.sha256(tasks_raw).hexdigest()
    if tasks_sha != signer.evaluation_set_sha256:
        raise ValueError("evaluation set differs from operator-pinned content")

    checkpoint_raw, checkpoint_sha = store.read_verified(policy_id, max_read_bytes=100_000)
    model = _strict_json(checkpoint_raw)
    if type(model) is not dict or set(model) != {"schema_version", "weight", "bias"}:
        raise ValueError("invalid CPU fixture checkpoint")
    if type(model["schema_version"]) is not int or model["schema_version"] != 1:
        raise ValueError("invalid checkpoint schema version")
    weight, bias = _real(model["weight"]), _real(model["bias"])
    pack = _strict_json(tasks_raw)
    if type(pack) is not dict or set(pack) != {"schema_version", "tasks"}:
        raise ValueError("invalid evaluation set")
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 1:
        raise ValueError("invalid evaluation set version")
    rows = pack["tasks"]
    if type(rows) is not list or not 1 <= len(rows) <= 10000:
        raise ValueError("invalid held-out task count")
    per_task: dict[str, float] = {}
    by_family: dict[str, list[float]] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "x", "label", "family"}:
            raise ValueError("invalid task shape")
        task_id = row["id"]
        family = row["family"]
        label = row["label"]
        if (
            not isinstance(task_id, str) or not 1 <= len(task_id) <= 128
            or task_id in per_task
            or not isinstance(family, str) or not 1 <= len(family) <= 128
            or type(label) is not int or label not in (0, 1)
        ):
            raise ValueError("invalid or repeated task identity/label")
        x = _real(row["x"])
        score = float(int(weight * x + bias >= 0) == label)
        per_task[task_id] = score
        by_family.setdefault(family, []).append(score)

    # The second read catches same-host mutation while scoring; immutable
    # object storage and a separate UID/VM remain required against attackers.
    if store.digest(policy_id) != checkpoint_sha:
        raise ValueError("checkpoint changed while evaluating")
    report = EvaluationReport(
        policy_id=policy_id,
        primary=fmean(per_task.values()),
        slices={name: fmean(values) for name, values in by_family.items()},
        n=len(per_task),
        metadata={"per_task_scores": per_task},
    )
    return signer.attest(report, checkpoint_sha256=checkpoint_sha)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VARE trusted CPU evaluator protocol fixture")
    parser.add_argument("--checkpoint-store", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--heldout-set", required=True)
    parser.add_argument("--heldout-sha256", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--promotion-nonce", required=True)
    args = parser.parse_args(argv)
    key_file = Path(args.key_file)
    if key_file.is_symlink() or not key_file.is_file() or key_file.stat().st_size != 32:
        raise ValueError("signing key file must be a regular 32-byte trusted file")
    key = key_file.read_bytes()
    if (re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", args.heldout_sha256) is None):
        raise ValueError("invalid pinned SHA-256")
    signer = EvidenceTrustRoot(
        key=key,
        protocol_sha256=args.protocol_sha256,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run_id=args.run_id,
        evaluation_set_sha256=args.heldout_sha256,
        promotion_nonce=args.promotion_nonce,
        strict_checkpoint_binding=True,
    )
    report = evaluate_cpu_fixture(
        TrustedCheckpointStore(args.checkpoint_store, read_only=True),
        args.policy_id,
        Path(args.heldout_set),
        signer,
    )
    print(canonical_json(asdict(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
