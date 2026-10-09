#!/usr/bin/env python3
"""Data selection and metric helpers for the frozen HH reward-model study."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MARKER = "\n\nAssistant:"
HASH_DOMAIN = b"vare-hh-helpful-context-v1\0"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "audit.json"):
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def parse_pair(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    chosen_at = chosen.rfind(MARKER)
    rejected_at = rejected.rfind(MARKER)
    if chosen_at < 0 or rejected_at < 0:
        return None
    chosen_context = chosen[:chosen_at + len(MARKER)]
    rejected_context = rejected[:rejected_at + len(MARKER)]
    chosen_response = chosen[chosen_at + len(MARKER):]
    rejected_response = rejected[rejected_at + len(MARKER):]
    if chosen_context != rejected_context or not chosen_response or not rejected_response:
        return None
    return {"context": chosen_context, "chosen": chosen, "rejected": rejected,
            "chosen_response": chosen_response, "rejected_response": rejected_response}


def context_hash(context: str) -> str:
    return sha256_bytes(HASH_DOMAIN + context.encode("utf-8"))


def tokenize_pair(pair: Dict[str, str], tokenizer: Any, maximum: int) -> Optional[Dict[str, Any]]:
    """Return eligibility metadata without retaining token IDs in result artifacts."""
    try:
        chosen = tokenizer(pair["chosen"], add_special_tokens=True, truncation=False,
                           return_offsets_mapping=True)
        rejected = tokenizer(pair["rejected"], add_special_tokens=True, truncation=False,
                             return_offsets_mapping=True)
        chosen_offsets = chosen["offset_mapping"]
        rejected_offsets = rejected["offset_mapping"]
        chosen_start = len(pair["context"])
        rejected_start = len(pair["context"])
        chosen_positions = [i for i, (start, end) in enumerate(chosen_offsets)
                            if end > start and start >= chosen_start]
        rejected_positions = [i for i, (start, end) in enumerate(rejected_offsets)
                              if end > start and start >= rejected_start]
        chosen_ids = chosen["input_ids"]
        rejected_ids = rejected["input_ids"]
        if len(chosen_ids) > maximum or len(rejected_ids) > maximum:
            return None
        if not chosen_positions or not rejected_positions:
            return None
        return {"chosen_ids": chosen_ids, "rejected_ids": rejected_ids,
                "chosen_feature_positions": chosen_positions,
                "rejected_feature_positions": rejected_positions,
                "chosen_tokens": len(chosen_positions), "rejected_tokens": len(rejected_positions)}
    except (ValueError, KeyError, TypeError, OverflowError):
        return None


def eligible_pair(row: Dict[str, Any], tokenizer: Any, maximum: int) -> Optional[Dict[str, Any]]:
    pair = parse_pair(row)
    if pair is None:
        return None
    tokenized = tokenize_pair(pair, tokenizer, maximum)
    if tokenized is None:
        return None
    return {**pair, **tokenized, "context_hash": context_hash(pair["context"])}


def hash_rank_key(context: str) -> Tuple[bytes, bytes]:
    encoded = context.encode("utf-8")
    return hashlib.sha256(HASH_DOMAIN + encoded).digest(), encoded


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def pair_accuracy(margin: float) -> float:
    return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5


def pair_nll(margin: float) -> float:
    return max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))


def metric_summary(margins: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in margins]
    if not values:
        raise ValueError("cannot compute metrics for an empty pair set")
    probs = [sigmoid(value) for value in values]
    accuracy = [pair_accuracy(value) for value in values]
    nll = [pair_nll(value) for value in values]
    brier = [(probability - 1.0) ** 2 for probability in probs]
    ece = 0.0
    for bin_index in range(10):
        low = bin_index / 10.0
        high = (bin_index + 1) / 10.0
        indexes = [i for i, probability in enumerate(probs)
                   if low <= probability < high or (bin_index == 9 and probability == 1.0)]
        if indexes:
            confidence = sum(probs[i] for i in indexes) / len(indexes)
            observed = 1.0  # Every retained HH pair records chosen as the observed preference.
            ece += (len(indexes) / len(values)) * abs(confidence - observed)
    return {"pairwise_accuracy": sum(accuracy) / len(accuracy),
            "logistic_nll": sum(nll) / len(nll),
            "brier_score": sum(brier) / len(brier),
            "expected_calibration_error": ece}


def paired_bootstrap_interval(deltas: List[float], seed: int, resamples: int) -> List[float]:
    import numpy as np
    if not deltas:
        raise ValueError("cannot bootstrap an empty paired metric")
    array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975], method="linear")]


def load_locked_protocol(spec_path: Path, lock_path: Path) -> Tuple[Dict[str, Any], str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_hash = lock.pop("sha256", None)
    observed_hash = sha256_bytes(canonical(spec))
    if locked_hash != observed_hash or canonical(lock) != canonical(spec):
        raise ValueError("HH reward-model protocol differs from its frozen lock")
    return spec, observed_hash
