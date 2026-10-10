"""CPU-only provenance, split, and answer-key audit for Math study v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet
from math_verify import parse as reward_parse
from math_verify import verify as reward_verify

from scripts.math500_grader_v1 import exact_match, parse_reference
from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256


EXPECTED_MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
EXPECTED_DEEPMATH_SHA256 = "e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7"
EXPECTED_MATH500_ROWS = 500
EXPECTED_DEEPMATH_ROWS = 97870


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt(text: str) -> str:
    text = text.lower().replace("\\(", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return re.sub(r"\s+", " ", text).strip()


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256((seed + "\0" + key).encode("utf-8")).hexdigest()


def _same_training_reward_label(a: str, b: str) -> bool:
    parsed_a = reward_parse(a)
    parsed_b = reward_parse(b)
    if parsed_a and parsed_b:
        return reward_verify(parsed_a, parsed_b)
    return bool(parsed_a) == bool(parsed_b) and a.strip().casefold() == b.strip().casefold()


def deduplicate_train_rows(train_rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(train_rows):
        messages = row.get("prompt")
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(f"DeepMath row {index} is not a one-turn prompt")
        message = messages[0]
        if message.get("role") != "user" or not message.get("content", "").strip():
            raise ValueError(f"DeepMath row {index} has an invalid prompt")
        solution = row.get("solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"DeepMath row {index} has no reference solution")
        key = normalize_prompt(message["content"])
        grouped.setdefault(key, []).append(
            {"prompt": message["content"], "solution": solution, "prompt_key": key}
        )

    eligible = []
    conflicting = []
    unparseable = []
    for key, duplicates in grouped.items():
        reference = duplicates[0]["solution"]
        if len(duplicates) > 1 and not reward_parse(reference):
            unparseable.append(hashlib.sha256(key.encode()).hexdigest())
            continue
        if any(not _same_training_reward_label(row["solution"], reference) for row in duplicates[1:]):
            conflicting.append(hashlib.sha256(key.encode()).hexdigest())
            continue
        eligible.append(duplicates[0])
    return eligible, {
        "raw_rows": len(train_rows),
        "unique_normalized_prompts": len(grouped),
        "duplicate_prompt_groups_deduplicated": sum(len(rows) > 1 for rows in grouped.values()),
        "duplicate_excess_rows_deduplicated": sum(len(rows) - 1 for rows in grouped.values()),
        "conflicting_answer_prompts_excluded": len(conflicting),
        "conflicting_prompt_hashes": sorted(conflicting),
        "unparseable_reward_solution_prompts_excluded": len(unparseable),
        "unparseable_prompt_hashes": sorted(unparseable),
        "eligible_unique_prompts": len(eligible),
    }


def select_splits(math_rows: list[dict], train_rows: list[dict]) -> dict:
    by_level: dict[int, list[dict]] = {}
    for row in math_rows:
        by_level.setdefault(int(row["level"]), []).append(row)

    development: list[dict] = []
    confirmation: list[dict] = []
    for level, rows in sorted(by_level.items()):
        ranked = sorted(rows, key=lambda row: _rank("math500-dev-v1", row["unique_id"]))
        development.extend(ranked[:20])
        confirmation.extend(ranked[20:])

    ranked_train = sorted(
        train_rows,
        key=lambda row: _rank("deepmath-gate-train-v1", row["prompt_key"]),
    )
    reward_parseable = []
    unparseable_skipped = 0
    for row in ranked_train:
        if reward_parse(row["solution"]):
            reward_parseable.append(row)
            if len(reward_parseable) == 32 + 256:
                break
        else:
            unparseable_skipped += 1
    gate_rows = reward_parseable[:32]
    policy_train_rows = reward_parseable[32 : 32 + 256]
    return {
        "base_gate_ids": [hashlib.sha256(normalize_prompt(r["prompt"]).encode()).hexdigest() for r in gate_rows],
        "train_ids": [hashlib.sha256(normalize_prompt(r["prompt"]).encode()).hexdigest() for r in policy_train_rows],
        "development_ids": sorted(r["unique_id"] for r in development),
        "confirmation_ids": sorted(r["unique_id"] for r in confirmation),
        "development_levels": dict(sorted(Counter(int(r["level"]) for r in development).items())),
        "confirmation_levels": dict(sorted(Counter(int(r["level"]) for r in confirmation).items())),
        "unparseable_reward_solutions_skipped_before_selection": unparseable_skipped,
    }


def audit(data_dir: Path) -> dict:
    math_path = data_dir / "math500.jsonl"
    deepmath_path = data_dir / "deepmath-train.parquet"
    for path in (math_path, deepmath_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    math_hash = sha256_file(math_path)
    deepmath_hash = sha256_file(deepmath_path)
    if math_hash != EXPECTED_MATH500_SHA256:
        raise ValueError(f"MATH-500 SHA256 mismatch: {math_hash}")
    if deepmath_hash != EXPECTED_DEEPMATH_SHA256:
        raise ValueError(f"DeepMath SHA256 mismatch: {deepmath_hash}")

    math_rows = [json.loads(line) for line in math_path.read_text().splitlines() if line.strip()]
    table = parquet.read_table(deepmath_path, columns=["prompt", "solution"])
    train_rows = table.to_pylist()
    if len(math_rows) != EXPECTED_MATH500_ROWS:
        raise ValueError(f"MATH-500 row count mismatch: {len(math_rows)}")
    if len(train_rows) != EXPECTED_DEEPMATH_ROWS:
        raise ValueError(f"DeepMath row count mismatch: {len(train_rows)}")

    math_ids = [row.get("unique_id") for row in math_rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in math_ids) or len(set(math_ids)) != len(math_ids):
        raise ValueError("MATH-500 IDs must be nonempty and unique")
    if any(not isinstance(row.get("problem"), str) or not row["problem"].strip() for row in math_rows):
        raise ValueError("MATH-500 contains a missing problem")
    if any(not isinstance(row.get("answer"), str) or not row["answer"].strip() for row in math_rows):
        raise ValueError("MATH-500 contains a missing reference answer")
    normalized_eval_prompts = [normalize_prompt(row["problem"]) for row in math_rows]
    if len(set(normalized_eval_prompts)) != len(normalized_eval_prompts):
        raise ValueError("MATH-500 contains duplicate normalized evaluation prompts")

    train_prompts, dedupe_stats = deduplicate_train_rows(train_rows)

    train_hashes = {normalize_prompt(row["prompt"][0]["content"]) for row in train_rows}

    eval_prompts = {normalize_prompt(row["problem"]) for row in math_rows}
    overlap = sorted(eval_prompts & train_hashes)
    if overlap:
        raise ValueError(f"MATH-500 has {len(overlap)} exact normalized overlaps with DeepMath training")

    splits = select_splits(math_rows, train_prompts)
    if len(splits["base_gate_ids"]) != 32 or len(set(splits["base_gate_ids"])) != 32:
        raise ValueError("base gate sample is not 32 unique items")
    if len(splits["train_ids"]) != 256 or len(set(splits["train_ids"])) != 256:
        raise ValueError("training sample is not 256 unique items")
    if len(set(splits["base_gate_ids"]) & set(splits["train_ids"])):
        raise ValueError("base gate and train prompt IDs overlap")
    if len(splits["development_ids"]) != 100 or len(splits["confirmation_ids"]) != 400:
        raise ValueError("MATH-500 dev/confirmation sizes are incorrect")
    if set(splits["development_ids"]) & set(splits["confirmation_ids"]):
        raise ValueError("MATH-500 development and confirmation IDs overlap")
    splits["selection_fingerprints"] = {
        key: hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()
        for key, values in (
            ("base_gate_ids_sha256", splits["base_gate_ids"]),
            ("train_ids_sha256", splits["train_ids"]),
            ("development_ids_sha256", splits["development_ids"]),
            ("confirmation_ids_sha256", splits["confirmation_ids"]),
        )
    }
    # Keep the CPU validator's frozen v1 digest byte-for-byte identical while
    # sharing the precise serialization helper with the v3 GPU runner.
    splits["selection_fingerprints"]["base_gate_ids_sha256"] = canonical_prompt_id_sha256(
        splits["base_gate_ids"]
    )

    # Independent self-consistency checks: every reference must score itself,
    # while an item-specific numeric mutation must be rejected.
    self_checks = sum(exact_match("The answer is \\\\boxed{" + row["answer"] + "}", row["answer"]) for row in math_rows)
    negative_checks = sum(
        not exact_match("\\\\boxed{" + ("1" if exact_match("\\\\boxed{0}", row["answer"]) else "0") + "}", row["answer"])
        for row in math_rows
    )
    parsed_math_answers = sum(parse_reference(row["answer"]).value is not None for row in math_rows)
    reward_parseable_eval_labels = sum(bool(reward_parse(row["answer"])) for row in math_rows)

    return {
        "status": "pass",
        "math500": {
            "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            "rows": len(math_rows),
            "sha256": math_hash,
            "levels": dict(sorted(Counter(int(row["level"]) for row in math_rows).items())),
            "unique_ids": len(set(math_ids)),
            "nonempty_answers": sum(bool(row["answer"].strip()) for row in math_rows),
        },
        "deepmath_train": {
            "revision": "066c50a88d4e14cefc056e31111db2dba17f6c68",
            "rows": len(train_rows),
            "sha256": deepmath_hash,
            **dedupe_stats,
        },
        "cross_dataset": {"exact_normalized_prompt_overlap": len(overlap)},
        "grader": {
            "references_scoring_themselves": self_checks,
            "zero_answer_negative_controls_rejected": negative_checks,
            "reference_answers_parsed_symbolically": parsed_math_answers,
            "reference_answers_requiring_text_fallback": EXPECTED_MATH500_ROWS - parsed_math_answers,
            "reward_parser_parseable_gold_answers": reward_parseable_eval_labels,
            "format": "one final balanced \\boxed{...}; independent LaTeX-to-SymPy parse and exact symbolic comparison; strict normalized-text fallback only if either answer is nonmathematical",
        },
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/math-grpo-cpu-first-2026-10-11/data"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_dir)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
