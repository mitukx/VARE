"""Score a frozen MATH-500 prediction file with the independent study grader."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.math500_grader_v2 import exact_match, has_valid_final_box
from scripts.validate_math500_study_v1 import EXPECTED_MATH500_SHA256, sha256_file


def evaluate(data_dir: Path, split_manifest: Path, predictions_path: Path, split: str) -> dict:
    if split not in {"development", "confirmation"}:
        raise ValueError("split must be development or confirmation")
    dataset_path = data_dir / "math500.jsonl"
    if sha256_file(dataset_path) != EXPECTED_MATH500_SHA256:
        raise ValueError("MATH-500 dataset hash does not match the frozen revision")
    manifest = json.loads(split_manifest.read_text())
    if manifest.get("status") != "pass" or manifest["math500"]["sha256"] != EXPECTED_MATH500_SHA256:
        raise ValueError("split manifest is not a passing audit for the pinned MATH-500 file")

    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    expected_ids = set(manifest["splits"][split + "_ids"])
    selected = {row["unique_id"]: row for row in rows if row["unique_id"] in expected_ids}
    if set(selected) != expected_ids:
        raise ValueError("split manifest references missing MATH-500 IDs")

    predictions = {}
    for line_number, line in enumerate(predictions_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        row_id = item.get("unique_id")
        completion = item.get("completion")
        if not isinstance(row_id, str) or not isinstance(completion, str):
            raise ValueError(f"prediction line {line_number} requires string unique_id and completion")
        if row_id in predictions:
            raise ValueError(f"duplicate prediction ID: {row_id}")
        if row_id not in expected_ids:
            raise ValueError(f"prediction ID is outside the locked {split} split: {row_id}")
        predictions[row_id] = completion
    missing = sorted(expected_ids - predictions.keys())
    if missing:
        raise ValueError(f"prediction file is incomplete: missing {len(missing)} rows")

    by_level: dict[str, Counter] = {}
    scored_rows = []
    for row_id in sorted(expected_ids):
        row = selected[row_id]
        completion = predictions[row_id]
        valid = has_valid_final_box(completion)
        correct = exact_match(completion, row["answer"])
        level = str(row["level"])
        stats = by_level.setdefault(level, Counter())
        stats["n"] += 1
        stats["valid_final_box"] += int(valid)
        stats["correct"] += int(correct)
        scored_rows.append({"unique_id": row_id, "level": int(level), "valid_final_box": valid, "correct": correct})

    n = len(scored_rows)
    valid_n = sum(row["valid_final_box"] for row in scored_rows)
    correct_n = sum(row["correct"] for row in scored_rows)
    return {
        "status": "pass",
        "split": split,
        "n": n,
        "correct": correct_n,
        "exact_match": correct_n / n if n else None,
        "valid_final_box": valid_n,
        "valid_final_box_rate": valid_n / n if n else None,
        "level_metrics": {
            level: {
                "n": count["n"],
                "correct": count["correct"],
                "exact_match": count["correct"] / count["n"],
                "valid_final_box": count["valid_final_box"],
            }
            for level, count in sorted(by_level.items(), key=lambda pair: int(pair[0]))
        },
        "dataset_sha256": EXPECTED_MATH500_SHA256,
        "split_manifest_sha256": sha256_file(split_manifest),
        "predictions_sha256": sha256_file(predictions_path),
        "scored_rows": scored_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "confirmation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data_dir, args.split_manifest, args.predictions, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "split", "n", "correct", "exact_match", "valid_final_box_rate")}, indent=2))


if __name__ == "__main__":
    main()
