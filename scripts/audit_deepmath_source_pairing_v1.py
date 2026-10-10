"""CPU-only exact-overlap and source-label audit for the frozen MATH-500 pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet

from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256,
    EXPECTED_MATH500_SHA256,
    deduplicate_train_rows,
    normalize_prompt,
    select_splits,
    sha256_file,
)


ORIGINAL_SOURCE_SHA256 = {
    "train-00000-of-00010.parquet": "d6412432f30425e848a224dc641e681eb1ed51b970d52536eda7daefc01d8c8b",
    "train-00001-of-00010.parquet": "eef9d3012456239eb0f4cd462ac7bebb7d6c4f675c41329c680ef8a506ded512",
    "train-00002-of-00010.parquet": "20e4dc6527d94c2057a9b727f8b58994395de602e284b3294f7d2451f75d681a",
    "train-00003-of-00010.parquet": "c5b20135f93e7890da191973ad77d88e67c95144ccc169ec592489f044e7c38a",
    "train-00004-of-00010.parquet": "4153e531d78ca278d2e12e6d08099b66126b26b99ca6a92b3d6f95c156818749",
    "train-00005-of-00010.parquet": "110586bdc6f35b0434bccbd582f1e97e8328da0752ccd36590a50178885f0360",
    "train-00006-of-00010.parquet": "e39c00ed42a6a1af74ddc042840884a59adcb9139a23f535e399ddf0c292ef76",
    "train-00007-of-00010.parquet": "fdeb213b5c2d0bb50f1081ae48b7ce4fa38147efe623de5d6d836a32f4044dad",
    "train-00008-of-00010.parquet": "d8e5b1417f0364312896d259efbea0600c1fac22eacb2f71187c5b0e9704f388",
    "train-00009-of-00010.parquet": "67639c6620cabce348e91c2cc331c4877307ed1a5d8fbd1c0d7dcca561cea8df",
}


def normalize_answer(value: str) -> str:
    value = value.strip().casefold().strip("$")
    return re.sub(r"\s+", "", value)


def audit(data_dir: Path, source_dir: Path) -> dict:
    math_path = data_dir / "math500.jsonl"
    mirror_path = data_dir / "deepmath-train.parquet"
    if sha256_file(math_path) != EXPECTED_MATH500_SHA256:
        raise ValueError("MATH-500 checksum mismatch")
    if sha256_file(mirror_path) != EXPECTED_DEEPMATH_SHA256:
        raise ValueError("TRL DeepMath mirror checksum mismatch")

    math_rows = [json.loads(line) for line in math_path.read_text().splitlines() if line.strip()]
    math_prompts = {normalize_prompt(row["problem"]) for row in math_rows}
    mirror_rows = parquet.read_table(mirror_path, columns=["prompt", "solution"]).to_pylist()
    eligible, _ = deduplicate_train_rows(mirror_rows)
    split = select_splits([], eligible)
    selected = {
        hashlib.sha256(row["prompt_key"].encode()).hexdigest(): row
        for row in eligible
        if hashlib.sha256(row["prompt_key"].encode()).hexdigest()
        in set(split["base_gate_ids"] + split["train_ids"])
    }
    if len(selected) != 288:
        raise ValueError(f"expected 288 selected mirror rows, found {len(selected)}")

    selected_ids = set(selected)
    source_prompts: set[str] = set()
    source_label_map: dict[str, list[str]] = {}
    total_rows = 0
    duplicate_rows = 0
    for filename, expected_hash in ORIGINAL_SOURCE_SHA256.items():
        path = source_dir / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"missing or hash-mismatched original source file: {filename}")
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=128, columns=["question", "final_answer"]
        ):
            for row in batch.to_pylist():
                total_rows += 1
                prompt_key = normalize_prompt(row["question"])
                if prompt_key in source_prompts:
                    duplicate_rows += 1
                source_prompts.add(prompt_key)
                selected_id = hashlib.sha256(prompt_key.encode()).hexdigest()
                if selected_id in selected_ids:
                    source_label_map.setdefault(selected_id, []).append(row["final_answer"])

    overlap = math_prompts & source_prompts
    if overlap:
        raise ValueError(f"found {len(overlap)} exact normalized MATH-500 overlaps")
    missing = selected_ids - source_label_map.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected prompts do not map to original source")

    label_matches = 0
    duplicate_source_selected = 0
    for selected_id, labels in source_label_map.items():
        if len(labels) > 1:
            duplicate_source_selected += 1
        wanted = normalize_answer(selected[selected_id]["solution"])
        if not all(normalize_answer(label) == wanted for label in labels):
            raise ValueError(f"TRL/original answer labels differ for selected prompt {selected_id}")
        label_matches += 1

    return {
        "status": "pass",
        "math500_exact_prompt_overlap_with_full_original_source": len(overlap),
        "math500_rows": len(math_rows),
        "original_source_revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "original_source_rows": total_rows,
        "original_source_unique_normalized_prompts": len(source_prompts),
        "original_source_duplicate_excess_rows": duplicate_rows,
        "selected_mirror_prompts_mapped": len(source_label_map),
        "selected_mirror_labels_matched_after_wrapper_whitespace_normalization": label_matches,
        "selected_prompts_with_duplicate_original_source_rows": duplicate_source_selected,
        "source_files_sha256": ORIGINAL_SOURCE_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/math-grpo-cpu-first-2026-10-11/data"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/math-grpo-cpu-first-2026-10-11/data/deepmath-original"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_dir, args.source_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
