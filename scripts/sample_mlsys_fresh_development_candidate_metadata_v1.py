#!/usr/bin/env python3
"""Select a deterministic, per-repository metadata-only manual-review sample."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_metadata_sample_v1.json"
SOURCE_PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_screen_v4.json"
SOURCE_SCREEN_PATH = ROOT / "results/mlsys-fresh-development-candidate-screen-v4/screen.json.gz"
SOURCE_MANIFEST_PATH = ROOT / "results/mlsys-fresh-development-candidate-screen-v4/evidence-manifest.json"
SOURCE_COMPRESSED_MANIFEST_PATH = ROOT / "results/mlsys-fresh-development-candidate-screen-v4/compressed-evidence-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-metadata-sample-v1":
        raise ValueError("unexpected metadata sample protocol ID")
    if protocol.get("sampler_sha256") != sha256_file(Path(__file__)):
        raise ValueError("sampler source differs from the frozen protocol")
    if protocol.get("source_screen_protocol_sha256") != sha256_file(SOURCE_PROTOCOL_PATH):
        raise ValueError("source screen protocol differs from the frozen sample protocol")
    if protocol.get("source_screen_gzip_sha256") != sha256_file(SOURCE_SCREEN_PATH):
        raise ValueError("compressed source screen differs from the frozen sample protocol")
    if protocol.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise ValueError("source evidence manifest differs from the frozen sample protocol")
    if protocol.get("source_compressed_manifest_sha256") != sha256_file(SOURCE_COMPRESSED_MANIFEST_PATH):
        raise ValueError("source compression manifest differs from the frozen sample protocol")
    return protocol


def load_screen(protocol: dict[str, Any]) -> dict[str, Any]:
    with gzip.open(SOURCE_SCREEN_PATH, "rb") as stream:
        raw = stream.read()
    if sha256_bytes(raw) != protocol.get("source_screen_sha256"):
        raise ValueError("decompressed source screen differs from the frozen sample protocol")
    payload = json.loads(raw.decode("utf-8"))
    rows = payload.get("candidates")
    if type(rows) is not list or len(rows) != protocol.get("source_candidate_count"):
        raise ValueError("source screen candidate count differs from the frozen sample protocol")
    if payload.get("provisional_count") != protocol.get("provisional_candidate_count"):
        raise ValueError("source screen provisional count differs from the frozen sample protocol")
    return payload


def select_sample(payload: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload["candidates"]:
        if row.get("coarse_screen") != "provisional_for_manual_review":
            continue
        if (
            row.get("qualification_status") != "UNQUALIFIED"
            or row.get("split_assignment") != "UNASSIGNED"
            or row.get("task_family_overlap") != "NOT_ASSESSED"
        ):
            raise ValueError("source screen promoted a metadata candidate beyond its frozen status")
        repo = row.get("repository")
        if type(repo) is not str:
            raise ValueError("provisional candidate has no repository identity")
        grouped.setdefault(repo, []).append(row)
    sample: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seed = protocol["selection"]["sha256_seed"]
    sample_size = protocol["selection"]["sample_size_per_repository"]
    for repo in protocol["repositories"]:
        ranked: list[tuple[str, dict[str, Any]]] = []
        for row in grouped.get(repo, []):
            number = row.get("number")
            if type(number) is not int:
                raise ValueError("provisional candidate has an invalid PR number")
            rank = hashlib.sha256(f"{seed}|{repo}#{number}".encode("utf-8")).hexdigest()
            ranked.append((rank, row))
        ranked.sort(key=lambda entry: entry[0])
        chosen = ranked[:sample_size]
        counts[repo] = len(chosen)
        for rank, row in chosen:
            sample.append({**row, "sample_rank_sha256": rank})
    sample.sort(key=lambda row: (row["repository"], row["sample_rank_sha256"]))
    return sample, counts


def run(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    payload = load_screen(protocol)
    sample, counts = select_sample(payload, protocol)
    raw = (json.dumps({
        "schema_version": 1,
        "sample_id": "mlsys-fresh-development-candidate-metadata-sample-v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_screen_sha256": protocol["source_screen_sha256"],
        "source_screen_gzip_sha256": protocol["source_screen_gzip_sha256"],
        "source_candidate_count": protocol["source_candidate_count"],
        "source_provisional_count": protocol["provisional_candidate_count"],
        "sample_size_per_repository": protocol["selection"]["sample_size_per_repository"],
        "selection_seed": protocol["selection"]["sha256_seed"],
        "sample_count": len(sample),
        "sample_counts_by_repository": counts,
        "task_qualification_performed": False,
        "split_assignment_performed": False,
        "model_or_agent_inference_authorized": False,
        "training_authorized": False,
        "candidates": sample,
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sample_path = output_dir / "review-sample.json"
    sample_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "sampler_sha256": sha256_file(Path(__file__)),
        "source_screen_sha256": protocol["source_screen_sha256"],
        "source_screen_gzip_sha256": protocol["source_screen_gzip_sha256"],
        "files": {"review-sample.json": {"bytes": len(raw), "sha256": sha256_bytes(raw)}},
    }
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output_dir),
        "sample_count": len(sample),
        "sample_counts_by_repository": counts,
        "sample_sha256": manifest["files"]["review-sample.json"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(load_protocol(), args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
