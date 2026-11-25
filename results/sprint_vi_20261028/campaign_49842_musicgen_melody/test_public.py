"""Public incremental-decoding regression check for this Development task."""
from __future__ import annotations

import json
import math
import secrets
import subprocess
import sys
from pathlib import Path

TARGET = "src/transformers/models/musicgen_melody/modeling_musicgen_melody.py"
WORKER = r'''import ast, json, math, sys, torch, torch.nn as nn
source_path, payload_text = sys.argv[1], sys.argv[2]
payload = json.loads(payload_text)
source = open(source_path, encoding="utf-8").read()
tree = ast.parse(source, filename=source_path)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MusicgenMelodySinusoidalPositionalEmbedding")
namespace = {"math": math, "torch": torch, "nn": nn}
exec(compile(ast.get_source_segment(source, node), source_path, "exec"), namespace)
cls = namespace["MusicgenMelodySinusoidalPositionalEmbedding"]
try:
    module = cls(payload["initial_positions"], payload["embedding_dim"])
    inputs = torch.zeros((1, payload["seq_len"], payload["embedding_dim"]), dtype=torch.float32)
    result = module(inputs, past_key_values_length=payload["past"])
    record = {"nonce": payload["nonce"], "shape": list(result.shape), "values": result.tolist(), "weight_rows": int(module.weights.shape[0]), "error": None}
except Exception as exc:
    record = {"nonce": payload["nonce"], "error": type(exc).__name__ + ": " + str(exc)[:180]}
print("VARE_CHILD_RESULT:" + json.dumps(record, separators=(",", ":")))
'''


def _reference(case):
    dim = case["embedding_dim"]
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    inv = [math.exp(-scale * i) for i in range(half)]
    rows = []
    for position in range(case["past"], case["past"] + case["seq_len"]):
        row = [math.cos(position * value) for value in inv]
        row.extend(math.sin(position * value) for value in inv)
        if dim % 2:
            row.append(0.0)
        rows.append(row)
    return rows


def _run_case(case):
    source = Path(__file__).resolve().parent / TARGET
    nonce = secrets.token_hex(16)
    payload = dict(case, nonce=nonce)
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, str(source), json.dumps(payload, separators=(",", ":"))],
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"worker exit {proc.returncode}: {proc.stdout[-300:]} {proc.stderr[-300:]}"
    lines = [line[len("VARE_CHILD_RESULT:"):]
             for line in proc.stdout.splitlines() if line.startswith("VARE_CHILD_RESULT:")]
    assert len(lines) == 1, f"expected one structured worker result, got {len(lines)}"
    record = json.loads(lines[0])
    assert record.get("nonce") == nonce, "worker result nonce mismatch"
    assert record.get("error") is None, record.get("error")
    assert record.get("shape") == [case["seq_len"], case["embedding_dim"]], record
    assert record.get("weight_rows", 0) >= case["past"] + case["seq_len"], record
    expected = _reference(case)
    actual = record["values"]
    assert len(actual) == len(expected)
    for got_row, want_row in zip(actual, expected):
        assert len(got_row) == len(want_row)
        assert max(abs(float(g) - w) for g, w in zip(got_row, want_row)) < 2e-5


def test_long_incremental_decode_does_not_index_past_position_table():
    _run_case({"initial_positions": 4, "embedding_dim": 8, "seq_len": 1, "past": 7})
